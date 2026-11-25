"""Context-parallel attention: the sequence is split across P ranks.

**Ring attention** (Liu et al. 2023; the context-parallel scheme in Megatron and
PyTorch). Each rank keeps its query shard and passes its (K, V) shard around a ring. After
P steps every query has seen every key. Partial results merge exactly with the same
online-softmax rule as FlashAttention, carrying a log-sum-exp per row, so no rank ever holds
the full L × L scores or the full K, V.

The backward pass is written by hand. Each rank recomputes its block probabilities from
the saved LSE and accumulates dQ locally. dK and dV travel around the ring *with* their
K, V shard and arrive back at the owner after P hops.

**Causal load balance.** With contiguous shards, rank r attends to r + 1 of the P key
shards, so the last rank does about twice the average work. Zigzag sharding gives rank r
chunks r and 2P − 1 − r of a 2P-way split, which makes every rank's work equal. Masks are
built from each token's *global* position, so one code path handles both layouts.

**Ulysses** (DeepSpeed). An all-to-all turns sequence shards into head shards: every rank
holds the full sequence for H/P heads. It runs ordinary attention, then a second
all-to-all turns the result back. Each rank sends about 4 · L/P · d per layer, which stays
constant as P grows with L. Ring attention sends about 2 · (P − 1) · L/P · d, but it can
overlap that with compute and has no H ≥ P limit.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

NEG = float("-inf")


# ------------------------------------------------------------------- one block, exact merge
def block_forward(q, k, v, pos_q, pos_k, scale, causal):
    """Attention of q over one key block -> (o, lse). Rows with no visible key get o = 0
    and lse = −inf."""
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        s = s.masked_fill(pos_q[:, None] < pos_k[None, :], NEG)
    lse = torch.logsumexp(s, -1)
    safe = torch.where(torch.isinf(lse), torch.zeros_like(lse), lse)
    p = torch.exp(s - safe[..., None])
    return p @ v, lse


def merge(o, lse, o_b, lse_b):
    new = torch.logaddexp(lse, lse_b)
    safe = torch.where(torch.isinf(new), torch.zeros_like(new), new)
    w, w_b = torch.exp(lse - safe), torch.exp(lse_b - safe)
    return o * w[..., None] + o_b * w_b[..., None], new


def visible(pos_q, pos_k, causal) -> bool:
    return (not causal) or bool(pos_q.max() >= pos_k.min())


# ------------------------------------------------------------------------- communication
def ring_shift(tensors, group=None):
    """Send to rank + 1, receive from rank − 1, for each tensor."""
    r, P = dist.get_rank(group), dist.get_world_size(group)
    out = [torch.empty_like(t) for t in tensors]
    ops = []
    for t, o in zip(tensors, out, strict=True):
        ops.append(dist.isend(t.contiguous(), (r + 1) % P, group))
        ops.append(dist.irecv(o, (r - 1) % P, group))
    for op in ops:
        op.wait()
    return out


def all_to_all(chunks, group=None):
    """``chunks[j]`` goes to rank j; returns the list received from every rank. This uses
    point-to-point ops so it works on gloo; NCCL would use ``all_to_all_single``."""
    r, P = dist.get_rank(group), dist.get_world_size(group)
    out = [torch.empty_like(c) for c in chunks]
    out[r] = chunks[r].clone()
    ops = []
    for j in range(P):
        if j != r:
            ops.append(dist.isend(chunks[j].contiguous(), j, group))
            ops.append(dist.irecv(out[j], j, group))
    for op in ops:
        op.wait()
    return out


# ---------------------------------------------------------------------------------- ring
class _Ring(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, pos_q, pos_k, causal, group):
        P = dist.get_world_size(group)
        scale = q.shape[-1] ** -0.5
        o = torch.zeros_like(q)
        lse = torch.full(q.shape[:-1], NEG, dtype=q.dtype, device=q.device)
        kc, vc, pk = k, v, pos_k
        for step in range(P):
            for qs, ks in _pairs(pos_q, pk, causal):
                o_b, lse_b = block_forward(
                    q[..., qs, :], kc[..., ks, :], vc[..., ks, :], pos_q[qs], pk[ks], scale, causal
                )
                o[..., qs, :], lse[..., qs] = merge(o[..., qs, :], lse[..., qs], o_b, lse_b)
            if step < P - 1:
                kc, vc, pk = ring_shift([kc, vc, pk], group)
        ctx.save_for_backward(q, k, v, pos_q, pos_k, o, lse)
        ctx.causal, ctx.group, ctx.scale = causal, group, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, pos_q, pos_k, o, lse = ctx.saved_tensors
        group, causal, scale = ctx.group, ctx.causal, ctx.scale
        P = dist.get_world_size(group)
        D = (do * o).sum(-1)  # rowsum(dO ∘ O)
        dq = torch.zeros_like(q)
        kc, vc, pk = k, v, pos_k
        dk, dv = torch.zeros_like(k), torch.zeros_like(v)
        for _ in range(P):
            for qs, ks in _pairs(pos_q, pk, causal):
                qb, kb, vb, dob = q[..., qs, :], kc[..., ks, :], vc[..., ks, :], do[..., qs, :]
                s = (qb @ kb.transpose(-1, -2)) * scale
                if causal:
                    s = s.masked_fill(pos_q[qs][:, None] < pk[ks][None, :], NEG)
                p = torch.exp(s - lse[..., qs, None])  # exact probabilities from saved LSE
                dv[..., ks, :] += p.transpose(-1, -2) @ dob
                ds = p * (dob @ vb.transpose(-1, -2) - D[..., qs, None])
                dq[..., qs, :] += (ds @ kb) * scale
                dk[..., ks, :] += (ds.transpose(-1, -2) @ qb) * scale
            # K, V and their gradient accumulators move on together; after P hops they are
            # back where they started.
            kc, vc, pk, dk, dv = ring_shift([kc, vc, pk, dk, dv], group)
        return dq, dk, dv, None, None, None, None


def _pairs(pos_q, pos_k, causal, parts: int = 2):
    """(query slice, key slice) sub-blocks that have at least one visible pair. Splitting
    each shard in two lets zigzag shards skip their fully masked halves."""
    lq, lk = len(pos_q) // parts, len(pos_k) // parts
    for i in range(parts):
        qs = slice(i * lq, (i + 1) * lq)
        for j in range(parts):
            ks = slice(j * lk, (j + 1) * lk)
            if visible(pos_q[qs], pos_k[ks], causal):
                yield qs, ks


def ring_attention(q, k, v, pos_q, pos_k, causal=True, group=None):
    """``q, k, v (b, h, l_local, d)`` with global token positions ``pos_q, pos_k (l_local,)``."""
    return _Ring.apply(q, k, v, pos_q, pos_k, causal, group)


# ------------------------------------------------------------------------------ Ulysses
class _SeqToHead(torch.autograd.Function):
    """(b, H, L/P, d) sequence-sharded -> (b, H/P, L, d) head-sharded; its adjoint is the
    reverse all-to-all."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        P = dist.get_world_size(group)
        return torch.cat(all_to_all(list(x.chunk(P, 1)), group), 2)

    @staticmethod
    def backward(ctx, g):
        return _HeadToSeq.apply(g, ctx.group), None


class _HeadToSeq(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        P = dist.get_world_size(group)
        return torch.cat(all_to_all(list(x.chunk(P, 2)), group), 1)

    @staticmethod
    def backward(ctx, g):
        return _SeqToHead.apply(g, ctx.group), None


def ulysses_attention(q, k, v, causal=True, group=None):
    """Contiguous sequence shards ``(b, H, L/P, d)``; H must be divisible by P."""
    P = dist.get_world_size(group)
    if q.shape[1] % P:
        raise ValueError(f"Ulysses needs heads ({q.shape[1]}) divisible by ranks ({P})")
    qh, kh, vh = (_SeqToHead.apply(t, group) for t in (q, k, v))
    oh = torch.nn.functional.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return _HeadToSeq.apply(oh, group)


# ------------------------------------------------------------------------------ sharding
def shard_positions(L: int, P: int, rank: int, layout: str) -> torch.Tensor:
    if L % (2 * P):
        raise ValueError("L must be divisible by 2P")
    if layout == "contiguous":
        n = L // P
        return torch.arange(rank * n, (rank + 1) * n)
    if layout == "zigzag":
        c = L // (2 * P)
        a, b = rank, 2 * P - 1 - rank
        return torch.cat([torch.arange(a * c, (a + 1) * c), torch.arange(b * c, (b + 1) * c)])
    raise ValueError(layout)


def causal_work(L: int, P: int, layout: str) -> list[int]:
    """(query, key) score entries each rank actually computes in causal ring attention,
    counting the half-shard sub-blocks that are not skipped."""
    out = []
    for r in range(P):
        pq, n = shard_positions(L, P, r, layout), 0
        for j in range(P):
            pk = shard_positions(L, P, j, layout)
            n += sum((qs.stop - qs.start) * (ks.stop - ks.start) for qs, ks in _pairs(pq, pk, True))
        out.append(n)
    return out
