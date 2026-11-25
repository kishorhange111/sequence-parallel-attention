import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from seqpar.attention import (
    block_forward,
    causal_work,
    merge,
    ring_attention,
    shard_positions,
    ulysses_attention,
)

P, B, H, L, D = 4, 2, 4, 32, 8


def _full(seed=0):
    g = torch.Generator().manual_seed(seed)
    q, k, v, do = (torch.randn(B, H, L, D, generator=g, dtype=torch.float64) for _ in range(4))
    return q, k, v, do


def _reference(causal):
    q, k, v, do = _full()
    q, k, v = (t.requires_grad_() for t in (q, k, v))
    o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def _worker(rank, init, method, layout, causal):
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=P)
    try:
        q, k, v, do = _full()
        pos = shard_positions(L, P, rank, layout)
        ql, kl, vl = (t[:, :, pos].clone().requires_grad_() for t in (q, k, v))
        if method == "ring":
            o = ring_attention(ql, kl, vl, pos, pos, causal)
        else:
            o = ulysses_attention(ql, kl, vl, causal)
        o.backward(do[:, :, pos])
        ro, rq, rk, rv = _reference(causal)
        for got, ref in ((o, ro), (ql.grad, rq), (kl.grad, rk), (vl.grad, rv)):
            torch.testing.assert_close(got, ref[:, :, pos], atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "method,layout,causal",
    [
        ("ring", "contiguous", False),
        ("ring", "contiguous", True),
        ("ring", "zigzag", True),
        ("ulysses", "contiguous", True),
        ("ulysses", "contiguous", False),
    ],
)
def test_distributed_forward_and_backward_match_full_attention(tmp_path, method, layout, causal):
    init = "file:///" + os.path.abspath(tmp_path / "rdzv").replace("\\", "/").lstrip("/")
    mp.spawn(_worker, args=(init, method, layout, causal), nprocs=P, join=True)


def test_merging_two_halves_equals_softmax_over_all_keys():
    q, k, v, _ = _full()
    pos = torch.arange(L)
    o1, l1 = block_forward(q, k[:, :, :12], v[:, :, :12], pos, pos[:12], D**-0.5, True)
    o2, l2 = block_forward(q, k[:, :, 12:], v[:, :, 12:], pos, pos[12:], D**-0.5, True)
    o, lse = merge(o1, l1, o2, l2)  # rows 0..11 see nothing in the second block
    torch.testing.assert_close(o, F.scaled_dot_product_attention(q, k, v, is_causal=True))
    s = (q @ k.transpose(-1, -2)) * D**-0.5
    s = s.masked_fill(torch.ones(L, L, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, -1))


def test_shards_partition_the_sequence():
    for layout in ("contiguous", "zigzag"):
        allpos = torch.cat([shard_positions(L, P, r, layout) for r in range(P)])
        assert sorted(allpos.tolist()) == list(range(L))
    assert shard_positions(16, 2, 0, "zigzag").tolist() == [0, 1, 2, 3, 12, 13, 14, 15]


def test_zigzag_balances_causal_work():
    contiguous, zigzag = causal_work(1024, 8, "contiguous"), causal_work(1024, 8, "zigzag")
    assert len(set(zigzag)) == 1  # every rank does the same work
    assert max(contiguous) / min(contiguous) == pytest.approx(31 / 3)  # (4P−1) / 3 units
    # The critical path (slowest rank) shrinks by almost 2×.
    assert max(contiguous) / max(zigzag) == pytest.approx(31 / 17)
