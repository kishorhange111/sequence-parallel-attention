"""Measured cost of context parallelism on CPU processes (gloo).

    python -m seqpar.bench --out results

For P ranks, each a separate process with one thread, this runs forward and backward of
ring attention (contiguous and zigzag causal layouts) and Ulysses. It records:

- **per-rank wall time.** With contiguous causal shards the last rank does the most work,
  and the slowest rank sets the step time. Zigzag should flatten this.
- **bytes sent per rank**, counted by wrapping ``dist.isend``, against the closed forms.
  Ring forward plus backward sends K, V (and in the backward dK, dV) P − 1 or P times.
  Ulysses sends four all-to-alls forward and four backward, each (P − 1)/P of a local
  shard.
- **agreement** of the gathered float32 output with full attention on one process.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from . import attention as A
from .attention import causal_work, ring_attention, shard_positions, ulysses_attention


def _inputs(B, H, L, D, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(B, H, L, D, generator=g) for _ in range(4)]


def _worker(rank, P, init, cfg, out_dir):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=P)
    sent = [0]
    orig = dist.isend

    def counting_isend(t, dst, group=None, tag=0):
        sent[0] += t.numel() * t.element_size()
        return orig(t, dst, group, tag)

    A.dist.isend = counting_isend  # the module calls dist.isend through its own reference
    try:
        B, H, L, D = cfg["B"], cfg["H"], cfg["L"], cfg["D"]
        q, k, v, do = _inputs(B, H, L, D)
        out = {}
        for name, method, layout in (
            ("ring, contiguous", "ring", "contiguous"),
            ("ring, zigzag", "ring", "zigzag"),
            ("ulysses", "ulysses", "contiguous"),
        ):
            pos = shard_positions(L, P, rank, layout)
            ql, kl, vl = (t[:, :, pos].clone().requires_grad_() for t in (q, k, v))
            times = []
            for rep in range(cfg["reps"] + 1):
                for t in (ql, kl, vl):
                    t.grad = None
                sent[0] = 0
                dist.barrier()
                t0 = time.perf_counter()
                if method == "ring":
                    o = ring_attention(ql, kl, vl, pos, pos, True)
                else:
                    o = ulysses_attention(ql, kl, vl, True)
                o.backward(do[:, :, pos])
                el = time.perf_counter() - t0
                if rep:  # the first repetition is a warm-up
                    times.append(el)
            out[name] = {
                "seconds": sorted(times)[len(times) // 2],
                "bytes_sent": sent[0],
                "o": o.detach(),
                "pos": pos,
            }
        torch.save(out, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        A.dist.isend = orig
        dist.destroy_process_group()


def run(P, cfg):
    with tempfile.TemporaryDirectory() as d:
        init = "file:///" + os.path.abspath(os.path.join(d, "rdzv")).replace("\\", "/").lstrip("/")
        mp.spawn(_worker, args=(P, init, cfg, d), nprocs=P, join=True)
        return {r: torch.load(os.path.join(d, f"rank{r}.pt")) for r in range(P)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--P", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    cfg = {"B": 1, "H": a.H, "L": a.L, "D": a.D, "reps": a.reps}
    q, k, v, do = _inputs(1, a.H, a.L, a.D)
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    ref = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True)
    ref.backward(do)
    single = time.perf_counter() - t0
    rows = []
    shard = a.H * (a.L // 1) * a.D * 4  # bytes of one full-sequence tensor, all heads
    for P in a.P:
        res = run(P, cfg)
        local = shard // P
        theory = {
            # forward: K, V hop P−1 times; backward: K, V, dK, dV hop P times
            "ring, contiguous": (2 * (P - 1) + 4 * P) * local,
            "ring, zigzag": (2 * (P - 1) + 4 * P) * local,
            # 4 all-to-alls each way (q, k, v, o forward; their adjoints backward)
            "ulysses": 8 * local * (P - 1) // P,
        }
        work = {
            "ring, contiguous": causal_work(a.L, P, "contiguous"),
            "ring, zigzag": causal_work(a.L, P, "zigzag"),
        }
        for name in ("ring, contiguous", "ring, zigzag", "ulysses"):
            secs = [res[r][name]["seconds"] for r in range(P)]
            sent = [res[r][name]["bytes_sent"] for r in range(P)]
            o = torch.empty_like(ref)
            for r in range(P):
                o[:, :, res[r][name]["pos"]] = res[r][name]["o"]
            w = work.get(name)
            rows.append(
                {
                    "P": P,
                    "method": name,
                    "seconds_per_rank": secs,
                    "slowest_s": max(secs),
                    "time_imbalance": max(secs) / min(secs),
                    "work_imbalance": (max(w) / min(w)) if w else None,
                    "bytes_sent_per_rank": sent,
                    "bytes_theory": theory[name],
                    "max_abs_err": float((o - ref.detach()).abs().max()),
                }
            )
            print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(a.out, "bench.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lines = [
        f"Causal attention, forward + backward, L = {a.L}, {a.H} heads × {a.D}, float32, one "
        f"thread per rank. Single process (SDPA): {single:.2f} s.",
        "",
        "| P | method | slowest rank (s) | time max/min | work max/min | MB sent per rank "
        "| theory MB | max error vs full |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        wi = f"{r['work_imbalance']:.2f}" if r["work_imbalance"] else "–"
        lines.append(
            f"| {r['P']} | {r['method']} | {r['slowest_s']:.2f} | {r['time_imbalance']:.2f} | {wi} "
            f"| {max(r['bytes_sent_per_rank']) / 2**20:.1f} | {r['bytes_theory'] / 2**20:.1f} "
            f"| {r['max_abs_err']:.1e} |"
        )
    with open(os.path.join(a.out, "bench.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
