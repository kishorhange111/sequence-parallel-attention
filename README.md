# sequence-parallel-attention

[![CI](https://github.com/kishorhange111/sequence-parallel-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/sequence-parallel-attention/actions/workflows/ci.yml)

Context parallelism from scratch: two ways to split one attention computation over P
devices along the sequence. These are the techniques behind million-token training in
Megatron-LM, PyTorch context parallel and DeepSpeed. Everything is in
[`attention.py`](src/seqpar/attention.py) and checked **to 1e-10 against full attention,
forward and backward, on a real 4-process `torch.distributed` group**.

## Ring attention

Each rank keeps its query shard while the (K, V) shards travel around a ring. After P
steps every query has seen every key. Partial outputs merge exactly through a log-sum-exp
per row, the same rule FlashAttention uses across tiles. No rank ever holds the L × L
scores or the full K, V.

**Backward, by hand.** Each rank recomputes its block probabilities from the saved LSE and
uses `D = rowsum(dO ∘ O)`, then accumulates dQ locally. The dK and dV accumulators
**travel with their K, V shard** and land back at the owner after P hops, so gradient
memory stays O(L/P) as well.

**Causal load balance.** With contiguous shards, rank r sees r + 1 key shards and the last
rank is the bottleneck. **Zigzag** sharding gives rank r chunks r and 2P − 1 − r, and each
shard is processed as half-blocks so fully masked ones are skipped:

| P = 8 | min rank work | max rank work | critical path |
|---|---|---|---|
| contiguous | 3 units | 31 units | 1.00 |
| zigzag | 17 | 17 | **0.55** (1.82× faster) |

A unit is one (L/2P)² sub-block; [`causal_work`](src/seqpar/attention.py) computes these
counts, and the test pins them.

## Ulysses (DeepSpeed)

An all-to-all turns *sequence* shards `(b, H, L/P, d)` into *head* shards `(b, H/P, L, d)`.
Each rank then runs ordinary attention over the full sequence for its heads, and a second
all-to-all turns the result back. Its adjoint is the reverse all-to-all, written as a
pair of autograd functions.

| | data sent per rank per layer | overlap with compute | constraint |
|---|---|---|---|
| ring | 2(P − 1) · L/P · d (K and V around the ring) | yes (send the next shard while computing) | none |
| Ulysses | 4 · L/P · d · (P − 1)/P (q, k, v in and o out) | no | H divisible by P |

Ulysses' volume per rank stays flat as P grows with L, while ring's grows with P but
hides behind compute. Large systems often combine the two (2-D context parallelism).

## Measured: bytes on the wire and the cost of imbalance

[`bench.py`](src/seqpar/bench.py) runs forward plus backward of causal attention on P real
processes (gloo, one CPU thread each), counts every byte passed to `isend`, and times the
slowest rank. L = 4096, 8 heads × 64, float32
([results/bench.md](results/bench.md)). One process running PyTorch's fused SDPA takes
1.19 s for the same work.

| P | method | slowest rank (s) | MB sent per rank | closed form (MB) | max error vs full |
|---|---|---|---|---|---|
| 2 | ring, contiguous | 1.93 | 40.0 | 40.0 | 7.2e-07 |
| 2 | ring, zigzag | **1.42** | 40.0 | 40.0 | 7.2e-07 |
| 2 | Ulysses | 0.85 | 16.0 | 16.0 | 0 |
| 4 | ring, contiguous | 1.93 | 44.1 | 44.0 | 7.2e-07 |
| 4 | ring, zigzag | **1.59** | 44.1 | 44.0 | 7.2e-07 |
| 4 | Ulysses | 0.72 | 12.0 | 12.0 | 0 |

- **The byte counts match the closed forms.** Ring sends `(2(P − 1) + 4P) · L/P · H · d`
  floats forward plus backward (K, V for P − 1 hops, then K, V, dK, dV for P hops). Ulysses
  sends `8 · L/P · H · d · (P − 1)/P`. The extra 0.1 MB for ring is the int64 key positions
  that travel with each shard.
- **Zigzag is 1.36× faster at P = 2 and 1.21× at P = 4.** The ranks run in lockstep, so the
  imbalance shows up as waiting, and every rank finishes at about the same time. The
  slowest rank, which sets the step time, is what zigzag shortens.
- **Ulysses sends 2.5× (P = 2) to 3.7× (P = 4) fewer bytes than ring here**, and its
  volume per rank falls as P grows at fixed L. Ring's rises toward 6 · L · H · d.
- **Read the wall-clock numbers as relative only.** Ring's blocks are plain PyTorch ops in
  Python, while Ulysses calls fused SDPA locally, so the ring-against-Ulysses time gap is
  mostly kernel quality. Ring does not get faster from 2 to 4 ranks on this CPU. The likely
  cause is that smaller blocks mean more Python-level block calls, so per-block overhead
  and gloo copies dominate at this size. On GPUs with NCCL, the ring sends overlap with the block
  computation, which this benchmark does not try to do.

```
python -m seqpar.bench --P 2 4 --L 4096 --out results
```

## Tests

There are 9, on CPU, using `gloo` with 4 processes and float64. They check:
- ring attention (non-causal, causal contiguous, causal zigzag) and Ulysses (causal and
  non-causal): **output, dQ, dK and dV equal full attention to 1e-10** on every rank;
- merging two key blocks via LSE equals one softmax, including rows that see nothing in
  one of the blocks;
- both layouts partition the sequence;
- zigzag work is exactly balanced and contiguous work is imbalanced by (4P − 1)/3;
- the benchmark runs end to end (B = 1, H = 8) and its byte counts match the closed forms.

The collectives are written with `isend` / `irecv` so they run on `gloo`. On GPUs the
all-to-all would be NCCL's `all_to_all_single`.

```
pip install -e ".[dev]" && pytest -q
```
