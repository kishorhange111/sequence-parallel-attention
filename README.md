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

## Tests

There are 8, on CPU, using `gloo` with 4 processes and float64. They check:
- ring attention (non-causal, causal contiguous, causal zigzag) and Ulysses (causal and
  non-causal): **output, dQ, dK and dV equal full attention to 1e-10** on every rank;
- merging two key blocks via LSE equals one softmax, including rows that see nothing in
  one of the blocks;
- both layouts partition the sequence;
- zigzag work is exactly balanced and contiguous work is imbalanced by (4P − 1)/3.

The collectives are written with `isend` / `irecv` so they run on `gloo`. On GPUs the
all-to-all would be NCCL's `all_to_all_single`.

```
pip install -e ".[dev]" && pytest -q
```
