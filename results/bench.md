Causal attention, forward + backward, L = 4096, 8 heads × 64, float32, one thread per rank. Single process (SDPA): 1.19 s.

| P | method | slowest rank (s) | time max/min | work max/min | MB sent per rank | theory MB | max error vs full |
|---|---|---|---|---|---|---|---|
| 2 | ring, contiguous | 1.93 | 1.00 | 2.33 | 40.0 | 40.0 | 7.2e-07 |
| 2 | ring, zigzag | 1.42 | 1.02 | 1.00 | 40.0 | 40.0 | 7.2e-07 |
| 2 | ulysses | 0.85 | 1.00 | – | 16.0 | 16.0 | 0.0e+00 |
| 4 | ring, contiguous | 1.93 | 1.11 | 5.00 | 44.1 | 44.0 | 7.2e-07 |
| 4 | ring, zigzag | 1.59 | 1.00 | 1.00 | 44.1 | 44.0 | 7.2e-07 |
| 4 | ulysses | 0.72 | 1.01 | – | 12.0 | 12.0 | 0.0e+00 |
