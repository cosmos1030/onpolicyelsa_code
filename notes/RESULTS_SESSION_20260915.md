# Session 2026-09-15 — what is settled

Numbers below were read from the run's own log or wandb summary, not from a
notification. Jobs that finished but whose eval crashed are marked as such.

## 1. Systems: does a pruned Qwen3-4B actually run faster?

The control throughout is the same weights executed two ways, so generation
length and accuracy cannot enter the number.

**2:4 semi-structured — no, in any configuration** (A100-80GB, BF16).
The kernel reaches 1.19x at prefill (M>=512) but 0.58x at M=1; at model level,
under CUDA graph, prefill is 0.94x and decode 0.76-0.80x. Eager is far worse
(0.20x) because PyTorch's semi-structured path costs ~0.7ms of Python dispatch
per Linear and this model has 252 of them. `torch.compile` does not remove that
(0.20x); only a captured graph does. Batch decode does not run at all --
`SparseSemiStructuredTensor` has no `expand`.

**unstructured 80% + MACKO-SpMV — yes, at batch 1** (L40S, fp16, vLLM).

| output length | dense ms/tok | MACKO ms/tok | speedup |
|---|---|---|---|
| 512 | 12.86 | 6.03 | 2.135x |
| 2048 | 12.99 | 6.14 | 2.116x |
| 4096 | 13.20 | 6.35 | 2.081x |
| 8192 | 13.67 | 6.78 | 2.016x |

At 5000 tokens that is 66.9s vs 32.5s -- 34 seconds saved per answer.
Weights 8.04 -> 1.92 GB (4.2x). The SpMV kernel alone is 2.50x on A6000 and
2.58x on L40S; MLP matrices reach 5-6x while attention matrices only 1.1-2.0x,
and ~78% of Qwen3-4B's parameters are MLP.

Integrated into vLLM without patching vLLM or macko_spmv, via
`@register_quantization_config` (`elsa/scripts/systems/macko_vllm/`).
Compression runs in `process_weights_after_loading`, i.e. after vLLM has
sharded and fused, so QKV and gate/up are compressed as the single matrices
they have become -- 144 Linears, not 252. Dense numbers reproduce the
standalone run to the decimal, which is the check that the plugin does not
disturb the path it does not claim.

**What this costs.** MACKO is batch-1 only (`apply()` falls back to dense above
one token per step). Dense batching buys 18.6x throughput at 1.72x latency
(B=32, 2048 tokens), so this is a single-stream latency result and must be
written as one.

**Two caveats that have to travel with the numbers.** The checkpoint measured
is ALPS s80, and one-shot pruning at 80% is a dead model: MATH-500 pass@1 of
0.4-1.6% with 100% of generations hitting the token cap (`prune_eval` runs
`obvhpy0l`, `3uiz7qq3`). Speed depends only on mask density, so a usable s80
would measure the same -- but producing one is SCOUT's job, and that run is
still training. Separately, peak memory went UP (9.34 -> 10.08 GB) because the
prefill fallback keeps the dense copy; the 4.2x is a weights figure.

## 2. OPD's contribution grows with sparsity, and is negative at 50%

Each row matched to its own sparsity's published SCOUT row in delta, lr,
pgd_interval (8), mask_interval, rollout_interval and rollout length; only the
loss weights differ (0.33/0.33/0.33 -> 0.5/0.5/0).

| Qwen3-4B | delta | lr | w/o OPD | SCOUT | OPD contributes |
|---|---|---|---|---|---|
| s50 | 0.01 | 5e-5 | 63.41 | 62.06 | **-1.35** |
| s60 | 0.02 | 5e-5 | 53.91 | 55.35 | +1.44 |
| s70 | 0.02 | 1e-4 | 43.70 | 45.58 | **+1.88** |

At s70 the gain is concentrated in MATH (+4.8) and LCB (+3.7) and is negative
on GSM8K (-2.6): on-policy data helps the long chains, not short arithmetic.

This is the direct evidence for the paper's own trend -- SCOUT's margin over
the recovery-matched baseline is +2.2 / +2.4 / +4.7 across the same sparsities.

**Renormalising mattered.** The earlier 0.33/0.33/0 runs scored 59.41 at s50;
0.5/0.5/0 scores 63.41. Dropping OPD without renormalising also shrinks the
total loss to 2/3, so those runs confounded "no OPD" with "lower effective lr".

## 3. Qwen3-1.7B s70, one term removed at a time

delta=0.02, lr=1e-4, pgd_interval=8 throughout; only the weights differ.

| arm | NTP/KD/OPD | MATH | LCB | GPQA | GSM8K | IFEval | avg5 |
|---|---|---|---|---|---|---|---|
| SCOUT | 0.33/0.33/0.33 | 43.6 | 1.5 | 24.7 | 53.8 | 25.9 | **29.89** |
| w/o OPD | 0.5/0.5/0 | 39.0 | 0.4 | 23.2 | 50.3 | 22.6 | 27.08 (-2.81) |
| OPD only | 0/0/1 | 28.2 | 0.0 | 19.2 | 49.2 | 16.3 | 22.57 (-7.32) |

Removing the fixed-data anchor costs 2.6x what removing OPD costs. On-policy
data alone does not recover a 70%-sparse model; OPD is a term that adds +2.81
on top of NTP+KD, not one that stands on its own.

## 4. Milestone trajectories (ALPS -> recovery, avg5)

**Qwen3-4B s70**, scored A100 tp=1 except the 2048 endpoints (B200 tp=4):

| step | 2term (0.5/0.5/0) | 3term (0.33x3) | diff |
|---|---|---|---|
| 512 | 37.25 | 36.66 | -0.59 |
| 1024 | 39.29 | 39.61 | +0.32 |
| 1536 | 38.88 | 40.84 | +1.96 |
| 2048 | 39.12 | 40.93 | +1.81 |

The two arms differ in NTP/KD weight as well as in OPD, so this is not a clean
OPD ablation -- see section 2 for that.

**Qwen3-1.7B s70**, all four points on one path (A100 tp=1):

| step | 2term lr1e-4 | 3term lr1e-4 | 2term lr5e-5 | 3term lr5e-5 |
|---|---|---|---|---|
| 512 | 17.39 | 21.11 | 9.52 | 21.23 |
| 1024 | 19.27 | 22.57 | 17.42 | 20.90 |
| 1536 | 20.07 | 23.26 | 18.51 | 21.27 |
| 2048 | 20.56 | 23.31 | 17.62 | 20.11 |

OPD wins at every step and both learning rates (+2.5 to +3.8), unlike 4B where
it starts negative. lr 1e-4 beats 5e-5 on both arms, and the published 1.7B
ALPS+recovery row uses 5e-5 (21.76) -- our lr 1e-4 3-term reaches 23.31, so the
published baseline may not be its own optimum. LCB is 0.00 in all 20 points:
1.7B at 70% has lost coding entirely, so avg5 is effectively a 4-task mean.

## Still running
SCOUT s80 (1.7B 930596, 4B 935937), 1.7B w/o OPD at s50/s60, w/o NTP
(0/0.5/0.5) at s70 for both sizes, the noopd55 re-encode for both figure axes,
and the 1.7B-dense / 4B-s70 speed points.
