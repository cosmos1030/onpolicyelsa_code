# Session 2026-09-09 (new container) -- results and state

Machine-readable results for EVERY run live in `RESULTS_ALL.tsv` /
`RESULTS_ALL.md`, regenerated every 10 min by `harvest_loop.sh` ->
`harvest_results.py`.  Those two files are derived purely from the raw training
logs, so a container death loses nothing: re-run `python harvest_results.py`
once after a restart and the whole table rebuilds.  This file carries only what
cannot be derived from a log automatically -- decisions, causes, and what to do
next.

Avg throughout = mean of ALL FIVE quick-profile benchmarks (MATH500, GPQA
diamond, IFEval prompt-strict, LCB codegen, GSM8K).  GPQA is included.

## 1. Ablation results completed today

### A3 (one-shot jump to target) -- large negative, as intended
| Arm | Avg | vs SCOUT |
|---|---|---|
| 4B S70 d=0.02 SCOUT (`s70_klb0.02`, warmup 256 = the production launcher) | 45.58 | -- |
| 4B S70 d=0.02 **jump** | 36.55 | **-9.03** |
| 4B S70 d=0.02 **jump + B1 frozen pool** | 33.14 | **-12.44** |
| 8B S70 d=0.02 **jump** | 42.86 | -- |

The reference is `s70_klb0.02` (45.58), NOT `s70_warm512` (45.27): the jump arms
were launched from the production launcher, which sets `lr_warmup_steps=256`, so
the warmup-256 run is the matched control.

Jumping straight to the target sparsity and then running the identical
self-KL-gated maintenance loop costs 9.0 Avg at 4B.  Stacking the frozen pool on
top costs another 3.4.  This is the cleanest evidence that the *gate*, not the
end state, is what the method contributes: same final sparsity, same steps, same
budget, same everything else.

### B1 (frozen rollout pool) -- gate >> refresh
| Arm | Avg |
|---|---|
| 8B S70 d=0.02 **B1 frozen pool (ro=4096)** | 48.92 |

Pool sizing was patched so the frozen arm draws the SAME total rollouts (16,384)
as the refreshed baseline -- the earlier version would have given it 256 unique
rollouts and confounded the comparison entirely.  The log confirms it:
`initial pool filled with 16384 rollouts`.

### A2 (cubic schedule instead of the KL gate), pace-matched
| Arm | Avg |
|---|---|
| 4B S50 rule=schedule (matched) | 62.09 |
| 4B S60 rule=schedule (matched) | 54.60 |
| 4B S60 rule=schedule (end_ratio=0.5) | 55.18 |
| 4B S70 rule=schedule (matched) | 42.75 |
| 4B S70 rule=schedule (end_ratio=0.5) | 40.45 |

## 2. 2:4 -- what was decided and why

Four candidate explanations for SCOUT underperforming at 2:4 were tested against
existing logs only.  Three are ruled out by measurement:

- **Local compensation ceiling.** Only ~1.6% of the pruned mass is absorbable by
  the surviving pair.
- **Permutation / grouping.** Regrouping the 4-groups at random gives the same
  within-group importance spread (0.662 vs 0.642), so contiguity is not the
  problem.  An earlier claim that it was is retracted.
- **Group-aware proposal.** 3rd/2nd saliency ratio = 0.75 caps the achievable
  upside.

A fourth measurement, on the step-512 8B 2:4 checkpoint: 100% of groups are in
exact 2:4 state, but the surviving pair equals the top-2 by Fisher in only
**24.8%** of groups, and in **75.4%** of groups an evicted coordinate is MORE
sensitive than a survivor (mean ratio 1.32).  The constraint, not the selection
rule, is doing the damage.

Standing gap to beat: 8B 2:4 ALPS+retrain (tok=512) = **55.64** vs SCOUT d=0.02
(pre-opkdfix) = **51.31**, i.e. **-4.33**, concentrated in MATH (-7.6) and LCB
(-7.1).  Extrapolating the opkdfix lift gives 52.3-54.2, still 1.4-3.3 short.

### Controls the running compensation arms will be compared against
Both already exist and are post-opkdfix-valid at 4B (world_size == 1 is
unchanged by opkdfix, so the 4B 2:4 numbers carry over):

| Baseline (no compensation) | Avg |
|---|---|
| 4B 2:4 d=0.01 (`queue_gmp_pgd_grow_to_target_4b_24/klb0.01`) | **47.23** |
| 4B 2:4 d=0.02 (`queue_gmp_pgd_grow_to_target_4b_24/klb0.02`) | **45.17** |
| 8B 2:4 d=0.01 (pre-opkdfix) | 50.15 |
| 8B 2:4 d=0.02 (pre-opkdfix) | 51.31 |

Note the delta ordering FLIPS with scale: at 4B d=0.01 > d=0.02, at 8B
d=0.02 > d=0.01.  Both 4B compensation arms are therefore being run, so the
comparison is not read off a single cell.

**Decision: 2:4 stays out of the paper's main claims.**  The compensation work
below is appendix material only, per the user's explicit scope call.

## 3. Local survivor compensation -- implemented, measured, REJECTED

`--gmp_pgd_nm_compensate` applies `delta_S = C_SS^-1 C_SD w_D` to the two
survivors of every group that loses a coordinate, using a per-group 4x4 input
covariance captured from the same calibration batch the KL screen uses.  The
math is right -- verified on CPU against the closed form, max
|closed-form - impl| = 0.000e+00, removing 21.9% of the output MSE on a
correlated-input toy problem.

**It still makes the model worse.  Both arms lost to their own baselines:**

| 4B 2:4 | baseline | +compensation | delta |
|---|---|---|---|
| d=0.01 | 47.23 | **44.16** | **-3.07** |
| d=0.02 | 45.17 | **44.37** | **-0.80** |

The training loss said so first: at matched steps 260-320 the compensated arms
ran +0.020 (d=0.01) and +0.035 (d=0.02) ABOVE baseline.  An earlier note in
this file called the losses "indistinguishable" -- that was read off unmatched
steps and was wrong.

Why it fails, most likely: the compensation is applied AFTER the self-KL
re-measurement, so the trust-region guarantee does not hold on the weights
actually written, and observed `max |delta|` reached 3-16 with ridge=1e-6.
Only 1.6% of the pruned mass is absorbable, so there was little to gain and the
trust-region violation cost more than that.

This closes the 2:4 question the way the three pre-measurements predicted
(absorbable fraction 1.6%, random regrouping identical, 3rd/2nd = 0.75): no
local technique fixes the N:M deficit.  **Do NOT build the FSDP flat-shard path
(`_fsdp_nm_reconstruct`) for this feature** -- the 8B arm was a silent no-op and
implementing it properly would only reproduce a negative result at 8B.

The flag and its guards stay in the tree: the guards are what turned a silent
no-op into a visible one, and the negative result is appendix material.

## 3b. opkdfix lifted 8B 2:4, and flipped the delta ordering

| 8B 2:4 | pre-opkdfix | post-opkdfix | delta |
|---|---|---|---|
| d=0.01 | 50.15 | **52.64** | +2.49 |
| d=0.02 | 51.31 | **52.20** | +0.89 |

That lands at the top of the +1.1 to +2.9 range extrapolated on 2026-09-09, and
it REVERSES the ordering: pre-opkdfix d=0.02 > d=0.01, post-opkdfix d=0.01 wins.
Any 2:4 claim resting on the pre-opkdfix ordering has to be re-checked.

It does not close the gap.  ALPS+retrain at 8B 2:4 is 55.64, so SCOUT is still
**-3.00** short, and the two deltas sit 0.44 apart, i.e. the curve is flat and
no delta choice rescues it.  d=0.005 is running to pin the low end.

Also filled: **8B S50 d=0.005 = 63.83**, the cell that OOM'd on one GPU on
2026-09-08 (now FSDP 2-GPU).  It is BELOW d=0.01 (64.68), so S50's optimum
stays at d=0.01 and the low end is not where the wins are.

## 4. Cause of the 22:21 total-GPU-idle event

Both compensation runs died at startup with
`AttributeError: 'GradualMaskManager' object has no attribute 'capture_group_cov'`.
The new method had been anchored next to `capture_wanda_stats`, which belongs to
`FisherAccumulator`, not `GradualMaskManager`; the call site uses `maskmgr`.
Moved to the correct class and verified by AST that ownership is now right.

This was the second idle-GPU incident.  The `gpu_idle_watchdog.sh` was extended
from GPU 0-3 to **0-7** (including the `for g in 0 1 2 3` loop body, which the
first edit missed).  Queue waits use explicit PIDs, never `pgrep -f` patterns.

## 5. Current allocation -- GPU 0-3 ONLY (constraint set 2026-09-10 08:35)

| GPU | Run | Purpose |
|---|---|---|
| 0 | 4B 2:4 ALPS+retrain **tok=512, lr=1e-4** | token-matched baseline, SCOUT's lr |
| 3 | 4B 2:4 ALPS+retrain **tok=512, lr=5e-5** | token-matched baseline, the tok=256 run's lr -- also the sanity check that the hub prune is the right one (should land at or above 49.31) |
| 1,2 | 8B 2:4 d=0.005 | pins the low end of a flat 2:4 delta curve |

Queued behind them on the same four GPUs (`queue_0to3.sh`, waits on explicit
PIDs, pairs any two free of {0,1,2,3} since FSDP does not need adjacency):

1. **8B S70 d=0.03** -- d=0.03 was the best S70 cell pre-opkdfix (49.74) and
   post-opkdfix brackets it (0.02 = 50.12, 0.05 = 49.11).  The one remaining
   hole that can still move the S70 headline.
2. **8B S60 d=0.005** -- completeness; at S50 the low end came in below
   d=0.01, so a win here is unlikely.
3. **8B 2:4 d=0.03** -- completes the 2:4 sweep.  Lowest value: 2:4 trails
   ALPS+retrain by 3.00 and no delta closes that.

Two jobs launched on GPU 4-7 at 08:30 (8B 2:4 d=0.03, 8B S60 d=0.005) were
killed minutes later when the 0-3 constraint arrived; they are items 2-3 above.

### post-opkdfix 8B grid, as it stands

| sparsity | 0.005 | 0.01 | 0.02 | 0.03 | 0.05 |
|---|---|---|---|---|---|
| S50 | 63.83 | **64.68** | 63.91 | 63.56 | -- |
| S60 | queued | **58.90** | 57.29 | 58.59 | -- |
| S70 | -- | 49.84 | **50.12** | queued | 49.11 |
| 2:4 | running | **52.64** | 52.20 | queued | -- |

All five runs checkpoint every 256 steps.  Mask shards are rank-local:
**resume each run on the same GPU COUNT it started on.**

## 5b. 4B baselines DO exist -- in wandb, not in this container

> **"ALPS+retrain" means:** ALPS one-shot prune, then the mask is FROZEN
> (`--gmp_fixed_mask=true`) and only the weights train, under a THREE-term
> objective -- `gmp_ntp_lambda=0.33` (next-token prediction) +
> `gmp_kd_lambda=0.33` (offline KD from the dense Qwen3 teacher) +
> `gmp_onpolicy_kd_lambda=0.33` (on-policy KD on vLLM rollouts).  It is not
> plain SFT, and calling it "SFT" hides which term a knob actually moves --
> `max_new_tokens` 256 vs 512 changes the OPKD term ONLY (twice the tokens per
> rollout; the rollout COUNT is 16,640 either way), leaving NTP and offline KD
> untouched.


An earlier claim in this file that "every ALPS / ALPS+retrain baseline in this
project is 8B" was WRONG, and so was the 4B ALPS+retrain run queued on the strength
of it (queue disarmed, nothing was re-run).  The 4B baselines were trained on
the log_cluster side (H200, host `n92`, paths under `/home1/doyoonkim/`), so this
container's `logs/` has no trace of them -- but wandb does.  Searching only local
logs and local launchers is not evidence of absence; **`harvest_wandb.py` now
pulls these into `RESULTS_WANDB_BASELINES.tsv`** so the gap cannot recur.

### The complete 4B table (first time it has existed)

| 4B | SCOUT | ALPS+retrain (best lr) | delta |
|---|---|---|---|
| S50 | 62.06 (d=0.01) | 61.23 (lr=5e-5, tok=512) | **+0.83** |
| S60 | 55.93 (d=0.01) | 52.96 (lr=1e-4, tok=512) | **+2.97** |
| S70 | 45.58 (d=0.02) | 40.85 (lr=1e-4, tok=512) | **+4.73** |
| 2:4 | 47.23 (d=0.01) | 49.31 (lr=5e-5, tok=256) | **-2.08** |

SCOUT rows are `queue_gmp_pgd_grow_to_target_4b{,_24}` only -- NOT the `klgate`
capped/uncapped runs, which are neither the method nor a clean ablation.
ALPS+retrain rows take the best lr of each sparsity's sweep (1e-5/5e-5/1e-4/2e-4).

4B reproduces the 8B pattern exactly: SCOUT wins at every unstructured sparsity,
by a margin that GROWS with sparsity (+0.83 / +2.97 / +4.73), and loses only at
2:4.  The per-benchmark signature is the same too -- at 2:4 SCOUT loses MATH
(-7.8) and LCB (-3.7) and wins GPQA (+7.1).  So the 2:4 deficit is NOT an
8B artefact; it replicates across scale, which is what the "structural
constraint" reading predicted.

### The 2:4 baseline is the only one under-resourced, so -2.08 is a FLOOR

Every 4B ALPS+retrain run in the sweep used `max_new_tokens=512`.  The 2:4 one is
the single exception at **256** -- half the on-policy KD tokens per rollout,
exactly the trap the 8B launcher's own comment warns about.  It also used
lr=5e-5 where SCOUT 2:4 used 1e-4.  Both cuts run AGAINST the baseline, so a
token-matched baseline would score ABOVE 49.31 and SCOUT's deficit would widen.

### Rollout-cadence audit of that baseline (it is sound)

The config says `gmp_onpolicy_kd_interval=1`, which reads as "regenerate every
step".  It did not: the log's refill events are at steps 32, 64, 96, ... 2048 --
63 intervals, every one of them exactly 32.  `ro` did not drive the refill
cadence in that August code; `mask_interval=32` did (the same bug class as the
B1 frozen-pool sizing patched this session).  Consequences:
  - total rollouts = 65 x 256 = **16,640**, and SCOUT 4B 2:4 (ro=32,
    mask_interval=32) draws the identical 16,640.  The rollout budget matches by
    coincidence, so the comparison is not confounded on that axis.
  - **it was single-GPU** -- `gpu_count=1`, `gmp_use_fsdp=False`, one H200 --
    so the FSDP OPKD-duplication bug that `opkdfix` (23aa3ed) fixed cannot
    apply: that commit is a no-op at `world_size == 1`.

## 5c. harvest_results.py bug found and fixed (2026-09-10)

The `model` column was derived from the LOG PATH while the `label` came from
the run-dir config, so any log whose directory name lacks "8b" -- e.g.
`resweep2_opkdfix/n24_klb0.01_resume.log`, `delta_resweep_opkdfix/*` -- was
labelled "8B ..." but filed under model `?`.  Every query that filtered on the
model column silently dropped the entire post-opkdfix resweep.  Config now
wins, path is the fallback.  Anyone who pulled an 8B grid before this fix got a
grid missing its newest rows.

## 6. Still outstanding

- Run `delete_verified.sh` (341.1 GB, 22 Hub-verified deletions) -- `rm` is
  permission-blocked for the agent, the user must run it.  The 16.4 GB debug
  artifact `models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260901_105142` is NOT in
  that script and also wants deleting.
- Regenerate `scout_ablation_iclr2027.zip` with post-opkdfix 8B numbers once the
  runs above land.
- Open decision: LCB/MATH 32768 re-measurement scope (appendix contrast table vs
  full re-measure).  The quick profile caps BOTH `max_new_tokens` and
  `max_model_length` at 8192; GPQA truncation rate is 0.87, so LCB/MATH numbers
  are truncation-limited, not capability-limited.
