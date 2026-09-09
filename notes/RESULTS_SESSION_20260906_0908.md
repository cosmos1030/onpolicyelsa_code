# Session results, 2026-09-06 to 2026-09-08 (container shutdown ~15:00 on 09-08)

All numbers are the quick eval profile (5 tasks, MATH-500 / LCB / GPQA-diamond /
GSM8K / IFEval, 8192 budget except GSM8K 2048/4096) on Qwen3-8B unless noted.
avg5 is their unweighted mean. Every model below is on the hub; per-example eval
data for all of them is in this repo under logs/eval_details_distilled/.

## 1. The OPKD rollout-sharding bug, and what fixing it did

Under FSDP every rank read the OPKD rollout pool from index 0, so an 8B 2-rank
run trained on 128 unique rollouts per refill window where the 1-rank 4B recipe
saw 256, and each rollout was consumed twice. Fixed: pool scales with
world_size, pointer starts at local_rank and strides by world_size.

Paired re-runs at delta=0.03, fixed vs old sharding:

    s50   avg5 -0.62   ppl improved   target reached 104 -> 96
    s60   avg5 +0.10   ppl improved   144 -> 136
    s70   avg5 -0.79   ppl improved   192 -> 168

ppl improved 3/3 and growth accelerated 3/3; avg5 moved -0.44 +- 0.81, i.e. not
distinguishable from zero. The fix was never about raw score -- it was about
matching the rollout budget the ALPS+retrain baselines always had (they are
single-GPU, so they never had the bug).

## 2. Delta re-sweep: the published default 0.03 is never optimal

                 0.01      0.02      0.03      0.05
    s50       64.676*   63.908    63.560         -     strictly monotone
    s60       58.902*   57.286    58.592         -
    s70       49.840    50.124*   48.954    49.114     0.01/0.02 tie
    ppl best    0.01      0.01      0.01                (8.879 / 10.927 / 13.432)

Mechanism, confirmed at zero GPU cost from existing 4B logs: 4B was always
world_size=1, never had the bug, and its optimum was 0.02. 8B looked like 0.03
while buggy and moved to 0.01-0.02 once fixed -- i.e. onto 4B's optimum. Halving
the effective rollout diversity made a larger KL budget look necessary.

## 3. Fair baselines: ALPS+retrain re-run at rollout length 512

> **"ALPS+retrain" means:** ALPS one-shot prune, then the mask is FROZEN
> (`--gmp_fixed_mask=true`) and only the weights train, under a THREE-term
> objective -- `gmp_ntp_lambda=0.33` (next-token prediction) +
> `gmp_kd_lambda=0.33` (offline KD from the dense Qwen3 teacher) +
> `gmp_onpolicy_kd_lambda=0.33` (on-policy KD on vLLM rollouts).  It is not
> plain SFT, and calling it "SFT" hides which term a knob actually moves --
> `max_new_tokens` 256 vs 512 changes the OPKD term ONLY (twice the tokens per
> rollout; the rollout COUNT is 16,640 either way), leaving NTP and offline KD
> untouched.


Every 8B PGD run passed gmp_onpolicy_max_new_tokens=512; every ALPS+retrain baseline
used the launcher default of 256, i.e. half the on-policy KD tokens per rollout
(rollout COUNT was matched at 256/window on both sides -- only length differed).
Defaults are now 512 everywhere.

    sparsity   ours (best delta)   ALPS+retrain (tok=512)   gap      ALPS+retrain (tok=256, old)
    s50            64.676             62.444     +2.23        63.341 (cluster)
    s60            58.902             57.614     +1.29        56.327
    s70            50.124             46.664     +3.46        46.402
    2:4            51.310             55.642     -4.33        55.060

SCOUT wins at every unstructured sparsity, but NOT monotonically in sparsity
(+2.23 / +1.29 / +3.46) -- do not write it up as a monotone trend. The earlier
"s50 is a tie (+0.22)" claim is retracted: that baseline was a cluster run that
also had the OPKD duplication bug.

2:4 is the one place SCOUT loses, and the comparison is not yet fair on our
side: klb 0.005 and 0.01 OOM'd before the memory fixes in 22f7547, so 0.02 is
the only surviving point and it was never swept. n24_klb0.01 got to step 559
today and has a step-512 checkpoint to resume from.

Per-task pattern is consistent: GPQA is where SCOUT dominates (+9.1 at s70,
+4.0 at s60, +8.6 at s50); LCB is the only loss.

### Why LCB loses, checked per-example (s70, ours vs ALPS+retrain (tok=512))

    output len mean   6953  vs  6956      truncation  81.7% vs 81.0%
    accuracy if truncated  0.00% vs 0.00% (both -- no code block ever emitted)
    accuracy if finished  46.94% (n=49) vs 76.47% (n=51)
    on the 38 problems BOTH finished:     52.6%   vs   81.6%

Not a length or truncation artifact -- verbosity and truncation are identical.
It is a real code-generation deficit on the ~18% of problems that fit the
budget. Note the whole benchmark currently rests on those 49/51 finished
samples, so re-evaluating LCB at a larger budget is needed before writing this
up either way.

## 4. Trust region ablation: the published control understates it ~4x

kl_budget=99999 loosens the gate but does not remove pacing -- the per-step
search only gets gmp_pgd_kl_bisect_iters=6, so it still creeps to target over 48
steps. --gmp_pgd_jump_to_target (new in 156a439) takes the whole prune candidate
set on the FIRST projection.

    SCOUT (delta=0.02)        avg5 50.124   ppl 13.959   target at step 216
    w/o TR, jump (correct)    avg5 42.864   ppl 16.024   target at step 8   -7.26
    w/o TR, kl=99999 (old)    avg5 45.248   ppl 15.756   target at step 48  -1.96

Every 4B ablation arm in the paper uses kl=99999. 4B re-runs with the correct
control are in flight (logs/ablation4b_jump/, checkpointed).

## 5. Infrastructure bugs found and fixed

  * vLLM tensor_parallel_size=4 eval hangs in model.cleanup() before metrics.
    Cost 3 hangs x ~45 min on 4 GPUs before diagnosis. Every eval that ever
    completed here ran tp=2 (20/20). Run evals at tp<=2.
  * truncation_rate compared output length against a flat max_new_tokens while
    the real cap is max_model_length - prompt, so it read 0.0% forever. Actual
    rates: LCB 82%, GSM8K 43%, MATH 31%. Fixed in a67bf99.
  * Frozen-pool ablation arms died twice (step 39, step 43) on
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=480 -- the monitor thread killing the
    process because the watchdog thread starved, not a collective timeout. The
    sharding fix doubled the frozen pool (8192 -> 16384) and pushed the
    GIL-holding stretch past 480s. Raised to 3600 in the 8B FSDP launchers;
    VERIFIED -- the retry is past step 274 where both earlier attempts died.
  * Single-GPU 8B PGD is infeasible: OOM at step 8 (needs 4.64 GiB, 2.59 free).
    FSDP sharding of the optimizer state is what makes 8B fit. 8B PGD needs >=2
    GPUs; only the fixed-mask ALPS+retrain path fits on one.
  * GPU allocation compared CUDA_VISIBLE_DEVICES by string equality, so a pair
    looked free while a single-GPU job held half of it. All queues and the idle
    alarm now test for overlap.

## 6. Analysis that did NOT pan out

Length-stratified trust-region ablation (proposed as a main-text replacement for
the pace-matched cubic control). Run on gold-solution-length quartiles, which is
treatment-independent, scoring truncated as wrong:

    MATH-500  absolute  +7.09 +11.81  +9.92  +4.80   inverted U
              relative  +8.18 +17.05 +16.44 +11.76   inverted U
              err.red. +52.94 +38.46 +25.00  +8.11   monotonically DECREASING
    GSM8K     absolute  +5.00  +5.28  +3.44  +5.97   flat

No scale shows the hypothesised monotone growth, and the three scales disagree
about the direction -- which is itself evidence the effect is small next to the
difficulty gradient. Keep the cubic control in the main text.

One signal worth an appendix line: the excess truncation of the no-TR arm DOES
grow with horizon (MATH +6.3 -> +9.6 across quartiles), consistent with
Proposition 6.1's direction.

Caveat on all of the above: it used the flawed kl=99999 arm. Worth redoing
against the jump control now that the effect is ~4x larger.

## 7. Backups

  * datasets/cosmos1030/scout-artifacts -- distilled per-example eval data for
    all 416 (run, benchmark) pairs (303 GB -> 5.6 MB), raw logs, queue scripts,
    result tables, plan and resume docs.
  * qwen3-8b-alps-2to4 / -s60pct / -s70pct newly uploaded (only s50pct existed).
  * 8 locally-trained checkpoints whose hub upload could not be confirmed,
    re-uploaded under cosmos1030/backup-*.
  * 6 commits pushed to the code repo.

Note on storage: the Lustre mount is 99% full (7.8T of 600T free), but our own
footprint is 1.5T, i.e. 0.25% of what is used. Deleting our data does not fix
it; the exposure is other tenants filling the rest, which would make all writes
including checkpoints fail.
