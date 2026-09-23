# Which run is behind which row

Every claim here was checked against wandb configs / downloaded `output.log`s on
2026-09-22, not from memory. Checkpoints are matched to trainings through the
training run's `hub_model_id` summary field, because `main.py` builds the Hub
repo id from the **push** time, so it never equals the local save directory's
timestamp — matching by timestamp alone gives the wrong run.

Read this before trusting a Table 6 / Figure 4 row: three of the rows below are
not what their label says.

---

## Confirmed problems

### 1. `s3_4b_s70_norefresh_rep` is NOT a replicate (4B block, 34.34)

| | original `nve0ruhr` (49.20) | `_rep` `eniy0b69` (34.34) |
|---|---|---|
| training run | `laphj0g7` | `izc2311t` |
| checkpoint | `gmp-kd3e-1-s70pct-lr1e-4_20260903_081638` | `..._20260909_221717` |
| `gmp_onpolicy_kd_interval` | 4096 | 4096 |
| **`gmp_pgd_jump_to_target`** | **False** | **True** |
| `gmp_resume_from` | none | `ablation4b_jump/ckpt_s70_A3B1jump_4b/step000256.pt` |

The 15-point gap is `jump`, not run-to-run variance (jump alone costs −10.1 at
4B s70). It is a **two-factor** arm: no refresh *and* no trust region. Do not
cite it as a failed reproduction. Either drop the block or register it in
`METHOD` under a name that says what it is.

### 2. 8B 70% ablation arms do not share one trust radius

The paper (§B.3) selects δ per model and sparsity; 8B 70/80% is δ=0.03. The 8B
headline moved to 0.03 in the 2026-09-10 resweep, but two ablation arms were
trained days earlier at 0.02 and never redone:

| arm | training run | δ | `ri` | resume | matches SCOUT? |
|---|---|---|---|---|---|
| SCOUT (56.71) | `ajm5l60w` | **0.03** | 32 | **step001280.pt** | — (reference) |
| w/o OPD (54.70) | `g5htdr2q` | 0.03 | 32 | none | yes |
| w/o TR / jump (47.76) | `dkek5q9w` | **0.02** | 32 | none | **no** |
| w/o rollout refresh (52.42) | `n69llpef` | **0.02** | 4096 | none | **no** |

So the −4.3 that "w/o refresh" shows at 8B 70% mixes the refresh change with a
trust-radius change; it is not a single-variable measurement yet. `uywgha9p`
(δ=0.02, ri=32, no resume) is the matching reference and its eval
(`r3rftcwk`) was running on n79 as of 2026-09-22 21:00.

8B **60%** is clean: SCOUT `mgf2ka9s` and jump `c118wpje` are both δ=0.01.

### 3. The 8B 70% SCOUT reference is a resumed run

`ajm5l60w` sets `gmp_resume_from=.../resweep2_opkdfix/ckpt_s70_delta0.03_opkdfix/step001280.pt`.
The directory name says it resumed its own δ=0.03 configuration, but this is
the number in Table 5 and Figure 2/3/4, not an appendix cell, so it is worth
confirming rather than assuming.

---

## Checked and clean

- **`gmp_grad_accum` differs across arms but the global batch does not.** The
  launchers scale it inversely to rank count: single-GPU 8, fsdp2gpu 4x2,
  fsdp4gpu 2x4 — all 8 sequences. Verified from each run's own
  `Training FLOPs: <n_params> params x <n_tokens> tokens` line: 4B SCOUT s70,
  w/o OPD s50/s60/s70 and OPD-only s70 all processed **134,217,728 tokens**.
  The `flops` metric alone cannot show this: under FSDP `n_params` is the rank
  shard, so the world_size factor cancels and flops looks 1/N. Use the token
  count, not flops.
- **4B `w/o TR (jump)`** `3olz82te`: δ=0.02, same as its 4B SCOUT reference
  `r5j1uw8d`. Resumes from its own `ablation4b_jump/ckpt_s70_A3jump_4b/step000512.pt`.
- **4B `Cubic schedule, no PGD`** `ccddldfw` (eval `ay0dnnto`, 49.09):
  `gmp_pgd=false`, `gmp_tr_enabled=false`, cubic ramp with
  `gmp_pruning_end_ratio=0.11328125` -> `pruning_end_steps=232`. The ramp reaches
  0.700 at step 224 and `gmp_trainer.py` gates mask application on
  `step <= pruning_end_steps`, so the mask is frozen for the remaining 1816
  steps. Full 2048 steps, `Final sparsity: 0.7000`. On-policy KD stays on
  (pool refilled every 32 steps). Pace matches SCOUT s70, which reaches 0.700
  at step 224.
- **4B `Cubic schedule, no PGD` at s80** `ssiusvd3`: same launcher, only
  `sparsity_ratio` 0.7->0.8 and `gmp_pruning_end_ratio` 0.11328125->0.19140625.
  The ratio comes from SCOUT s80 (`q0kmiudv`) reaching 0.800 at step 392 of
  2048; the cubic ramp hit 0.800 at step 384.

---

## Scoring notes

### `strict/` tables

`accuracy_strict = accuracy * (1 - correct_truncation_rate)`, i.e. a response
that exhausted the generation budget counts as wrong even if its extracted
answer matched. `<bench>_correct_truncation_rate` is `P(truncated | correct)`
(`lighteval_bench.py` averages the truncation flag over the correct samples
only), so the product is the share of prompts that are both correct and
terminated, over the same denominator as the standard accuracy. This is **not**
the paper's Table 8, which divides by `1 - truncation_rate` and so changes the
denominator per arm. Regenerate with `python harvest_long_tsv_strict.py`; it
imports `harvest_long_tsv` and replaces only the accuracy cells, so labels and
grouping cannot drift between the two.

Seed spread grows under strict scoring (4B s70 SCOUT avg5 std 0.41 -> 0.91),
because how much gets truncated varies by sampling seed. Significance
thresholds are not the same in the two tables.

### GPQA on collapsed models is extraction noise, not partial credit

`Metrics.gpqa_instruct_pass_at_k` uses
`IndicesExtractionConfig(..., try_extract_without_anchor=True)`, so when the
response never emits `Answer: $LETTER` the extractor takes any A/B/C/D it can
find in the text. Measured on `eval_s3_8b_sparsegpt_s70` (GPQA 26.26,
truncation 100%): only **3 of 198** responses contain an `Answer:` anchor at
all; the rest are degenerate loops ("But I need to think of the problem." x
hundreds) truncated at the cap. Re-running the extractor reproduces lighteval's
score exactly (52/198 = 26.26%), and the extracted predictions are skewed
B=98, A=60, D=20, C=18 against a near-uniform gold distribution — the letters
come from the instruction the model parrots back, not from an answer. Strict
scoring zeroes these rows, which is the correct value.

### lighteval's sample cache only resumes two benchmarks

`cache_management._get_task_hash` hashes `LightevalTaskConfig.__str__(lite=True)`,
which is stable for `math_500` and `lcb:codegeneration` but not for
`gpqa:diamond`, `ifeval` or `gsm8k` (dozens of distinct hashes across the logs
in `logs/eval_8b_long/*.log`). A resumed eval therefore **always regenerates
gpqa, ifeval and gsm8k** even though their `GENERATIVE.parquet` is sitting in
the checkpoint directory. Budget a resumed long eval as a full run minus
math500 and minus whatever LCB rows were already written.
