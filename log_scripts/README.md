# log_scripts

4B **s80** reasoning evals to run on a non-SLURM server. Three arms, one
protocol:

| arm | checkpoint |
|---|---|
| `base` | `cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740` (SCOUT s80) |
| `dpo_lr1e5_ep04` | `Riasok/dpo-scout-s80-ultrafeedback-lr1e-5-beta0.05-epoch0.4_20260918` |
| `dpo_lr5e6_ep10` | `Riasok/dpo-scout-s80-ultrafeedback-lr5e-6-beta0.05-epoch1.0_20260918` |

`long` profile, seeds 0/1/42, math500 + gpqa + ifeval + lcb + gsm8k,
PPL and zero-shot skipped. Results land in wandb
`reasoning_qwen3_4b_nostrip8192` as `s3_4b_s80_*`, next to the rest of the
s3_* table.

## Run it

1. Edit `env.sh` -- only `REPO_ROOT` and `PYTHON` are required. Everything
   else has a working default and can be overridden from the shell.
2. `bash run_all_s80_long.sh` (sequential) or `bash run_all_s80_long.sh 0 1 2`
   (one arm per GPU).

A single arm: `bash eval_s80_long.sh dpo_lr1e5_ep04 0`.

`env.sh` aborts up front on a missing path, a checkout without the `long`
profile, or a non-executable python, rather than letting the run die forty
minutes in.

## Why base is included

The SCOUT-s80 numbers on record (wandb `q0kmiudv`: math500 43.0, gpqa 27.78,
ifeval 19.59, gsm8k 54.21, lcb 0.0) come from the training job's inline eval,
which uses the **`quick`** profile: math500/gpqa/ifeval/lcb at 8192 with
`max_model_length` also 8192 -- so the prompt is subtracted from the
generation budget -- gsm8k at 2048, one seed, n=1.

`long` gives math500/gpqa/ifeval 16384, lcb 32768, gsm8k 8192. At 80% sparsity
truncation is the dominant failure mode, so 8192 -> 16384 moves accuracy far
more than seed noise does. A DPO delta computed against the 8192 record would
be mostly budget, not DPO.

## Notes

- `TP_SIZE` defaults to 1. Raise it only on "No available memory for the cache
  blocks" -- 4B bf16 is ~8GB and the 33k-token KV cache is ~4.8GB.
- `eval_full.py` aborts if `wandb.init` fails. That is deliberate (results have
  been silently lost to a swallowed wandb timeout). `EVAL_ALLOW_NO_WANDB=1`
  bypasses it for a throwaway run.
- The truncation rate and mean generated tokens per benchmark are logged
  alongside accuracy, which is what the +3 / -10% tokens / -7-8%p truncation
  claim needs to be checked against.
