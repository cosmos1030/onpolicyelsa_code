# Qwen3 Dense Baselines — Official Protocol Eval (2026-08-11)

Official Qwen3 thinking-mode eval protocol: max_new_tokens=32768 (math500/gpqa/ifeval/lcb),
temperature=0.6, top_p=0.95, top_k=20, vLLM tensor_parallel_size=2.
MATH-500/IFEval: single seed (42) — large sample counts, low per-problem variance.
GPQA-Diamond (198 problems) / LiveCodeBench: 3 seeds (42, 0, 1) — small sample counts,
high per-problem variance (1 problem ≈ several points).

Source jobs: bench32k_{1.7b,4b,8b} (math500/ifeval, seed 42), gpqa32k_{1.7b,4b,8b}_{0,1}
+ seed-42 leg from bench32k (gpqa), lcb32k_{1.7b,4b,8b}_{0,1} + seed-42 leg from bench32k (lcb).
All wandb runs under project reasoning_qwen3_{1.7b,4b,8b}, entity
dyk6208-gwangju-institute-of-science-and-technology.

## Results

| Model | MATH-500 | IFEval | GPQA-Diamond (mean ± std, n) | LiveCodeBench (mean ± std, n) |
|---|---|---|---|---|
| Qwen3-1.7B | 89.6 | 71.4 | 40.1 ± 2.0 (n=3: 42.4/38.9/38.9) | 32.6 ± 1.8 (n=3: 31.7/31.3/34.7) |
| Qwen3-4B   | 95.8 | 81.0 | 55.2 ± 4.1 (n=3: 59.6/51.5/54.6) | 52.0 ± 0.6 (n=3: 51.9/52.6/51.5) |
| Qwen3-8B   | 96.8 | 83.9 | 58.6 ± 1.0 (n=3: 58.6/57.6/59.6) | 57.6 ± 0.9 (n=3: 58.6/57.5/56.7) |

Per-seed raw values (GPQA/LCB):

| Model | GPQA seed=42 | GPQA seed=0 | GPQA seed=1 | LCB seed=42 | LCB seed=0 | LCB seed=1 |
|---|---|---|---|---|---|---|
| 1.7B | 42.42 | 38.89 | 38.89 | 31.72 | 31.34 | 34.70 |
| 4B   | 59.60 | 51.52 | 54.55 | 51.87 | 52.61 | 51.49 |
| 8B   | 58.59 | 57.58 | 59.60 | 58.58 | 57.46 | 56.72 |

## Notes / caveats

- All 3 seeds now landed for every model/benchmark (4B GPQA seed=0, job 710046, was
  missed on the first timeout-retry pass — only seed=1 for 4B was resubmitted initially,
  seed=0 slipped through and was caught and resubmitted afterward).
- All GPQA/LCB retries needed a longer `--time` than the original submissions
  (2h/3h original limits were too short at 32768 budget + tp=2; retries used 8-10h).
  See job history: 709216/17/18/19 (gpqa, 2h) and 709225/26/27/28 (lcb, 3h) timed out;
  resubmitted as 709803/709852/709853/710046 (gpqa) and 709935/709936/709978 (lcb).
- These are the correct, "proper" official-protocol dense baselines — not to be confused
  with the old 8192-budget/no-top_k/single-seed numbers on the reasoning-bench artifact
  (https://claude.ai/code/artifact/ff2e0e8c-f600-4063-b593-2637d9d13498), which is a
  separate, intentionally-8192-only page for the ongoing TR-GMP sweep comparisons.
