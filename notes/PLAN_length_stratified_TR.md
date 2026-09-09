# Plan: length-stratified trust-region ablation (candidate replacement for the cubic control in Fig 2)

Goal: show empirically that the benefit of function-space control grows with the
generation horizon H, connecting Proposition 6.1 (D_TV <= sqrt(H*delta/2), i.e. a
fixed conditional-KL radius buys a weaker rollout guarantee as H grows) to a
measured phenomenon, and move the pace-matched cubic control to the appendix.

Decision rule agreed up front: replace the main-text panels ONLY if the effect is
monotone across bins or clearly larger in the longest bin. If it is messy, keep
cubic in the main text and drop this.

## 0. Why existing artifacts are not enough (measured, not assumed)

Probed the 4B s70 pair that would supply the figure -- SCOUT = klb0.02
(models/gmp_s70pct_..._20260903_065326), w/o TR = klb99999
(..._20260903_060450). Both have per-example details parquets and they align
(same order, identical input-token lengths), so the paired analysis is
mechanically possible. Three blockers, all measured:

1. GROUPING VARIABLE FLIPS THE ANSWER. Paired gap on MATH-500 by quartile:
       grouped by SCOUT's own output length:  +7.2  +11.2  +20.8   -5.6
       grouped by w/o-TR's own output length: +0.8   -3.2   +4.0  +32.0
   Opposite conclusions from identical data; the confound (~35 points) is far
   larger than the effect being hunted. A treatment-independent proxy is
   mandatory, and no dense Qwen3-4B evaluation exists yet -- it has to be run.

2. CENSORING DESTROYS THE TOP BIN. Truncation rates against the real per-sample
   cap (max_model_length - prompt, the metric fixed in commit a67bf99):
       math500  SCOUT 31.0%  w/oTR 38.6%  both 24.8%
       gsm8k          43.1%        48.6%       34.3%
       gpqa           77.3%        81.8%       71.7%
       ifeval         48.4%        42.5%       29.6%
       lcb            84.3%        89.2%       82.8%
   In MATH-500's longest quartile 61-76% are truncated in BOTH arms, and a
   truncated generation scores 0 for both (verified: on LCB every truncated
   generation scores 0 in both models). So the top bin is where the paired gap
   is mechanically pushed toward 0 -- exactly the regime the hypothesis is about.

3. GSM8K CANNOT BE QUARTILED AT ALL at its current budget: generation cap 2048
   with 34% of samples sitting at the cap, so qcut collapses to three bins
   (observed Q4 n=0).

Also note the per-benchmark gaps disagree in sign (math500 +8.40, gsm8k +4.93,
gpqa +0.51, ifeval -2.96, lcb -0.37), so a five-task pooled average would be
carried by MATH/GSM8K with the rest adding noise against the hypothesis.

## 1. Design decisions

BENCHMARKS: MATH-500 and GSM8K only.
  GPQA and LCB are unrecoverable (72-83% both-truncated even before binning) and
  IFEval is not horizon-driven -- instruction compliance does not get harder with
  longer reasoning in the way Prop 6.1 describes. State this restriction in the
  caption rather than burying it.

HORIZON PROXY: dense Qwen3-4B generated reasoning length, measured once, never
per-method. Quartiles computed WITHIN each benchmark (avoids the MATH-long /
GSM8K-short task-composition confound), then bins pooled across benchmarks.

BUDGETS: do NOT use `--profile official`. It sets max_new_tokens ==
max_model_length (32768/32768), which reproduces the same censoring -- the true
cap is max_model_length minus the prompt -- and it leaves GSM8K at 2048/4096.
Use explicit budgets with headroom so the intended cap is the binding one:
      MATH-500  max_new_tokens 24576, max_model_length 32768   (cap ~32.6k)
      GSM8K     max_new_tokens  4096, max_model_length  8192   (cap ~7.3k)
Verify after each run that {bench}_truncation_rate is low (<5%) for the dense
model; if it is not, the proxy is censored and the budget must go up again.

ESTIMATOR: paired, never absolute accuracy (otherwise "long problems are just
harder" explains everything). Per bin report
      Delta = acc(SCOUT) - acc(w/o TR)
and the equivalent discordant-pair decomposition
      P(SCOUT correct, w/oTR wrong) - P(SCOUT wrong, w/oTR correct)
which is the same number but exposes how many discordant pairs actually carry
it. Bootstrap CI (10k resamples, resampling examples within benchmark, then
recombining) on every bin.

## 2. Stage 1 -- pilot (MATH-500 only, 3 eval-only jobs, 1 GPU each, ~1h each)

  P1  dense Qwen3-4B      MATH-500 @ 24576/32768, --save-details
  P2  SCOUT 4B s70        (models/gmp_s70pct_..._20260903_065326) same budget
  P3  w/o-TR 4B s70       (models/gmp_s70pct_..._20260903_060450) same budget

Eval only -- no training. Run via elsa/scripts/lighteval_patched_runner.py with
explicit model_args, or lib.lighteval_bench._run_lighteval with the budgets above.

Then: join the three parquets on doc id, bin by P1's output length, compute the
estimator above.

DECISION GATE
  GO   if Delta is monotone in the bin index, or Q4 exceeds Q1 by more than the
       bootstrap CI half-width of their difference.
  NO-GO otherwise -> keep cubic in the main text, stop here, cost was ~3 GPU-h.

## 3. Stage 2 -- only if the gate passes

  S1  add GSM8K at 4096/8192 for all three models (3 more eval jobs)
  S2  optional but preferable for a main-text claim: re-run the "w/o TR" arm as
      the CORRECTED control, --gmp_pgd_jump_to_target=true, rather than
      kl_budget=99999. The published arm reaches target at step 48 instead of
      immediately (the bisection only gets gmp_pgd_kl_bisect_iters=6), so it
      changes the gate AND the trajectory at once. 4B is single-GPU, ~5h.
  S3  same treatment for w/o rollout refresh (ro=4096) if panel (c) is wanted.

## 4. Stage 3 -- paper edits

  Figure 2 becomes: (a) SCOUT gain over matched recovery vs sparsity [unchanged]
                    (b) SCOUT - w/o TR vs dense-response-length quartile
                    (c) SCOUT - w/o refresh vs the same quartiles
                        (or fold into (b) as a second line if refresh is flat)
  Section 6.2 keeps one sentence on the cubic control pointing at the appendix.
  Appendix gains the existing cubic trajectory, the 2.75*delta / 48% exceedance
  figure, and the 2.83-point statement at 70%.

## 5. Risks

  * Budget mismatch with Table 3. The analysis figure would sit at 24576 while
    the headline table is quick/8192. Defensible if stated, but a reviewer can
    ask; the alternative (re-running Table 3 at the larger budget) is a much
    bigger job and should be a separate decision.
  * Even uncensored, the top bin may just be harder for both arms. The paired
    estimator handles the level, but not a genuine interaction between
    difficulty and method. If Q4 shows a large gap, check it is not driven
    entirely by a handful of discordant pairs (report the discordant count).
  * MATH-500 alone gives 125 examples per bin. Bootstrap CIs will be wide;
    GSM8K (330/bin) is what makes the figure publishable, so Stage 2 S1 is
    effectively mandatory if the pilot passes.
