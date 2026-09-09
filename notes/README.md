# Session notes, results tables, and the tooling that produces them

Everything here used to live only in `/NHNHOME/log-postech/doyoonkim/logs`,
which is **outside this repository**. That container has already died twice
(2026-09-02 12:00, 2026-09-08 15:30); each time these files would have gone
with it. They are versioned here now.

## What is authoritative vs derived

| File | Kind |
|---|---|
| `RESULTS_SESSION_*.md` | **Authoritative.** Causes, decisions, retractions -- the things no script can re-derive. |
| `RESULTS_ALL.{tsv,md}` | **Derived snapshot.** Regenerate any time with `harvest_results.py`; it only reads training logs. |
| `RESULTS_WANDB_BASELINES.tsv` | **Derived snapshot** of baselines that exist only in wandb (`harvest_wandb.py`). |
| `NEXT_SESSION.md`, `RESUME_*.md` | Handover notes. |
| `queues/` | The exact queue scripts that launched each batch, kept for provenance. |

## The two harvesters, and why there are two

`harvest_results.py` reads this container's raw training logs.
`harvest_wandb.py` reads wandb.

Both are needed: many baselines were trained on the log_cluster side (H200,
host `n92`, paths under `/home1/doyoonkim/`), so this container's `logs/` has
no trace of them. Searching only local logs once produced the false conclusion
that "there is no 4B ALPS+retrain baseline" -- there are several, including the
2:4 one -- and nearly cost 8 GPU-hours re-running an experiment that already
existed. **Absence from local logs is not absence.**

Avg is always the mean of ALL FIVE quick-profile benchmarks (MATH500, GPQA
diamond, IFEval prompt-strict, LCB codegen, GSM8K). A run missing any of the
five gets a blank Avg rather than a mean over whatever finished, so a partial
eval can never pass as a complete one.

## Terminology

**ALPS+retrain**, not "ALPS+SFT": ALPS one-shot prune, then the mask is FROZEN
(`--gmp_fixed_mask=true`) and only weights train, under a THREE-term objective
-- `ntp_lambda=0.33` + `kd_lambda=0.33` (offline KD from the dense teacher) +
`onpolicy_kd_lambda=0.33` (on-policy KD on vLLM rollouts). Calling it "SFT"
hides which term a knob moves: `max_new_tokens` 256 vs 512 changes the OPKD
term ONLY.

## Refreshing

    bash notes/sync.sh     # re-harvest, copy into notes/, show what changed

It does not commit. Review the diff, then commit yourself.
