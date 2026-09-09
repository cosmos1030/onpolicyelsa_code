# Next session — start here

Container went down ~15:00 on 2026-09-08. Full results:
RESULTS_SESSION_20260906_0908.md. Failure diagnoses and per-job resume detail:
RESUME_20260908.md. Everything is mirrored at
https://huggingface.co/datasets/cosmos1030/scout-artifacts

## Step 0 — bring the box back

    cd /NHNHOME/log-postech/doyoonkim/onpolicyelsa_code && git pull
    nvidia-smi                                   # expect 4 idle B200s
    ls logs/*/ckpt_*/                            # confirm checkpoints survived

~/.bashrc is container-local: re-run `conda init bash` and re-paste the export
block from b200_scripts/README.md. Everything else (conda env, secrets, caches,
data) is on persistent storage.

## Step 1 — resume the four interrupted jobs, one command

    bash /NHNHOME/log-postech/doyoonkim/logs/resume_all.sh

It finds the newest stepNNNNNN.pt per job, relaunches with the same recipe and
the same GPU count, and rolls them (2+2+1+1 GPUs wanted, 4 available). Checkpoint
state as of shutdown:

    s70_B1frozen      step000512   8B, 2 GPU   frozen pool, ro=4096
    n24_klb0.01       step000512   8B, 2 GPU   2:4 delta point
    s70_A3jump_4b     step000256   4B, 1 GPU   corrected w/o-TR control
    s70_A3B1jump_4b   (none yet)   4B, 1 GPU   both arms

## Step 2 — priorities once those are running

1. FINISH THE 4B JUMP ABLATIONS. Every w/o-TR arm in the paper uses
   kl_budget=99999, which does not remove pacing (reaches target at step 48, not
   1). At 8B the corrected control costs 7.26 avg5 vs 1.96 for kl=99999 -- the
   published ablation understates the trust region roughly fourfold. These two
   4B runs are what let Figure 2 be redrawn honestly.

2. COMPLETE THE 2:4 DELTA SWEEP. 2:4 is the only place SCOUT loses
   (51.310 vs ALPS+retrain (tok=512)'s 55.642), and our side has exactly one delta point
   because klb 0.005/0.01 OOM'd before the memory fixes in 22f7547. n24_klb0.01
   resumes from step 512; klb0.005 has never run. Until both exist the 2:4 row
   is not a fair comparison and should not be presented as one.

3. RE-EVALUATE LCB AT A LARGER BUDGET. LCB is the only unstructured loss (8.6 vs
   14.6 at s70) and 82% of its generations hit the cap, so the score rests on the
   ~18% that finish. Checked per-example: verbosity and truncation are identical
   between the two models (6953 vs 6956 tokens, 81.7% vs 81.0%), and on the 38
   problems both finished it is 52.6% vs 81.6% -- a real code deficit, not an
   artifact. But 38 problems is a thin basis for a paper claim. Re-run LCB only,
   ours + ALPS+retrain, with max_new_tokens 16384-24576 and max_model_length above
   it (NOT --profile official: it sets gen == ctx, which reproduces the same
   censoring, and leaves GSM8K at 2048).

4. REDO THE LENGTH-STRATIFIED ANALYSIS AGAINST THE JUMP CONTROL. The version run
   this session used the flawed kl=99999 arm and found nothing on any scale
   (absolute, relative, error-reduction all disagree; see
   PLAN_length_stratified_TR.md). With a 4x larger effect it is worth one more
   look. No GPU needed -- logs/eval_details_distilled/ has per-example
   correctness, token counts, truncation flags and gold lengths for all runs.
   Keep the cubic control in the main text until this actually shows something.

## Standing constraints, learned the hard way

  * Run evals at tensor_parallel_size <= 2. tp=4 hangs in vLLM's
    model.cleanup() after generation; cost three ~45-minute hangs on 4 GPUs.
  * 8B PGD needs >= 2 GPUs. Single-GPU OOMs at step 8 -- FSDP sharding of the
    optimizer state is what makes it fit. Only the fixed-mask ALPS+retrain path fits
    on one GPU.
  * Frozen-pool arms (ro >= steps) need TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600,
    already set in the 8B FSDP launchers. At the default 480s the monitor thread
    kills rank 1 around step 40. Verified fixed: the retry ran past step 500.
  * Never edit a launcher while a job is executing it -- bash reads scripts by
    byte offset and the running job breaks. Use a temp file plus mv (atomic
    rename); the running process keeps the old inode.
  * Rollout length is 512 everywhere now (it was 256 by default, which quietly
    halved the baselines' on-policy KD tokens). Do not lower it without
    re-running both sides.
