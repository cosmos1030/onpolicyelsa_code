# SparseGPT self-gen v3 sweep (2026-09-06 submit, jobs 870723-870730)
#
# Pruning COMPLETED and every checkpoint was saved + pushed. The jobs then ran
# out of their 12h wall inside lighteval's generative tasks (see below), so only
# the lighteval bench needs re-running -- via
#   elsa/scripts/slurm_resume_eval_sgpt_selfgen.sh \
#       <MODEL_DIR> <WANDB_PROJECT> <WANDB_RUN_ID> <SPARSITY>
#
# The wandb run ids below had to be recovered through the wandb API rather than
# read from the job logs: the sgpt launchers write --output to /local-data only
# and had no NFS-copy trap, so those logs became unreachable once the jobs
# ended. The resume launcher adds such a trap.
#
# Why they timed out: cpus-per-task was 16 in these launchers and was lowered to
# 8 on 2026-09-07 while adding the CALIB arg, mechanically applying the
# "8 CPUs per GPU" rule. That rule is for TRAINING jobs; prune_and_eval.py's
# tail runs lighteval/vLLM, whose scheduler + (de)tokenizer are CPU-bound.
# Historical SparseGPT runs at 16 CPUs finished in 0:33-5:20 (4B) and
# 2:45-4:55 (1.7B); at 8 CPUs these were at 127-191 s/it and under half done
# after 9h50m. Reverted to 16, and the wall raised 12h -> 24h.
#
# model_dir                                          wandb_project          run_id     sparsity
elsa/models/qwen3_1.7b_sgpt_s50pct_n128_selfgenv3    reasoning_qwen3_1.7b   swnmbp8c   0.5
elsa/models/qwen3_1.7b_sgpt_s60pct_n128_selfgenv3    reasoning_qwen3_1.7b   ne8ou556   0.6
elsa/models/qwen3_1.7b_sgpt_s70pct_n128_selfgenv3    reasoning_qwen3_1.7b   0q71fs3p   0.7
elsa/models/qwen3_4b_sgpt_s50pct_n128_selfgenv3      reasoning_qwen3_4b     fbjdz1v4   0.5
elsa/models/qwen3_4b_sgpt_s60pct_n128_selfgenv3      reasoning_qwen3_4b     wzgmev4f   0.6
elsa/models/qwen3_4b_sgpt_s70pct_n128_selfgenv3      reasoning_qwen3_4b     uw9vc7du   0.7
elsa/models/qwen3_8b_sgpt_s70pct_n128_selfgenv3      reasoning_qwen3_8b     xlsykpao   0.7
