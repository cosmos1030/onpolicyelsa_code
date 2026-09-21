#!/bin/bash
# "w/o rollout refresh" (frozen rollout pool) at HIGH sparsity.
#
# The published w/o-refresh point is 4B/70% only, and there it costs nothing
# (49.20 vs SCOUT 48.81 +- 0.41 -- inside seed noise). That single point is
# what makes the on-policy framing look unsupported: if a pool filled once
# from the DENSE model matches one refreshed every 32 steps, "on-policy"
# reduces to "trained on generated text" rather than "trained on the current
# policy's text". The 512-token rollout length is the likely reason -- over
# 512 tokens a 70% model has not drifted far from the dense model the pool
# came from. At 80% it has (math500 46.1 vs 78.2), so if freshness matters
# anywhere, it matters here.
#
# Not a new method and not a new hyper-parameter: every value below is copied
# from the 4B s80 SCOUT run this has to be compared against (wandb q0kmiudv ->
# cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740) -- lr 1e-4, kl_budget
# 0.02, pgd_interval 8, mask_interval 32, steps 2048, warmup 256, grad_accum 8,
# OPD gen len 512, loss 0.33/0.33/0.33. The ONLY change is
# gmp_onpolicy_kd_interval: 32 -> 4096.
#
# What 4096 buys (gmp_trainer.py:4404): `onpolicy_interval >= total_steps`
# takes a dedicated branch that sizes the pre-loop pool to
# total_steps*grad_accum*world_size instead of mask_interval*grad_accum, so
# the frozen arm sees the SAME 16,384 rollouts the refreshed baseline does and
# staleness is not confounded with a 64x cut in unique rollouts. The fill runs
# before the training loop, i.e. while the model is still dense -- that is
# what makes the pool Q0 = P.
#
# This container has no SLURM -- run with bash.
#
# Usage: bash b200_scripts/gmp_frozenpool_qwen3_4b.sh [SPARSITY] [KL_BUDGET] [LR]
#   bash b200_scripts/gmp_frozenpool_qwen3_4b.sh        # s80, the point we need
#   bash b200_scripts/gmp_frozenpool_qwen3_4b.sh 0.7    # reproduce the published 70% arm
set -e

SPARSITY=${1:-${SPARSITY:-0.8}}
KL_BUDGET=${2:-${KL_BUDGET:-0.02}}
LR=${3:-${LR:-1e-4}}
STEPS=${STEPS:-2048}
ROLLOUT_INTERVAL=${ROLLOUT_INTERVAL:-4096}

# >= STEPS is the entire point; below it this silently becomes an ordinary
# refreshed run that looks identical in the logs until you read the pool size.
if [ "$ROLLOUT_INTERVAL" -lt "$STEPS" ]; then
    echo "ERROR: ROLLOUT_INTERVAL=$ROLLOUT_INTERVAL < STEPS=$STEPS -- that is a refreshed run, not the frozen-pool arm." >&2
    exit 2
fi

# The parent puts kl_budget/lr/pgd_interval in run_name_suffix but NOT the
# rollout interval, so without this the frozen arm and the refreshed SCOUT run
# collide on one name and one checkpoint stem.
export TAG_SUFFIX="${TAG_SUFFIX:-_ri${ROLLOUT_INTERVAL}}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== frozen rollout pool (w/o refresh) Qwen3-4B s${SPARSITY} ==="
echo "    rollout_interval=${ROLLOUT_INTERVAL} >= steps=${STEPS}  -> one dense fill, never refreshed"
echo "    matched to SCOUT s80 (q0kmiudv): klb=${KL_BUDGET} lr=${LR} pgdi=8 mask_interval=32"

#  1 SPARSITY  2 KL_BUDGET  3 OPD_GEN_LEN  4 MASK_INTERVAL  5 LR_SCHEDULER
#  6 STEPS  7 LR  8 DATA_PATH  9 SEQLEN  10 GRAD_CKPT  11 WANDB_PROJECT
#  12 SALIENCY  13 PRUNING_SCOPE  14 LOSS_WEIGHTS  15 ROLLOUT_INTERVAL
#  16 KD_NSAMPLES  17 CALIB_SIZE  18 PGD_INTERVAL  19 VLLM_GPU_MEM
bash "$HERE/gmp_pgd_grow_to_target_qwen3_4b.sh" \
    "${SPARSITY}" "${KL_BUDGET}" 512 32 cosine "${STEPS}" "${LR}" \
    /NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    8192 true reasoning_qwen3_4b_nostrip8192 fisher global 0.33,0.33,0.33 \
    "${ROLLOUT_INTERVAL}" 0 4 8 0.15
