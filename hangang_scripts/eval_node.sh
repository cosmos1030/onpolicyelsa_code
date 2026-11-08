#!/bin/bash
# Per-node body of eval_32b_4models.sbatch: evaluate the SLURM_NODEID-th "name:model_dir" line of EVAL_MODELS.
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null  # stale mount left by an earlier job on this node
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "[node $SLURM_NODEID] squashfuse mount failed"; exit 1; }
trap "fusermount -uz $E" EXIT
source $U/jobs/env_common.sh
ulimit -c 0
# TP stays inside the node (NVLink/P2P); the aws-ofi plugin is only needed for inter-node training
export NCCL_NET=Socket
line=$(echo "$EVAL_MODELS" | sed -n "$((SLURM_NODEID+1))p")
NAME=${line%%:*}; MODEL=${line#*:}
[ -f "$MODEL/config.json" ] || { echo "[node $SLURM_NODEID] no model at $MODEL"; exit 1; }
SMOKE=${EVAL_SMOKE:+--max_samples $EVAL_SMOKE}
OUT=$U/eval/$NAME
mkdir -p $OUT
cd $U/jobs
L=$U/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_$NAME
echo "[node $SLURM_NODEID $(hostname)] $NAME <- $MODEL"
CUDA_VISIBLE_DEVICES=0,1 VLLM_PORT=29600 python -u eval_long.py run --model_dir $MODEL --out_dir $OUT \
    --tasks lcb --tp 2 --tag lcb $SMOKE > ${L}_lcb.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 VLLM_PORT=29700 python -u eval_long.py run --model_dir $MODEL --out_dir $OUT \
    --tasks math500,ifeval,gsm8k --tp 2 --tag rest $SMOKE > ${L}_rest.log 2>&1 &
wait
WANDB_MODE=online python -u eval_long.py merge --model_dir $MODEL --out_dir $OUT \
    --run_name ${EVAL_SMOKE:+smoke_}eval_long_$NAME > ${L}_merge.log 2>&1
echo "[node $SLURM_NODEID] $NAME done rc=$?"
grep "lighteval/" ${L}_merge.log | sed "s/^/[$NAME] /"
