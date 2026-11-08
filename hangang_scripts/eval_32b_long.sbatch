#!/bin/bash
#SBATCH --job-name=eval32b
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Long-profile lighteval (lib/lighteval_bench.py _LONG_BENCHMARKS) for one 32B model on one node.
#   sbatch --job-name=eval32b-dense eval_32b_long.sbatch <model_dir> <name>
# Two tp=2 vLLM instances side by side: GPU0-1 runs lcb alone (32768 budget x 16 samples, by far the
# longest), GPU2-3 runs math500, gpqa, ifeval, gsm8k in turn. tp=1 would leave a 61GB model only ~20GB
# of KV per GPU (~3 concurrent lcb sequences).
MODEL=${1:?model dir}; NAME=${2:?name}; SMOKE=${3:+--max_samples $3}  # 3rd arg: samples per task (smoke test)
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null  # stale mount left by an earlier job on this node
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "squashfuse mount failed"; exit 1; }
trap "fusermount -uz $E" EXIT
source $U/jobs/env_common.sh
ulimit -c 0
# single node, no srun step -> no Slingshot VNI, so the forced aws-ofi plugin fails to open a CXI domain
# ("NET/OFI Couldn't create domain"); intra-node TP only needs NVLink/P2P
export NCCL_NET=Socket
OUT=$U/eval/$NAME
mkdir -p $OUT
cd $U/jobs
echo "model=$MODEL out=$OUT node=$(hostname)"
CUDA_VISIBLE_DEVICES=0,1 VLLM_PORT=29600 python -u eval_long.py run --model_dir $MODEL --out_dir $OUT \
    --tasks lcb --tp 2 --tag lcb $SMOKE > $U/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_lcb.log 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 VLLM_PORT=29700 python -u eval_long.py run --model_dir $MODEL --out_dir $OUT \
    --tasks math500,gpqa,ifeval,gsm8k --tp 2 --tag rest $SMOKE > $U/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rest.log 2>&1 &
wait
WANDB_MODE=online python -u eval_long.py merge --model_dir $MODEL --out_dir $OUT --run_name eval_long_$NAME
echo "=== EVAL EXIT: $? ==="
