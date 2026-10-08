#!/bin/bash
#SBATCH --job-name=awq_rtn
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/awq_rtn_%j.out
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
exec 2>&1
# RTN / AWQ+RTN / AWQ-transformed dense checkpoints for one model + bit-width
# (see awq_rtn_qwen3.py). Usage: sbatch slurm_awq_rtn_qwen3.sh <1.7b|4b> <BITS>
SIZE=${1:?"Usage: sbatch slurm_awq_rtn_qwen3.sh <1.7b|4b> <BITS>"}
BITS=${2:?"Usage: sbatch slurm_awq_rtn_qwen3.sh <1.7b|4b> <BITS>"}
case "$SIZE" in
    1.7b) MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" ;;
    4b)   MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" ;;
    *) echo "unknown size $SIZE"; exit 1 ;;
esac
M=/home1/doyoonkim/projects/elsa/models
export TOKENIZERS_PARALLELISM=false HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== AWQ/RTN Qwen3-${SIZE} w${BITS} g128 ==="; echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"
cd /home1/doyoonkim/projects/elsa
/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/awq_rtn_qwen3.py --model "$MODEL" --bits $BITS --group 128 \
    --nsamples 64 --seqlen 2048 --n_tok 4096 --seed 42 \
    --out_rtn $M/qwen3_${SIZE}_rtn_w${BITS}_g128 \
    --out_awq_dense $M/qwen3_${SIZE}_awq_dense_w${BITS}_g128 \
    --out_awq_rtn $M/qwen3_${SIZE}_awq_rtn_w${BITS}_g128
EXIT_CODE=$?
echo "=== Exit code: $EXIT_CODE ==="
exit $EXIT_CODE
