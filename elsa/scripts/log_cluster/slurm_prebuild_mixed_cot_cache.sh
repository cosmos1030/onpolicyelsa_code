#!/bin/bash
#SBATCH --job-name=prebuild_cache
#SBATCH --partition=3090
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/prebuild_cache_%j.out
exec 2>&1

# CPU-only cache pre-builder for MixedTextDataset (log_cluster). Deliberately
# requests no --gres=gpu: this cluster has no dedicated CPU partition, but
# omitting --gres still lets SLURM schedule the job without reserving a GPU,
#
# --mem=128G: 32G OOM-killed (exit 137) at 91% on the 100%-OpenThoughts3
# dataset (ot3_100pct_qwen3.jsonl, no FineWeb-Edu mixed in -- every sample is
# a long CoT trace, unlike PLAIN/THINKSTRIP's 80/20 mix). Bumped up front
# rather than re-discovering this per dataset.
# so a plain tokenization pass doesn't tie up a GPU slot other jobs need.
#
# Usage: sbatch slurm_prebuild_mixed_cot_cache.sh [DATA_PATH]
# e.g.: sbatch slurm_prebuild_mixed_cot_cache.sh
#       sbatch slurm_prebuild_mixed_cot_cache.sh /path/to/other.jsonl

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

REPO_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa"
DATA_PATH=${1:-${REPO_ROOT}/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl}

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/home/shared/huggingface
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  DATA_PATH=$DATA_PATH"
python "${REPO_ROOT}/scripts/log_cluster/prebuild_mixed_cot_cache.py" "$DATA_PATH"
EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
