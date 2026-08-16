#!/bin/bash
#SBATCH --job-name=mlp_gnpcg_full
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91,n61,n64
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/mlp_gnpcg_full_%j.out
exec 2>&1

# Full-model MLP-joint Gauss-Newton PCG refinement (all decoder layers,
# sequential propagation) applied on top of an already-pruned ALPS
# checkpoint, mask fixed -- then a full reasoning-bench eval so it's
# directly comparable to the existing ALPS (layer-wise-PCG-only) row in the
# artifact for the same sparsity.
#
# Usage: sbatch slurm_mlp_joint_gnpcg_full.sh <SPARSITY_PCT> <ALPS_MODEL_DIR> <PCG_ITERS>
# e.g.: sbatch slurm_mlp_joint_gnpcg_full.sh 50 /home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s50pct 10

SPARSITY_PCT=${1:?"Usage: <SPARSITY_PCT> <ALPS_MODEL_DIR> <PCG_ITERS>"}
ALPS_MODEL=${2:?"Usage: <SPARSITY_PCT> <ALPS_MODEL_DIR> <PCG_ITERS>"}
PCG_ITERS=${3:-10}

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
SAVE_DIR="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s${SPARSITY_PCT}pct_mlpgnpcg_iters${PCG_ITERS}"

export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1

echo "=== MLP-joint GN-PCG (all layers) Qwen3-1.7B s${SPARSITY_PCT}pct pcg_iters=${PCG_ITERS} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/ALPS

$PYTHON apply_mlp_joint_gnpcg.py "$DENSE_MODEL" "$ALPS_MODEL" \
    --data_path "$DATA" \
    --save "$SAVE_DIR" \
    --pcg_iters ${PCG_ITERS} \
    --gn_outer 1 \
    --damping 1e-3 \
    --nsamples 128 \
    --heldout 64 \
    --seqlen 2048 \
    --push_to_hub "cosmos1030/alps-s${SPARSITY_PCT}pct-mlpgnpcg-iters${PCG_ITERS}"

echo "=== refinement exit: $? ==="
echo "=== starting eval_full.py ==="

cd /home1/doyoonkim/projects/elsa

$PYTHON scripts/eval_full.py \
    --model_path "$SAVE_DIR" \
    --wandb_project reasoning_qwen3_1.7b_nostrip8192 \
    --run_name "alps_s${SPARSITY_PCT}pct_mlpgnpcg_iters${PCG_ITERS}" \
    --method gmp \
    --sparsity 0.${SPARSITY_PCT} \
    --gpu_util 0.85 \
    --profile quick

echo "=== eval exit: $? ==="
echo "##### END #####"
