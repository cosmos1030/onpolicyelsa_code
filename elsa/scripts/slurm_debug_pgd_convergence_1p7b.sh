#!/bin/bash
#SBATCH --job-name=dbg_pgd_conv_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/dbg_pgd_conv_1.7b_%j.out
exec 2>&1

# Diagnostic-only run (no eval/save/push): TR-GMP disabled entirely
# (--gmp_tr_enabled=false). mask_interval=1 + sparse_train_steps=steps-1
# (-> pruning_end_steps=1) forces the ENTIRE cubic ramp into a single
# step-1 snap straight to the final target sparsity (cubic fraction
# min(1,1)/1 = 1.0 -> current_sparsity=final_sparsity exactly at step 1)
# -- from step 2 onward step > pruning_end_steps so the schedule path
# never touches the mask again for the rest of the run. So there is
# exactly ONE growth-driven mask change (the unavoidable dense->target
# initialization), then PGD (--gmp_pgd=true, uncapped -- no
# kl_share/kl_budget/max_swap_frac set) is the ONLY thing moving the mask
# for the remaining ~2047 steps, at a truly fixed keep-count throughout.
# Purpose: does pgd/revivals + pgd/prunings ever decay toward 0 (mask
# actually converging) with literally nothing external ever disturbing
# the target again, or does it keep climbing/oscillating regardless.

SPARSITY=${1:-0.1}
STEPS=${2:-2048}
SPARSE_TRAIN_STEPS=$((STEPS - 1))

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb"
mkdir -p /home1/doyoonkim/projects/elsa/logs

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== [DEBUG] PGD-only mask convergence probe: Qwen3-1.7B fixed sparsity=${SPARSITY}, TR disabled, ramp completes by step ~$((STEPS - SPARSE_TRAIN_STEPS)), then ${SPARSE_TRAIN_STEPS} steps of PGD-only reprojection at fixed target ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_sparse_train_steps=${SPARSE_TRAIN_STEPS} \
    --gmp_dense_warmup_steps=0 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_mask_interval=1 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=8192 \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.33 \
    --gmp_kd_lambda=0.33 \
    --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 \
    --gmp_onpolicy_max_new_tokens=256 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cubic \
    --gmp_pgd=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=debug_pgd_convergence \
    --run_name_suffix="dbg_pgd_conv_s$(python3 -c "print(int(${SPARSITY}*100))")pct_sparsetrain${SPARSE_TRAIN_STEPS}_uncapped" \
    --seed=42

echo "##### END #####"
