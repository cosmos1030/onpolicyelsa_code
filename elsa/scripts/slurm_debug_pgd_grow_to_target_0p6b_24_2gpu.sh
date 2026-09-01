#!/bin/bash
#SBATCH --job-name=debug_pgd_grow_24_fsdp2
#SBATCH --partition=RTX3090
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/debug_pgd_grow_24_fsdp2_%j.out
exec 2>&1

# FSDP (2-GPU) correctness check for --gmp_pgd_grow_to_target's N:M path --
# validated single-GPU only so far (job 828652, 1.7B, 100 steps, 0 invariant
# violations, independently re-verified from the saved safetensors). This
# checks the SAME code under real multi-rank FSDP, where two bugs were
# caught by design review before ever running this (not by a crash --
# _pgd_nm_finished_swap_build gathers to a FULL, rank-IDENTICAL tensor for
# the finished-group score/candidate pools, unlike _pgd_nm_directional's
# genuinely-per-rank-local output; treating the finished pool the same way
# as the local one would (1) double/multiply-count candidates by
# world_size via a redundant all_reduce on already-identical data, and (2)
# let each rank's tie-breaking torch.rand draw diverge independently even
# on identical input, silently selecting DIFFERENT groups per rank -- fixed
# by not all_reducing _n_c_t and by seeding _pgd_topk_groups_from_scores's
# tie-break with `step` so every rank draws the same random sequence).
#
# Cheap/small on purpose: Qwen3-0.6B, 2xRTX3090, 40 steps, seqlen=1024, no
# eval/save -- just confirms the run doesn't hang/deadlock/crash and that
# _pgd_nm_check_invariant + the whole-step KL re-verification never fire
# under real 2-rank FSDP.

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
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
export NCCL_DEBUG=WARN

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")

echo "=== debug: gmp_pgd_grow_to_target N:M (2:4) under FSDP, 2xGPU (Qwen3-0.6B) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa

$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} main.py \
    --model="$MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --kd_nsamples=256 \
    --sparsity_ratio=0.5 \
    --sparsity_type=2:4 \
    --do_gmp=true \
    --gmp_use_fsdp=true \
    --steps=40 \
    --gmp_batch_size=1 \
    --gmp_grad_accum=4 \
    --lr=1e-4 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=4 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=8 \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=fisher \
    --gmp_pruning_scope=global \
    --seqlen=1024 \
    --gmp_gradient_checkpointing=true \
    --gmp_kl_chunk_size=512 \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=false \
    --gmp_ntp_lambda=0.5 \
    --gmp_kd_lambda=0.5 \
    --gmp_tr_enabled=false \
    --gmp_growth_schedule=cosine \
    --gmp_pruning_end_ratio=0.0 \
    --gmp_pgd=true \
    --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 \
    --gmp_pgd_kl_calib_size=4 \
    --gmp_pgd_kl_calib_seqlen=256 \
    --gmp_pgd_kl_bisect_iters=6 \
    --gmp_pgd_interval=1 \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=debug_pgd_convergence \
    --run_name_suffix="debug_pgd_grow_to_target_24_fsdp2_0p6b" \
    --seed=42

EXIT_CODE=$?
echo "=== TORCHRUN EXIT: $EXIT_CODE ==="
echo "##### END #####"
exit $EXIT_CODE
