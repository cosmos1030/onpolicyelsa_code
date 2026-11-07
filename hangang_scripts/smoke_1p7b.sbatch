#!/bin/bash
#SBATCH --job-name=scout-smoke
#SBATCH --partition=natStr
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment="field=efficientai;appl=pytorch-fsdp"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Hangang (GH200) smoke test of the SCOUT 32B layout on Qwen3-1.7B: 3 FSDP training ranks (GPU0-2),
# vLLM rollout server on its own GPU3 (gmp_opkd_vllm_gpu_index=-1 -> index world_size), teacher FSDP-sharded.
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null  # stale mount left by an earlier job on this node
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "squashfuse mount failed"; exit 1; }
trap "fusermount -uz $E" EXIT
# driver 570 = CUDA 12.8; torch cu129 / vLLM kernels need the 12.9 forward-compat libcuda (575)
export LD_LIBRARY_PATH=/apps/NVIDIA/hpc_sdk/Linux_aarch64/26.3/cuda/12.9/compat:${LD_LIBRARY_PATH:-}
export PATH=$E/bin:$PATH PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME=/tmp/hf_home TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor VLLM_CACHE_ROOT=/tmp/vllm_cache TMPDIR=/tmp
export ELSA_ATTN_IMPL=sdpa VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True NCCL_DEBUG=WARN WANDB_MODE=disabled
export VLLM_SERVER_LOG_DIR=$U/logs
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
cd $U/code/elsa
torchrun --nproc_per_node=3 --master_port=29511 main.py \
    --model=$U/models/Qwen3-1.7B --dataset=mixed_cot \
    --data_path=$U/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --gmp_prompt_path=$U/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --sparsity_ratio=0.7 --sparsity_type=unstructured --do_gmp=true --gmp_use_fsdp=true --gmp_teacher_fsdp=true \
    --steps=24 --gmp_batch_size=1 --gmp_grad_accum=4 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=4 \
    --gmp_warmup_ratio=0.05 --gmp_mask_interval=8 --gmp_fisher_beta=0.999 --gmp_saliency=fisher --gmp_pruning_scope=global \
    --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_kl_chunk_size=256 --gmp_max_prompt_len=512 \
    --gmp_kd_only=false --kd_nsamples=0 --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=8 --gmp_onpolicy_max_new_tokens=512 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.5 --gmp_opkd_vllm_gpu_index=-1 \
    --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4 --gmp_pgd_interval=4 \
    --save_model=false --push_to_hub=false --eval_math500=false --eval_full_bench=false --eval_zero_shot=false \
    --wandb=false --seed=42 --run_name_suffix=hangang_smoke
echo "=== main.py EXIT: $? ==="
