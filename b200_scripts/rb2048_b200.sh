#!/bin/bash
# One single-GPU Qwen3-1.7B run of the roll2048 rebuttal set on the review B200 box: same main.py
# arguments as hangang_scripts/rb2048_run.sh (sidecar vLLM = fixed sync path, rollout 2048, PGD KL
# screening on 4 whole rollout-pool sequences, full-vocab KL), flash-attn from the rac env.
# Resumes from the newest checkpoint in its ckpt dir. Long-profile eval is separate.
# usage: CUDA_VISIBLE_DEVICES=<gpu> rb2048_b200.sh <name> <sparsity> [extra main.py flags...]
N=${1:?name}; SP=${2:?sparsity}; shift 2
B=/NHNHOME/log-postech/doyoonkim
# a queued run moved to another machine: touch $B/logs/skip_<name> and the queue passes over it
[ -e $B/logs/skip_$N ] && { echo "=== $N skipped ($B/logs/skip_$N) ==="; exit 0; }
WT=$B/code_rb2048/elsa
CKPT=$B/ckpt/rb2048/$N
mkdir -p $CKPT $B/models_rb2048 $B/logs/wandb_rb2048
source $B/miniconda3/etc/profile.d/conda.sh
conda activate rac
export HF_TOKEN=$(cat $B/secrets/hf_token) WANDB_API_KEY=$(cat $B/secrets/wandb_api_key)
export HF_HOME=$B/.cache/huggingface WANDB_DIR=$B/logs/wandb_rb2048 WANDB_SERVICE_WAIT=300 WANDB_INIT_TIMEOUT=120
export TRITON_CACHE_DIR=$B/.cache/triton TORCHINDUCTOR_CACHE_DIR=$B/.cache/torchinductor VLLM_CACHE_ROOT=$B/.cache/vllm
export TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1 VLLM_NO_USAGE_STATS=1 TMPDIR=/tmp
export VLLM_PORT=$((30000 + 100 * ${CUDA_VISIBLE_DEVICES:-0}))
# newest stepNNNNNN.pt main file (not its _masks_rankN.pt companions)
RESUME=$(ls $CKPT/step[0-9][0-9][0-9][0-9][0-9][0-9].pt 2>/dev/null | sort | tail -1)
echo "=== $N sp=$SP gpu=$CUDA_VISIBLE_DEVICES resume=[${RESUME}] extra=[$*] ==="
cd $WT
python -u main.py \
    --model=Qwen/Qwen3-1.7B \
    --dataset=mixed_cot \
    --data_path=$B/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --sparsity_ratio=$SP --sparsity_type=unstructured --do_gmp=true --steps=2048 \
    --gmp_batch_size=1 --gmp_grad_accum=8 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 --gmp_mask_interval=32 --gmp_fisher_beta=0.999 --gmp_saliency=fisher \
    --gmp_pruning_scope=global --seqlen=8192 --gmp_gradient_checkpointing=true \
    --gmp_ckpt_every_steps=256 --gmp_ckpt_dir=$CKPT --gmp_resume_from=$RESUME \
    --gmp_max_prompt_len=512 --gmp_kd_only=false --gmp_kl_chunk_size=256 --kd_nsamples=0 \
    --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=2048 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.35 --gmp_opkd_vllm_sidecar=true \
    --gmp_prompt_path=$B/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
    --gmp_pgd_jump_to_target=false --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4 --gmp_pgd_interval=8 \
    --gmp_pgd_screen_source=rollout \
    --gmp_tr_kl_hook=false \
    --gmp_save_path=$B/models_rb2048 --save_model=true --push_to_hub=false \
    --eval_math500=false --eval_full_bench=false --eval_zero_shot=false \
    --wandb=true --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    "$@" \
    --run_name_suffix=${N}_b200_sidecar_roll2048 \
    --seed=42
echo "=== main.py EXIT: $? ==="
