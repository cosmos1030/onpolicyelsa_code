#!/bin/bash
#SBATCH --job-name=pregen_tgkd_1.7b
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n87,n91,n61,n64,n31,n19
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/pregen_tgkd_1.7b_%j.out
exec 2>&1

# Pre-generates the teacher-gen-KD (TGKD) chosen-continuation cache for the
# upcoming TR-GMP TGKD(forward-KL) + OPD(reverse-KL) combined run, via vLLM,
# so the real training job loads it instantly instead of generating 16384
# continuations inline. Uses ot3_fineweb_200k_qwen3_train.jsonl (first 180k
# of the 200k-line corpus) -- disjoint from
# ot3_fineweb_200k_qwen3_opdprompts.jsonl (last 20k lines), which the OPD
# on-policy rollouts will draw from (via --gmp_prompt_path), so TGKD and OPD
# never see the same prompts.
#
# n_pairs/gbs must match the real training job's steps/batch_size/grad_accum
# exactly -- the cache key includes data_path + gbs, so a mismatch means the
# real job won't find this cache and regenerates from scratch.

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl"

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_HOME="/home1/doyoonkim/.cache/huggingface"
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd /home1/doyoonkim/projects/elsa
export PYTHONPATH="/home1/doyoonkim/projects/elsa:${PYTHONPATH}"

/home1/doyoonkim/miniconda3/envs/rac/bin/python scripts/pregenerate_tgkd_cache.py \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --steps=2048 \
    --batch_size=1 \
    --grad_accum=8 \
    --seqlen=2048 \
    --max_prompt_len=512 \
    --max_new_tokens=512 \
    --temperature=0.7 \
    --cache_dir=/home1/doyoonkim/projects/elsa/.cache/dpo_chosen

echo "##### END #####"
