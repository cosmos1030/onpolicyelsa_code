#!/bin/bash
#SBATCH --job-name=sal_scout_1.7b
#SBATCH --partition=A100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/doyoonkim/onpolicyelsa_code/elsa/logs/sal_scout_1.7b_%j.out
exec 2>&1

# Rebuttal SWVr (ii): SCOUT with ONLY the saliency swapped. Everything else is the
# 1.7B s70 SCOUT run ycs8ahvs (rollout 512, in-process vLLM, PGD grow-to-target under
# the KL budget, global scope), loaded verbatim from flags/scout_1.7b_s70_roll512_inproc.flags.
# Changes vs that run:
#   - --gmp_saliency: fisher (= SCOUT) | wanda (|w|*||x||) | obs (w^2/[H^-1]_jj,
#     SparseGPT's criterion without its weight update). Wanda/OBS scores are ranked
#     within each layer (per-layer output-error units), so every layer keeps the same
#     sparsity; the KL bisection over k is unchanged.
#   - Wanda/OBS statistics: 128 x 2048 training-document tokens, refreshed with the KL
#     calibration batch (every mask_interval = 32 steps).
#   - --gmp_opkd_legacy_partial_sync=true: the in-process engine on the current code
#     reproduces the pre-856f814 partial weight sync of the original run.
#   - KL budget: arg 2 (default 0.02 = SCOUT). Raised only if a criterion cannot reach
#     70% at a comparable step under 0.02.
#
# Usage: sbatch slurm_saliency_scout_qwen3_1.7b.sh <fisher|wanda|obs> [KL_BUDGET]
# Debug: DEBUG=1 sbatch slurm_saliency_scout_qwen3_1.7b.sh obs

SAL=${1:?"Usage: sbatch $0 <fisher|wanda|obs> [KL_BUDGET]"}
KLB=${2:-0.02}
# LAYER_RANK=true: rank fisher/magnitude within each layer too (uniform per-layer allocation, the
# control for Wanda/OBS, which are always ranked per layer)
LAYER_RANK=${LAYER_RANK:-false}
# RAW_GLOBAL=true: rank wanda/obs raw scores globally too (no within-layer rank)
RAW_GLOBAL=${RAW_GLOBAL:-false}
if [ "$RAW_GLOBAL" = "true" ]; then RANK_TAG="globalrank"; LAYER_RANK=false
elif [ "$LAYER_RANK" = "true" ] || [ "$SAL" = "wanda" ] || [ "$SAL" = "obs" ]; then RANK_TAG="layerrank"; else RANK_TAG="globalrank"; fi
REPO_ROOT="/home/doyoonkim/onpolicyelsa_code/elsa"
DATA_ROOT="/home/doyoonkim/projects/onpolicyelsa_code/elsa/data"
DATA_PATH="${DATA_ROOT}/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="${DATA_ROOT}/ot3_fineweb_200k_qwen3_opdprompts.jsonl"
MODEL="/home/shared/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
FLAGFILE="${REPO_ROOT}/scripts/log_cluster/flags/scout_1.7b_s70_roll512_inproc.flags"

if [ "${DEBUG:-0}" = "1" ]; then
    EXTRA="--steps=40 --gmp_pgd_nm_refit_calib_size=8 --save_model=false --push_to_hub=false --eval_full_bench=false --wandb=false --gmp_ckpt_every_steps=0"
    TAG="debug"
else
    EXTRA="--gmp_pgd_nm_refit_calib_size=128 --gmp_ckpt_every_steps=256"
    TAG="rebuttal"
fi

for f in "$DATA_PATH" "$OPD_PROMPT_PATH" "$MODEL/config.json" "$FLAGFILE"; do
    if [ ! -e "$f" ]; then echo "ERROR: missing $f" >&2; exit 1; fi
done

source /opt/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate rac

LOCAL_JOB_BASE="/tmp/${USER}/job_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_JOB_BASE/wandb" "${REPO_ROOT}/logs" "${REPO_ROOT}/models"

export WANDB_DIR="$LOCAL_JOB_BASE/wandb"
export WANDB_RUN_ID_OUTPUT="$LOCAL_JOB_BASE/wandb_run_id"
export WANDB_SERVICE_WAIT=300
export WANDB_INIT_TIMEOUT=120
export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || cat ~/.hf_token 2>/dev/null || echo "")
export WANDB_API_KEY=${WANDB_API_KEY:-$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)}
export HF_HOME=/home/shared/huggingface
export HF_HUB_DISABLE_XET=1
# in-process vLLM (sleep mode, CuMemAllocator) asserts on expandable_segments
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export ELSA_EVAL_SKIP_TASKS=${ELSA_EVAL_SKIP_TASKS:-gpqa}
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

echo "=== SCOUT saliency=${SAL} ${RANK_TAG} kl_budget=${KLB} Qwen3-1.7B s70 (${TAG}) ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  COMMIT=$(git -C "$REPO_ROOT" rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

cd "$REPO_ROOT"

python main.py \
    --flagfile="$FLAGFILE" \
    --model="$MODEL" \
    --data_path="$DATA_PATH" \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_save_path="${REPO_ROOT}/models" \
    --gmp_dpo_cache_dir="${REPO_ROOT}/.cache/dpo_chosen" \
    --gmp_ckpt_dir="${REPO_ROOT}/models/ckpt_sal_scout_${SAL}_${RANK_TAG}_klb${KLB}_1.7b_s70_${TAG}_${SLURM_JOB_ID}" \
    --gmp_resume_from="${RESUME_FROM:-}" \
    --gmp_saliency=${SAL} \
    --gmp_pgd_kl_budget=${KLB} \
    --gmp_pgd_layer_rank=${LAYER_RANK} \
    --gmp_pgd_raw_global_rank=${RAW_GLOBAL} \
    --gmp_opkd_vllm_sidecar=false \
    --gmp_opkd_legacy_partial_sync=true \
    --gmp_pgd_refit_calib_source=train \
    --gmp_pgd_nm_refit_seqlen=2048 \
    --run_name_suffix="rbsal_scout_${SAL}_${RANK_TAG}_klb${KLB}_roll512_inproc_legacysync_${TAG}" \
    ${EXTRA}

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
