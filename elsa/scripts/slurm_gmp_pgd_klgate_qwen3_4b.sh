#!/bin/bash
#SBATCH --job-name=gmp_pgd_klgate_4b
#SBATCH --partition=H200-PCIe-ZT
#SBATCH --qos=zt
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=n89,n90,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/gmp_pgd_klgate_4b_%j.out
exec 2>&1

# Qwen3-4B version of slurm_gmp_pgd_klgate_qwen3_1.7b.sh -- same TR-GMP
# KL-gated growth + PGD-KL-gated per-step reprojection (--gmp_pgd_pgd=true,
# --gmp_pgd_kl_budget bisects how many lowest-importance prune candidates to
# accept each step so self-KL(pre||post-prune) on a small cached calibration
# batch stays within budget; revive count always set equal to accepted prune
# count). PGD/growth/TR logic itself is untouched -- only the SBATCH
# resource footprint and MODEL path differ from the 1.7B script, following
# the existing Qwen3-4B single-GPU (no FSDP) convention used by
# slurm_gmp_tr_ntpkd_opkd_24_qwen3_4b_pgd.sh and slurm_gmp_tr_opkd_dense_qwen3_4b.sh
# (H200-PCIe-ZT/zt, cpus=16, no FSDP needed -- only 8B needed FSDP).
#
# Usage: sbatch slurm_gmp_pgd_klgate_qwen3_4b.sh <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT] [SALIENCY] [PRUNING_SCOPE] [LOSS_WEIGHTS] [SPARSITY_TYPE] [L1_LAMBDA] [ROLLOUT_INTERVAL] [KD_NSAMPLES] [CALIB_SIZE] [DEBUG_IMPORTANCE_HIST] [PGD_INTERVAL] [PGD_POST_TARGET_ONLY]
# e.g.: sbatch slurm_gmp_pgd_klgate_qwen3_4b.sh 0.5 0.001 0.02 256 32 cosine 2048 0 1e-4 \
#         /home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl 8192 true reasoning_qwen3_4b_nostrip8192

SPARSITY=${1:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]"}
KL_BUDGET=${2:?"Usage: <SPARSITY> <KL_BUDGET> <KL_THRESHOLD> [OPD_GEN_LEN] [MASK_INTERVAL] [LR_SCHEDULER] [STEPS] [POST_TARGET_STEPS] [LR] [DATA_PATH] [SEQLEN] [GRAD_CKPT] [WANDB_PROJECT]"}
KL_THRESHOLD=${3:-0.02}
OPD_GEN_LEN=${4:-256}
MASK_INTERVAL=${5:-32}
LR_SCHEDULER=${6:-cosine}
STEPS=${7:-2048}
POST_TARGET_STEPS=${8:-0}
LR=${9:-1e-4}
DATA_PATH_ARG=${10:-/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl}
SEQLEN=${11:-8192}
GRAD_CKPT=${12:-true}
WANDB_PROJECT=${13:-reasoning_qwen3_4b_nostrip8192}
SALIENCY=${14:-fisher}
PRUNING_SCOPE=${15:-global}
LOSS_WEIGHTS=${16:-0.33,0.33,0.33}  # NTP,KD,OPKD -- e.g. 0,0.5,0.5 to drop NTP and split KD/OPKD evenly
SPARSITY_TYPE=${17:-unstructured}   # unstructured | 2:4 | 4:8
L1_LAMBDA=${18:-0.0}                # gmp_l1_lambda -- structured-L1 pre-shrink for N:M endgame (0=off)
ROLLOUT_INTERVAL=${19:-${MASK_INTERVAL}}  # gmp_onpolicy_kd_interval -- defaults to mask_interval (old behavior); set lower (e.g. half) to refresh the on-policy rollout more often than the mask changes, for staleness sensitivity checks under PGD
KD_NSAMPLES=${20:-0}  # gmp_kd_nsamples -- 0 = full dataset (production); set small (e.g. 256) for fast debug/smoke-test tokenization instead of the full 40k-sample cache build
CALIB_SIZE=${21:-4}  # gmp_pgd_kl_calib_size -- number of sequences in PGD's self-KL calibration batch (default 4). Larger dilutes the influence of any single near-deterministic/high-confidence token that would otherwise dominate the mean KL and collapse k_actual to 0.
DEBUG_IMPORTANCE_HIST=${22:-false}  # gmp_pgd_debug_importance_hist -- purely diagnostic (dumps fisher*weight^2 quantile/density every 5 steps), costs ~0.6s/step amortized. Off by default; set true only when actively debugging PGD churn/threshold placement.
PGD_INTERVAL=${23:-1}  # gmp_pgd_interval -- run PGD's reprojection (the dominant per-step cost) only every Nth step, decoupled from MASK_INTERVAL's growth cadence. Default 1 = every recovery step (prior behavior).
PGD_POST_TARGET_ONLY=${24:-false}  # gmp_pgd_post_target_only -- only let PGD reproject once TR-GMP growth has reached final_sparsity, isolating PGD's post-target maintenance role from its during-growth-ramp role. Default false = PGD fires from step 1 onward as before.
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
SPARSITY_PCT=$(python3 -c "print(int(${SPARSITY}*100))")

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
DATA_PATH="$DATA_PATH_ARG"
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
# NOTE: expandable_segments is incompatible with vLLM's CuMemAllocator,
# which the OPKD vLLM engine now requires (enable_sleep_mode=True, added in
# the 2026-08-13 log_cluster pull) -- LLM(...) hard-asserts on this at
# load_model() time if set. Use max_split_size_mb instead -- a different
# fragmentation mitigation the CuMemAllocator assertion doesn't check for
# (it only greps for the literal string "expandable_segments:True") --
# leaving fragmentation completely unmitigated caused a real OOM after
# ~760 steps on the 1.7B single-GPU 2:4 canary (720073); same mitigation
# applies here since PGD's extra calibration forward passes add the same
# kind of fragmentation pressure regardless of model size.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0
# Disables vLLM's background usage-reporting thread (report_usage() ->
# _report_continuous_usage, a `while True: time.sleep(600)` loop) --
# observed to cause a rare but reproducible interpreter-level crash
# ("Fatal Python error: none_dealloc: deallocating None") deep into
# training, always right at a vLLM wake_up() call. Not needed for a
# research training loop.
export VLLM_NO_USAGE_STATS=1
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== TR-GMP NTP+KD+OPKD(${NTP_LAMBDA}/${KD_LAMBDA}/${OPKD_LAMBDA}) Qwen3-4B s${SPARSITY_PCT} PGD-KL-budget(self-KL per step<=${KL_BUDGET}, tr_kl=${KL_THRESHOLD}) lr=${LR} mask_interval=${MASK_INTERVAL} rollout_interval=${ROLLOUT_INTERVAL} lr_scheduler=${LR_SCHEDULER} steps=${STEPS} post_target_steps=${POST_TARGET_STEPS} saliency=${SALIENCY} (OT80/FW20) ==="
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
    --sparsity_type=${SPARSITY_TYPE} \
    --gmp_l1_lambda=${L1_LAMBDA} \
    --do_gmp=true \
    --steps=${STEPS} \
    --gmp_post_target_steps=${POST_TARGET_STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=${LR} \
    --lr_scheduler=${LR_SCHEDULER} \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --gmp_fisher_beta=0.999 \
    --gmp_saliency=${SALIENCY} \
    --gmp_pruning_scope=${PRUNING_SCOPE} \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=${GRAD_CKPT} \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --kd_nsamples=${KD_NSAMPLES} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_kd_interval=${ROLLOUT_INTERVAL} \
    --gmp_onpolicy_max_new_tokens=${OPD_GEN_LEN} \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.15 \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_tr_enabled=true \
    --gmp_tr_delta_init=0.05 \
    --gmp_tr_delta_min=0.001 \
    --gmp_tr_kl_threshold=${KL_THRESHOLD} \
    --gmp_tr_kl_reduce=mean \
    --gmp_pgd=true \
    --gmp_pgd_kl_budget=${KL_BUDGET} \
    --gmp_pgd_kl_calib_size=${CALIB_SIZE} \
    --gmp_pgd_debug_importance_hist=${DEBUG_IMPORTANCE_HIST} \
    --gmp_pgd_interval=${PGD_INTERVAL} \
    --gmp_pgd_post_target_only=${PGD_POST_TARGET_ONLY} \
    --gmp_pgd_skip_growth_step=true \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=true \
    --push_to_hub=true \
    --eval_math500=false \
    --eval_full_bench=true \
    --eval_zero_shot=true \
    --wandb=true \
    --wandb_project=${WANDB_PROJECT} \
    --run_name_suffix="${RUN_TAG:+${RUN_TAG}_}pgd_klbudget${KL_BUDGET}_skipgrowth_lr${LR}_mi${MASK_INTERVAL}_ro${ROLLOUT_INTERVAL}_kl${KL_THRESHOLD}_${PRUNING_SCOPE}scope_$(basename "$DATA_PATH" .jsonl)" \
    --seed=42

echo "##### END #####"
