#!/bin/bash
#SBATCH --job-name=saliency_diag_pilot
#SBATCH --partition=RTX3090
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-02:00:00
#SBATCH --exclude=n3,n42,n51,n54,n60,n77,n80,n91
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/saliency_diag_pilot_%j.out
exec 2>&1

# Cheap single-snapshot real-model saliency comparison, on the same ALPS-s70
# frozen-mask setup as slurm_pgd_ablation_alps_s70_pilot.sh, but on RTX3090
# (much shorter queue than A100-80GB) since this needs no meaningful training
# -- just enough steps for the OPKD vLLM rollout pool to fill once (so the
# "on-policy" candidate is a REAL student rollout, not a fallback to the
# fixed reference batch) plus real Adam state for the baseline candidate,
# then a single diagnostic pass (saliency_snapshot_diagnostic in
# gmp_trainer.py) that measures the ACTUAL self-KL each of 5 candidate
# saliencies (baseline composite-Adam / magnitude / fixed-Fisher /
# on-policy-Fisher / 75:25 fixed+on-policy mix) would cause if used to pick
# the same k lowest-scored weights, via the same _compute_tr_kl forward-pass
# primitive PGD's own kl-budget bisection uses. Exits immediately after
# logging -- no full training run, no eval.
#
# Ported from a CPU toy (scout_cpu_saliency_poc.zip) that suggested on-policy
# Fisher / fixed+on-policy mixing beats the current composite-Adam saliency
# on random-group KL correlation -- but a fuller CPU sweep (5 seeds, 200
# groups) showed magnitude actually wins there, which contradicts what we
# already know from real Qwen runs (magnitude/wanda underperform badly) --
# so the toy alone doesn't settle anything. This is the real-model check.
#
# Usage: sbatch slurm_saliency_diag_alps_s70_pilot.sh [DIAG_STEP] [K] [MC_NSAMPLES] [MASK_INTERVAL] [VLLM_GPU_MEM] [SEQLEN] [MODE] [N_GROUPS]
#   DIAG_STEP: must be > MASK_INTERVAL so the OPKD pool has filled at least once (default 40, mask_interval=32).
#              MODE=corr accepts a comma-separated list (e.g. "40,80,120") -- runs the
#              diagnostic at each step within this ONE run instead of relaunching per step.
#   MODE: diag (default, single bottom-k point estimate, single-int DIAG_STEP only) |
#         corr (random-group Spearman correlation, statistically sounder, supports multi-step)

DIAG_STEP=${1:-40}
K=${2:-4096}
# outer MC-sample count; total backward passes = MC_NSAMPLES * calib batch
# size (4) -- CPU-toy evidence says total samples need to reach ~16-32 to
# clearly beat the Adam baseline (N<8 total underperforms it), so 8*4=32
MC_NSAMPLES=${3:-8}
MASK_INTERVAL=${4:-32}
# 0.15 (copied from the A100-80GB pilot) starves vLLM's KV cache on a 24GB
# RTX3090: 0.15*23.56GB=3.53GB budget vs 3.22GB just for model weights,
# leaving ~0GB for KV cache -> vLLM engine init hard-fails
# ("No available memory for the cache blocks"). 0.35 leaves ~5GB for KV
# cache after weights, plenty for the small OPKD rollout pool this pilot uses.
VLLM_GPU_MEM=${5:-0.35}
# 8192 (matching the production pilot's calibration-window convention) makes
# every full-vocab KL call in the anchor-KD path (~152k vocab x 8192 tokens,
# ~5GB per (B,T,V) tensor) OOM on both a 24GB RTX3090 and a 40GB A100 --
# this diagnostic doesn't need long-context calibration, so a short seqlen
# cuts that memory ~4x and sidesteps the OOM entirely instead of needing
# --gmp_kl_chunk_size.
SEQLEN=${6:-2048}
MODE=${7:-diag}  # diag | corr
N_GROUPS=${8:-20}
# "s70" (default) = ALPS-pruned s70 frozen-mask setup (original pilot design).
# "dense" = plain dense Qwen3-1.7B, sparsity_ratio=0, no mask/PGD at all --
# matches the CPU-toy reference's own setup exactly (that toy never prunes
# anything either), which sidesteps two real confounds the s70 case has: (1)
# the alive-only restriction needed there isn't needed here (nothing's dead),
# (2) no circularity from PGD having used baseline_composite_adam scores to
# select which weights survived to be measured in the first place.
STUDENT=${9:-s70}
CORR_SEED=${10:-0}  # gmp_saliency_corr_seed -- RNG seed for random-group sampling in MODE=corr
EMA_EVERY_STEP=${11:-false}  # gmp_saliency_ema_every_step -- expensive (extra forward+backward EVERY step), only for direct apples-to-apples EMA-depth testing
LOSS_WEIGHTS=${12:-0.33,0.33,0.33}  # NTP,KD,OPKD -- e.g. 0,0.5,0.5 to test whether Adam's exp_avg_sq (baseline_composite_adam) loses its saliency signal quality without NTP's hard-label empirical-Fisher contribution
NTP_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f1)
KD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f2)
OPKD_LAMBDA=$(echo "$LOSS_WEIGHTS" | cut -d, -f3)
KD_ONLY=$(python3 -c "print('true' if float('${NTP_LAMBDA}')==0.0 else 'false')")
NTP_EMA_EVERY_STEP=${13:-false}  # gmp_saliency_ntp_ema_every_step -- decoupled-saliency test: track NTP-only grad^2 EMA regardless of NTP_LAMBDA, for mask-selection use while training (optionally) drops NTP
EMA_NSAMPLES=${14:-1}  # gmp_saliency_ema_nsamples -- MC-Fisher samples per step inside EMA_EVERY_STEP; raise above 1 to test whether on_fisher_ema's per-sample multinomial-resampling variance (not just EMA depth) explains its underperformance vs baseline_composite_adam
LAST_STEP=$(echo "$DIAG_STEP" | tr ',' '\n' | sort -n | tail -1)
STEPS=$((LAST_STEP + 8))  # small safety margin past the last diag step; the run exits right after it regardless

if [ "$MODE" = "corr" ]; then
    DIAG_FLAGS="--gmp_saliency_corr_step=${DIAG_STEP} --gmp_saliency_corr_group_size=${K} --gmp_saliency_corr_groups=${N_GROUPS} --gmp_saliency_diag_mc_nsamples=${MC_NSAMPLES} --gmp_saliency_corr_seed=${CORR_SEED} --gmp_saliency_ema_every_step=${EMA_EVERY_STEP} --gmp_saliency_ntp_ema_every_step=${NTP_EMA_EVERY_STEP} --gmp_saliency_ema_nsamples=${EMA_NSAMPLES}"
else
    DIAG_FLAGS="--gmp_saliency_diag_step=${DIAG_STEP} --gmp_saliency_diag_k=${K} --gmp_saliency_diag_mc_nsamples=${MC_NSAMPLES}"
fi

ALPS_MODEL="/home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_s70pct"
DENSE_MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
if [ "$STUDENT" = "dense" ]; then
    STUDENT_MODEL="$DENSE_MODEL"
    SPARSITY=0.0
    PGD_ENABLED=false
else
    STUDENT_MODEL="$ALPS_MODEL"
    SPARSITY=0.7
    PGD_ENABLED=true
fi
DATA_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl"
OPD_PROMPT_PATH="/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

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
export VLLM_USE_V1=0
export VLLM_HOST_IP=127.0.0.1
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== Saliency diagnostic (mode=${MODE}): ALPS-s70 frozen mask, diag_step=${DIAG_STEP} k=${K} mc_nsamples=${MC_NSAMPLES} mask_interval=${MASK_INTERVAL} n_groups=${N_GROUPS} ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID  MODEL=$ALPS_MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON main.py \
    --model="$STUDENT_MODEL" \
    --gmp_teacher_model="$DENSE_MODEL" \
    --dataset=mixed_cot \
    --data_path="$DATA_PATH" \
    --sparsity_ratio=${SPARSITY} \
    --sparsity_type=unstructured \
    --do_gmp=true \
    --gmp_fixed_mask=true \
    --gmp_tr_enabled=false \
    --gmp_mask_interval=${MASK_INTERVAL} \
    --steps=${STEPS} \
    --gmp_batch_size=1 \
    --gmp_grad_accum=8 \
    --lr=5e-5 \
    --lr_scheduler=cosine \
    --lr_warmup_steps=256 \
    --gmp_warmup_ratio=0.05 \
    --seqlen=${SEQLEN} \
    --gmp_gradient_checkpointing=true \
    --gmp_max_prompt_len=512 \
    --gmp_kd_only=${KD_ONLY} \
    --gmp_ntp_lambda=${NTP_LAMBDA} \
    --gmp_kd_lambda=${KD_LAMBDA} \
    --gmp_onpolicy_kd_lambda=${OPKD_LAMBDA} \
    --gmp_onpolicy_max_new_tokens=512 \
    --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=${VLLM_GPU_MEM} \
    --gmp_prompt_path="$OPD_PROMPT_PATH" \
    --gmp_pgd=${PGD_ENABLED} \
    --gmp_pgd_interval=8 \
    --gmp_pgd_skip_growth_step=true \
    --gmp_kl_chunk_size=2048 \
    $([ "$PGD_ENABLED" = "true" ] && echo "--gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4") \
    ${DIAG_FLAGS} \
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \
    --save_model=false \
    --push_to_hub=false \
    --eval_math500=false \
    --eval_full_bench=false \
    --eval_zero_shot=false \
    --wandb=true \
    --wandb_project=reasoning_qwen3_1.7b_nostrip8192 \
    --run_name_suffix="saliency_${MODE}_${STUDENT}_step${DIAG_STEP}_k${K}_mc${MC_NSAMPLES}_emaN${EMA_NSAMPLES}_$([ "${NTP_LAMBDA}" = "0" ] && echo kdopdonly_)" \
    --seed=42

echo "##### END #####"
