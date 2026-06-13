#!/bin/bash
# GMP NTP sweep for DeepSeek-R1-Distill-Qwen-7B (s50 + s70)
# 4x A100-80GB per job, FSDP sharding
# global_batch = 4 GPUs × batch_size=1 × grad_accum=2 = 8
# 3 LR values × 2 sparsities = 6 jobs total

set -e
cd "$(dirname "$0")/.."
SUBMIT_DIR=$(pwd)

MODEL="/home1/doyoonkim/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
DATA_PATH="/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl"

LR_VALUES=(5e-5 1e-4 2e-4)
SPARSITIES=(0.5 0.7)

mkdir -p tmp_scripts out

for SP in "${SPARSITIES[@]}"; do
    SP_TAG=$(echo $SP | tr -d '.')   # 05, 07
    for LR in "${LR_VALUES[@]}"; do
        LR_TAG=$(echo $LR | tr -d '.')  # 5e5, 1e4, 2e4
        JOB_NAME="gmp7b_s${SP_TAG}_lr${LR_TAG}"
        SCRIPT="tmp_scripts/${JOB_NAME}.sh"

        cat > $SCRIPT << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --exclude=n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/${JOB_NAME}_%j.out
exec 2>&1
mkdir -p /local-data/user-data/\$USER/job_\$SLURM_JOB_ID/slurm
mkdir -p /local-data/user-data/\$USER/job_\$SLURM_JOB_ID/wandb

export WANDB_DIR=/local-data/user-data/\$USER/job_\$SLURM_JOB_ID/wandb
export WANDB_API_KEY=\$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1)
export HF_TOKEN=\$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

echo "=== Job ${JOB_NAME} | sp=${SP} lr=${LR} ==="
echo "SLURM_JOB_ID=\$SLURM_JOB_ID"
echo "NODE: \$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd ${SUBMIT_DIR}
torchrun \\
    --nproc_per_node=4 \\
    --master_port=29500 \\
    main.py \\
    --model=${MODEL} \\
    --dataset=math_cot \\
    --data_path=${DATA_PATH} \\
    --sparsity_ratio=${SP} \\
    --do_gmp=true \\
    --gmp_use_fsdp=true \\
    --gmp_steps=1024 \\
    --gmp_batch_size=1 \\
    --gmp_grad_accum=2 \\
    --gmp_lr=${LR} \\
    --gmp_warmup_ratio=0.05 \\
    --gmp_mask_interval=32 \\
    --gmp_fisher_beta=0.999 \\
    --gmp_kd_lambda=0.0 \\
    --gmp_save_path=/home1/doyoonkim/projects/elsa/models \\
    --gmp_max_prompt_len=512 \\
    --gmp_max_seq_len=2048 \\
    --save_model=true \\
    --eval_math500=true \\
    --math500_max_new_tokens=8192 \\
    --math500_max_samples=500 \\
    --eval_zero_shot=false \\
    --wandb=true \\
    --wandb_project=gmp_qwen3_7b \\
    --push_to_hub=false \\
    --seed=42
EOF

        JID=$(sbatch --parsable $SCRIPT)
        echo "Submitted: ${JOB_NAME} → job $JID (sp=${SP}, lr=${LR})"
    done
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u $USER"
