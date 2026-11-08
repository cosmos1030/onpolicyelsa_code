#!/bin/bash
#SBATCH --job-name=alpssft32b-s70
#SBATCH --partition=natStr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch-fsdp"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Qwen3-32B ALPS s70 -> NTP/KD/OPD training (fixed ALPS mask), the baseline for the 32B SCOUT run
# (run_32b_s70.sbatch): paper 8B ALPS+retrain recipe (b200_scripts/alps_sft_ntpkd_opkd_qwen3_8b_fsdp2gpu.sh:
# lr 1e-4 cosine, rollout interval 32, rollout 512 tokens, NTP/KD/OPD 0.33 each) with the SAME budget and
# memory options as the SCOUT run: 512 steps (warmup 64), global batch 7 (7 ranks x accum 1), chunked
# full-vocab NTP+KD loss, Adam on CPU outside optimizer.step(). Student = ALPS one-shot model, teacher =
# dense Qwen3-32B. Sharded checkpoints every 128 steps (resume: RESUME_FROM=<stepNNNNNN.pt>, same layout).
U=/scratch/t512a04/users/doyoon
CK=$U/ckpt/alpssft32b_s70
mkdir -p $CK $U/models_out
export GMP_MEM_DEBUG=1 GMP_PHASE_TIMING=1  # per-rank memory at the first steps + per-phase seconds (cost breakdown)
export STAGE_ROOT=/tmp/scout_stage
export STAGE_DIRS="$U/models_out/qwen3_32b_alps_s70pct $U/models/Qwen3-32B"
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=29514
echo "nodes: $SLURM_JOB_NODELIST master=$MASTER_ADDR resume=${RESUME_FROM:-none}"
[ -z "${RESUME_FROM:-}" ] && rm -f $CK/.owner.json
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --kill-on-bad-exit=1 bash $U/jobs/node_launch.sh \
    --model=$U/models_out/qwen3_32b_alps_s70pct --gmp_teacher_model=$U/models/Qwen3-32B --dataset=mixed_cot \
    --data_path=$U/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --gmp_prompt_path=$U/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --sparsity_ratio=0.7 --sparsity_type=unstructured --do_gmp=true --gmp_fixed_mask=true \
    --gmp_use_fsdp=true --gmp_teacher_fsdp=true \
    --gmp_chunked_ntpkd_loss=true --gmp_offload_optimizer_fwd=true --gmp_adam_foreach=false --gmp_skip_initial_vllm_sync=true \
    --steps=512 --gmp_batch_size=1 --gmp_grad_accum=1 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=64 \
    --gmp_mask_interval=32 --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_kl_chunk_size=256 --gmp_max_prompt_len=512 \
    --gmp_kd_only=false --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=512 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.85 --gmp_opkd_vllm_gpu_index=3 \
    --gmp_ckpt_sharded=true --gmp_ckpt_every_steps=128 --gmp_ckpt_dir=$CK --gmp_resume_from="${RESUME_FROM:-}" \
    --gmp_save_path=$U/models_out --save_model=true \
    --push_to_hub=true --hub_model_id=cosmos1030/qwen3-32b-alps-ntpkdopd-s70-roll512-512steps \
    --eval_math500=false --eval_full_bench=false --eval_zero_shot=false \
    --wandb=true --wandb_project=reasoning_qwen3_32b --seed=42 --run_name_suffix=hangang_alpssft32b_s70_512steps
echo "=== main.py EXIT: $? ==="
