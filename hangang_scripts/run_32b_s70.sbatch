#!/bin/bash
#SBATCH --job-name=scout32b-s70
#SBATCH --partition=natStr
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=efficientai;appl=pytorch-fsdp"
#SBATCH --output=/scratch/t512a04/users/doyoon/logs/%x_%j.out
# Qwen3-32B SCOUT s70, rebuttal scalability run (SWVr-2): paper Qwen3-8B s70 recipe (klb 0.02 per user -- paper 8B used 0.03; mask/rollout
# interval 32, rollout 512 tokens, lr 1e-4, NTP/KD/OPD 0.33 each) but 512 steps instead of 2048
# (lr warmup 256 -> 64). 2 GH200 nodes: 7 FSDP ranks + vLLM on node0 GPU3. Checkpoints every 128 steps;
# resubmit with RESUME_FROM=<stepNNNNNN.pt> (same 7-rank layout) if the 24h walltime cuts it.
U=/scratch/t512a04/users/doyoon
CK=$U/ckpt/scout32b_s70
mkdir -p $CK $U/models_out
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) MASTER_PORT=29513
echo "nodes: $SLURM_JOB_NODELIST master=$MASTER_ADDR resume=${RESUME_FROM:-none}"
srun --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_ON_NODE --kill-on-bad-exit=1 bash $U/jobs/node_launch.sh \
    --model=$U/models/Qwen3-32B --dataset=mixed_cot \
    --data_path=$U/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    --gmp_prompt_path=$U/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl \
    --sparsity_ratio=0.7 --sparsity_type=unstructured --do_gmp=true --gmp_use_fsdp=true --gmp_teacher_fsdp=true \
    --gmp_chunked_ntpkd_loss=true --gmp_offload_optimizer_fwd=true \
    --steps=512 --gmp_batch_size=1 --gmp_grad_accum=4 --lr=1e-4 --lr_scheduler=cosine --lr_warmup_steps=64 \
    --gmp_warmup_ratio=0.05 --gmp_mask_interval=32 --gmp_fisher_beta=0.999 --gmp_saliency=fisher --gmp_pruning_scope=global \
    --seqlen=8192 --gmp_gradient_checkpointing=true --gmp_kl_chunk_size=256 --gmp_max_prompt_len=512 \
    --gmp_kd_only=false --kd_nsamples=0 --gmp_ntp_lambda=0.33 --gmp_kd_lambda=0.33 --gmp_onpolicy_kd_lambda=0.33 \
    --gmp_onpolicy_kd_interval=32 --gmp_onpolicy_max_new_tokens=512 --gmp_opkd_prev_mask_teacher=false \
    --gmp_opkd_vllm_gpu_mem=0.85 --gmp_opkd_vllm_gpu_index=3 \
    --gmp_tr_enabled=false --gmp_pruning_end_ratio=0.0 --gmp_pgd=true --gmp_pgd_grow_to_target=true \
    --gmp_pgd_kl_budget=0.02 --gmp_pgd_kl_calib_size=4 --gmp_pgd_interval=8 \
    --gmp_ckpt_every_steps=128 --gmp_ckpt_dir=$CK --gmp_resume_from="${RESUME_FROM:-}" \
    --gmp_save_path=$U/models_out --save_model=true --push_to_hub=false \
    --eval_math500=false --eval_full_bench=false --eval_zero_shot=false \
    --wandb=false --seed=42 --run_name_suffix=hangang_32b_s70_512steps
echo "=== main.py EXIT: $? ==="
