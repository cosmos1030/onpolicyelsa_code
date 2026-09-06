#!/bin/bash
#SBATCH --job-name=grpo_overthinking_s70
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/grpo_overthinking_s70_%j.out
exec 2>&1

# Usage: sbatch slurm_grpo_overthinking_s70.sh [MAX_STEPS] [OUTPUT_DIR] [CONFIG_YAML]
# Small-scale verification first: sbatch slurm_grpo_overthinking_s70.sh 10 /home1/doyoonkim/projects/elsa/models/grpo_trgmp_s70_smoketest

MAX_STEPS=${1:-200}
OUTPUT_DIR=${2:-/home1/doyoonkim/projects/elsa/models/grpo_trgmp_s70_overthinking}
CONFIG_YAML=${3:-recipes/Qwen3-4B/grpo/config_overthinking_s70.yaml}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm
NFS_LOG_COPY="${OUTPUT_DIR}_slurmlog_${SLURM_JOB_ID}.out"
copy_log_to_nfs() {
  mkdir -p "$(dirname "$NFS_LOG_COPY")"
  cp "/local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm/grpo_overthinking_s70_${SLURM_JOB_ID}.out" "$NFS_LOG_COPY" 2>/dev/null
}
trap copy_log_to_nfs EXIT

source ~/miniconda3/etc/profile.d/conda.sh
# Both the vLLM server AND the training/client side use rac_vllm084 (vllm==0.8.4)
# so their NCCL/torch versions match -- mixing rac (vllm 0.10.0) for training
# with rac_vllm084 (vllm 0.8.4) for the server caused "NCCL error: unhandled
# system error" during the weight-sync communicator handshake, since NCCL
# requires matching versions across all processes in a communicator.
# lighteval evals are unaffected -- they still run from the `rac` env.
conda activate rac_vllm084

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd /home1/doyoonkim/projects/RAC/open-r1-main
export PYTHONPATH=/home1/doyoonkim/projects/RAC/open-r1-main/src:/home1/doyoonkim/projects/RAC/open-r1-main/src/open_r1

echo "=== GRPO overthinking fix: TR-GMP-s70, MAX_STEPS=$MAX_STEPS, OUTPUT_DIR=$OUTPUT_DIR ==="
echo "SLURM_JOB_ID=$SLURM_JOB_ID  NODE=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODEL_NAME=$(/home1/doyoonkim/miniconda3/envs/rac_vllm084/bin/python -c "
import yaml
print(yaml.safe_load(open('$CONFIG_YAML'))['model_name_or_path'])
")

# GPU 1 dedicated to the vLLM generation server; GPU 0 for training.
# (Cut down from 4 GPUs to 2: the A100-80GB partition is fragmented cluster-
# wide right now -- every node has at most 1-2 free GPUs -- so a single-node
# 4-GPU request was stuck behind Priority indefinitely. 2 GPUs schedules
# immediately on nodes with just 2 free slots.)
# IMPORTANT: launch via the VENDORED trl fork's own vllm_serve module, not the
# separately pip-installed `trl` CLI -- they're both nominally "0.21.0" but
# have incompatible /init_communicator/ request schemas (the real pip trl's
# server expects a "client_device_uuid" field the vendored client never sends),
# so client and server must come from the exact same codebase.
echo "Starting vLLM server (GPU 1) for $MODEL_NAME ..."
# Runs in an ISOLATED env (rac_vllm084, cloned from rac + vllm==0.8.4) --
# vllm>=0.10 hits a known upstream trl/vLLM server-mode deadlock in
# update_named_param (huggingface/trl#3608), community-confirmed fixed by
# downgrading to vllm==0.8.4. The shared `rac` env stays on vllm==0.10.0
# (needed for lighteval compat + other running jobs); only this standalone
# server subprocess uses the older vllm.
CUDA_VISIBLE_DEVICES=1 /home1/doyoonkim/miniconda3/envs/rac_vllm084/bin/python -m open_r1_trl.trl.scripts.vllm_serve \
  --model "$MODEL_NAME" \
  --tensor_parallel_size 1 \
  --gpu_memory_utilization 0.85 \
  --max_model_len 10240 \
  --enforce_eager False \
  --host 127.0.0.1 \
  --port 8000 > "${OUTPUT_DIR}_vllm_server_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM server to become healthy..."
# Cold start (weight load off NFS + torch.compile) measured at ~14 min once;
# give it a generous 30 min before giving up.
for i in $(seq 1 180); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health/ 2>/dev/null | grep -q "200"; then
    echo "vLLM server healthy after ${i}0s"
    break
  fi
  sleep 10
done

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file recipes/plain_1gpu.yaml src/open_r1/grpo.py \
  --config "$CONFIG_YAML" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS"
GRPO_EXIT=$?

echo "Shutting down vLLM server (pid $VLLM_PID)"
kill $VLLM_PID 2>/dev/null

echo "=== DONE (grpo exit $GRPO_EXIT) ==="
exit $GRPO_EXIT
