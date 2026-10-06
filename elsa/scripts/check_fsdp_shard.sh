#!/bin/bash
#SBATCH --job-name=check_fsdp_shard
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/check_fsdp_shard_%j.out
exec 2>&1

TORCHRUN=/home1/doyoonkim/miniconda3/envs/rac/bin/torchrun
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
$TORCHRUN --nproc_per_node=2 --master_port=${MASTER_PORT} \
  /home1/doyoonkim/projects/elsa/scripts/check_fsdp_shard_shape.py
echo "DONE"
