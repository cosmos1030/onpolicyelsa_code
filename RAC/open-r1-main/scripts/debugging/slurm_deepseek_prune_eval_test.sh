#!/bin/bash
#SBATCH --job-name=deepseek_prune_TEST
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=/home1/doyoonkim/projects/RAC/open-r1-main/logs/%j_deepseek_prune_test.out
#SBATCH -t 0-02:00:00
#SBATCH --exclude=n58,n80

# Debug script: DeepSeek-1.5B pruning + lighteval math500 eval
# Tests that vLLM OOM fix (gpu_memory_utilization=0.75) works for the larger model.
# Uses small calib tokens for speed.

mkdir -p /home1/doyoonkim/projects/RAC/open-r1-main/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate rac

cd /home1/doyoonkim/projects/RAC/open-r1-main

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copy the HF-cached dataset to /tmp (node-local tmpfs) so that mmap of
# 500 MB arrow shards succeeds on compute nodes whose shared filesystem
# enforces strict virtual-address-space limits.
HF_DS_SRC="/home1/doyoonkim/.cache/huggingface/datasets/open-r1___open_r1-math-220k"
HF_DS_DST="/tmp/hf_datasets_cache_doyoon/open-r1___open_r1-math-220k"
if [ ! -d "${HF_DS_DST}" ]; then
    echo "Copying dataset to local /tmp ..."
    mkdir -p /tmp/hf_datasets_cache_doyoon
    cp -r "${HF_DS_SRC}" /tmp/hf_datasets_cache_doyoon/
    echo "Dataset copy done."
fi
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_doyoon

echo "=== DeepSeek-1.5B prune + lighteval test ==="
date

/home1/doyoonkim/miniconda3/envs/rac/bin/python src/open_r1/grpo.py \
  --config recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/config_pruning.yaml \
  --dataset_name=open-r1/OpenR1-Math-220k \
  --dataset_prompt_column=problem \
  --do_train=False \
  --prune=True \
  --pruning_method=SparseGPT \
  --prune_sparsity=0.5 \
  --prune_calib_tokens=50000 \
  --math_eval_max_new_tokens=4096 \
  --push_to_hub=False \
  --score_completions=False \
  --report_to=none

echo "=== Test done ==="
date
