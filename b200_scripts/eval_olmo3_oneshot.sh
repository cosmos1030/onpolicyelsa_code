#!/bin/bash
# Eval-only for an OLMo-3-7B-Think checkpoint, long profile, seed 42, reasoning benchmarks only
# (same as the in-training eval of the SCOUT / ALPS+retrain runs).
# Usage: CUDA_VISIBLE_DEVICES=<gpu> [BENCH=lcb,gsm8k] [MODEL=<path|repo>] [RUN=<wandb run>] [METHOD=alps|gmp] bash eval_olmo3_oneshot.sh <SPARSITY_PCT>
#   default MODEL: cosmos1030/olmo3-7b-think-alps-oneshot-s<P>; BENCH restricts to a subset (resume after a container loss)
set -u
R=/NHNHOME/log-postech/doyoonkim; P=${1:?sparsity pct}
MODEL=${MODEL:-cosmos1030/olmo3-7b-think-alps-oneshot-s$P}; RUN=${RUN:-olmo3_7b_alps_oneshot_s$P}; METHOD=${METHOD:-alps}
source $R/miniconda3/etc/profile.d/conda.sh; conda activate rac_olmo3
export HF_TOKEN=$(cat $R/secrets/hf_token) WANDB_API_KEY=$(cat $R/secrets/wandb_api_key)
export HF_HOME=$R/.cache/huggingface VLLM_CACHE_ROOT=$R/.cache/vllm TRITON_CACHE_DIR=$R/.cache/triton TORCHINDUCTOR_CACHE_DIR=$R/.cache/torchinductor
export NLTK_DATA=/NHNHOME/log-postech/doyoonkim/nltk_data
export TMPDIR=/tmp TOKENIZERS_PARALLELISM=false VLLM_USE_V1=0 VLLM_HOST_IP=127.0.0.1
cd $R/onpolicyelsa_code/elsa
python scripts/eval_full.py --model_path "$MODEL" \
  --wandb_project reasoning_olmo3_7b_nostrip8192 --run_name "$RUN" \
  --method "$METHOD" --sparsity 0.$((P/10)) --profile long --seeds 42 --tp_size 1 --gpu_util 0.90 \
  ${BENCH:+--benchmarks $BENCH} --skip_ppl --skip_zeroshot --out_base $R/logs/eval_olmo3_oneshot/${RUN}${BENCH:+_$(echo $BENCH | tr , _)}
echo "=== EXIT: $? ==="
