#!/bin/bash
# Sets up the two conda envs needed to reproduce the IPO (DPO/IPO on
# UltraFeedback) and GRPO (overthinking-reward + noLen-ablation) experiments,
# plus the reasoning-bench eval (math500/lcb/gpqa/ifeval/gsm8k) used to score
# their checkpoints.
#
# Two SEPARATE envs are required, not one:
#   - `rac`         : reasoning eval only (torch 2.7.1, vllm 0.10.0)
#   - `rac_vllm084` : IPO/GRPO training (torch 2.6.0, vllm 0.8.4)
# GRPO's vLLM generation server and its training client must share the same
# vllm/NCCL build or the weight-sync communicator handshake fails ("NCCL
# error: unhandled system error") -- vllm>=0.10 also hits a known upstream
# trl/vLLM server-mode deadlock in update_named_param (huggingface/trl#3608)
# that's fixed by pinning vllm==0.8.4 for training. `rac` stays on vllm 0.10.0
# because that's what the shared eval/lighteval path needs.
#
# Usage: bash setup_ipo_grpo_env.sh
# (run from anywhere; paths below are relative to this script's directory)

set -e
cd "$(dirname "$0")/.."   # -> elsa/

echo "=== Creating env: rac (reasoning eval) ==="
conda create -y -n rac python=3.10
conda run -n rac pip install -r requirements.txt
conda run -n rac pip install flash-attn==2.8.3 --no-build-isolation

echo "=== Creating env: rac_vllm084 (IPO/GRPO training) ==="
conda create -y -n rac_vllm084 python=3.10
conda run -n rac_vllm084 pip install -r requirements-vllm084.txt
conda run -n rac_vllm084 pip install flash-attn==2.8.3 --no-build-isolation

cat <<'EOF'

=== Done ===
Two envs created: `rac` (eval) and `rac_vllm084` (IPO/GRPO training).

Notes:
- Both scripts under scripts/slurm_ipo_ultrafeedback_s70_fullft.sh and
  scripts/slurm_grpo_overthinking_s70.sh already `conda activate
  rac_vllm084` for training and switch to `rac` for the eval step -- no
  manual env-switching needed if you keep these two names.
- RAC/open-r1-main's `open_r1` package is NOT pip-installed; both launcher
  scripts add it to PYTHONPATH directly
  (RAC/open-r1-main/src and RAC/open-r1-main/src/open_r1). If you invoke
  src/open_r1/{dpo,grpo}.py yourself outside those launchers, set the same
  PYTHONPATH first.
- See README.md's "Environment variables / secrets" section for the
  HF_TOKEN/WANDB_API_KEY/etc. env vars both launcher scripts expect.
EOF
