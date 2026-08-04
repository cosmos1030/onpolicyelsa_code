#!/bin/bash
#SBATCH --job-name=eval_ppl
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/eval_ppl_%j.out
exec 2>&1

# Usage:
#   sbatch slurm_eval_ppl.sh <MODEL_PATH> [WANDB_RUN_ID] [WANDB_PROJECT]
#
# Loads a sparse HF model from MODEL_PATH, computes wikitext2 + c4 PPL,
# logs results to wandb (resuming run if WANDB_RUN_ID given).

MODEL_PATH=${1:?"Usage: sbatch slurm_eval_ppl.sh <MODEL_PATH> [WANDB_RUN_ID] [WANDB_PROJECT]"}
WANDB_RUN_ID=${2:-""}
WANDB_PROJECT=${3:-"reasoning_qwen3_1.7b"}

mkdir -p /local-data/user-data/$USER/job_$SLURM_JOB_ID/slurm

export WANDB_DIR=/local-data/user-data/$USER/job_$SLURM_JOB_ID/wandb
export WANDB_API_KEY=$(grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2 | tail -1 | tr -d ' \n\r')
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
export TOKENIZERS_PARALLELISM=false
export HF_HOME="/home1/doyoonkim/.cache/huggingface"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python

echo "=== eval_ppl ==="
echo "MODEL_PATH=$MODEL_PATH"
echo "WANDB_RUN_ID=$WANDB_RUN_ID"
echo "WANDB_PROJECT=$WANDB_PROJECT"
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if ! curl -s --connect-timeout 10 https://api.wandb.ai/healthz > /dev/null 2>&1; then
    echo "ERROR: No internet. Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

$PYTHON - <<PYEOF
import torch, wandb, json, os, sys, math
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path    = "$MODEL_PATH"
wandb_run_id  = "$WANDB_RUN_ID"
wandb_project = "$WANDB_PROJECT"

device = torch.device("cuda:0")
print(f"Loading model from {model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
model.eval()
model.to(device)

# Check actual sparsity
total, zero = 0, 0
for name, p in model.named_parameters():
    if p.dim() >= 2 and "embed" not in name and "lm_head" not in name:
        total += p.numel()
        zero  += (p.data == 0).sum().item()
sparsity = zero / total if total > 0 else 0.0
print(f"Sparsity: {sparsity:.4f}")

# Import existing eval_ppl
sys.path.insert(0, "/home1/doyoonkim/projects/elsa")
from lib.eval import eval_ppl

class FakeFlags:
    seqlen = 2048
    eval_dataset = "wikitext2"

flags = FakeFlags()
model.seqlen = flags.seqlen

print("Computing PPL on wikitext2 + c4 ...")
try:
    ppls = eval_ppl(flags, model, tokenizer, device, data_path=None)
    print(f"PPL results: {ppls}")
except Exception as e:
    print(f"PPL eval failed: {e}")
    ppls = {}

metrics = {
    "eval/sparsity": sparsity,
    **{f"eval/ppl_{k}": v for k, v in ppls.items()}
}

out_path = os.path.join(model_path, "ppl_results.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Saved to {out_path}")
print(f"Metrics: {metrics}")

if wandb_run_id:
    run = wandb.init(project=wandb_project, id=wandb_run_id, resume="must")
else:
    run = wandb.init(project=wandb_project, name=os.path.basename(model_path) + "_ppl")
run.log(metrics)
run.finish()
print("wandb logging done")
PYEOF

EXIT_CODE=$?
echo "=== EXIT: $EXIT_CODE ==="
exit $EXIT_CODE
