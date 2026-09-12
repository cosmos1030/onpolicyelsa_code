#!/bin/bash
#SBATCH --job-name=prompt_cache
#SBATCH --partition=cpu-max16
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/policy_divergence/prompt_cache_%j.out
exec 2>&1

# Materialise the six prompts + their dataset CoTs into prompts.json, so the GPU
# jobs never spend 30 minutes pulling OpenThoughts3-1.2M to get six rows.
# Deterministic given (source, seed, n_prompts), so it reproduces exactly the
# prompts the cached rollouts were generated from.

OUTDIR=${1:-/home1/doyoonkim/projects/elsa/logs/policy_divergence/n6_k96_base}
N=${2:-6}
SEED=${3:-42}
SOURCE=${4:-ot3}

export TMPDIR=/tmp
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_OFFLINE=0

cd /home1/doyoonkim/projects/elsa
/home1/doyoonkim/miniconda3/envs/rac/bin/python - "$OUTDIR" "$N" "$SEED" "$SOURCE" <<'PY'
import sys, os, json, glob
sys.path.insert(0, "/home1/doyoonkim/projects/elsa/scripts")
from transformers import AutoTokenizer
from onpolicy_mismatch_diag import build_prompts

outdir, n, seed, source = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
D = glob.glob("/home1/doyoonkim/.cache/huggingface/hub/"
              "models--Qwen--Qwen3-4B/snapshots/*")[0]
tok = AutoTokenizer.from_pretrained(D, trust_remote_code=True)
prompts, solutions, ids = build_prompts(tok, n, seed, True, source)
print(f"prompts={len(prompts)}  CoT chars={[len(s) for s in solutions]}", flush=True)
p = os.path.join(outdir, "prompts.json")
tmp = p + ".tmp"
json.dump({"prompts": prompts, "solutions": solutions, "ids": ids,
           "source": source, "seed": seed, "n_prompts": n}, open(tmp, "w"))
os.replace(tmp, p)
print(f"wrote {p}", flush=True)
PY
echo "=== EXIT: $? ==="
