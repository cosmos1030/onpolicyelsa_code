#!/bin/bash
#SBATCH --job-name=openbook_cmp
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3,n60,n80
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

exec 2>&1

ENV_FILE="/run/slurm/job_env_${SLURM_JOB_ID}"
[ -f "$ENV_FILE" ] && source "$ENV_FILE"

if [ -z "${LOCAL_JOB_BASE:-}" ]; then
    LOCAL_JOB_BASE="/local-data/user-data/${USER}/job_${SLURM_JOB_ID}"
fi
mkdir -p "$LOCAL_JOB_BASE/slurm"

source ~/miniconda3/etc/profile.d/conda.sh
cd /home1/doyoonkim/projects/elsa

export TRITON_CACHE_DIR=/tmp/triton_cache_doyoon
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT_DIR="/home1/doyoonkim/projects/elsa/eval_outputs/openbookqa_compare"
mkdir -p "$OUT_DIR"

DENSE="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
GMP="cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260705_130752"

echo "Node: $(hostname)"
echo "=== Dense model ==="

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'PYEOF'
import torch, json, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

OUT_DIR = "/home1/doyoonkim/projects/elsa/eval_outputs/openbookqa_compare"

for tag, model_id in [
    ("dense", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("gmp_s50_ntp_kd", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260705_130752"),
]:
    print(f"\n{'='*60}\nEvaluating: {tag} ({model_id})\n{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=["openbookqa"],
        num_fewshot=0,
        log_samples=True,
        batch_size="auto",
        random_seed=42,
        numpy_random_seed=42,
        torch_random_seed=42,
        cache_requests=False,
        check_integrity=False,
    )

    # 정확도 출력
    acc = results["results"]["openbookqa"]["acc,none"]
    acc_norm = results["results"]["openbookqa"]["acc_norm,none"]
    print(f"  acc={acc:.4f}  acc_norm={acc_norm:.4f}")

    # 개별 샘플 저장
    samples = results.get("samples", {}).get("openbookqa", [])
    out_path = os.path.join(OUT_DIR, f"{tag}_openbookqa_samples.jsonl")
    with open(out_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  saved {len(samples)} samples → {out_path}")

    # 요약 저장
    summary = {"model": model_id, "tag": tag, "acc": acc, "acc_norm": acc_norm}
    with open(os.path.join(OUT_DIR, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    del model, lm
    torch.cuda.empty_cache()
    import gc; gc.collect()

print("\nDone.")
PYEOF

echo "##### END #####"
