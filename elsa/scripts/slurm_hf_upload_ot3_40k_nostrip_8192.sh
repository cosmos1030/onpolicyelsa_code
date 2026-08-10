#!/bin/bash
#SBATCH --job-name=hf_upload_ot3_40k_nostrip_8192
#SBATCH --partition=cpu-max24
#SBATCH --qos=nogpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/hf_upload_ot3_40k_nostrip_8192_%j.out
exec 2>&1

export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")

echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

if ! curl -s --connect-timeout 10 https://huggingface.co > /dev/null 2>&1; then
    echo "ERROR: No internet on $(hostname). Exiting."
    exit 1
fi

cd /home1/doyoonkim/projects/elsa

/home1/doyoonkim/miniconda3/envs/rac/bin/python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()
repo_id = "cosmos1030/ot3-fineweb-40k-qwen3-nostrip-8192"
api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
print("repo created/exists", flush=True)
api.upload_file(
    path_or_fileobj="data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl",
    path_in_repo="ot3_fineweb_40k_qwen3_nostrip_8192.jsonl",
    repo_id=repo_id,
    repo_type="dataset",
)
print("DONE uploading", flush=True)
EOF

echo "=== EXIT: $? ==="
