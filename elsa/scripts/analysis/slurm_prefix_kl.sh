#!/bin/bash
#SBATCH --job-name=prefix_kl
#SBATCH --partition=A100-80GB
#SBATCH --qos=hpgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3,n42,n46,n51,n54,n60,n77,n80,n84,n87,n91
#SBATCH --output=/local-data/user-data/%u/prefix_kl_%j.out
exec 2>&1
# dense 와 student 를 한 GPU 에 같이 올린다 (4B bf16 두 개 = 약 16GB).
# 사용: sbatch slurm_prefix_kl.sh <student_path> <student_gen_dir> <out.csv>
OUT_NFS=/home1/doyoonkim/projects/elsa/logs/analysis; mkdir -p "$OUT_NFS"
trap 'cp /local-data/user-data/$USER/prefix_kl_${SLURM_JOB_ID}.out "$OUT_NFS/" 2>/dev/null || true' EXIT
STUDENT=${1:?student 경로}
STUDENT_GEN=${2:?student generation 디렉터리}
OUT=${3:-$OUT_NFS/prefix_kl_${SLURM_JOB_ID}.csv}
DENSE=/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
DENSE_GEN=/home1/doyoonkim/projects/elsa/logs/eval_details/s3_4b_dense_mathdet_JOBID
export HF_HOME=/home1/doyoonkim/.cache/huggingface
export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")
export TOKENIZERS_PARALLELISM=false
echo "NODE=$(hostname) JOB=$SLURM_JOB_ID"
/home1/doyoonkim/miniconda3/envs/rac/bin/python \
  /home1/doyoonkim/projects/elsa/scripts/analysis/prefix_kl.py \
  --dense "$DENSE" --student "$STUDENT" \
  --dense_gen "${DENSE_GEN_OVERRIDE:-$DENSE_GEN}" --student_gen "$STUDENT_GEN" \
  --n "${N:-100}" --max_new "${MAX_NEW:-2048}" --out "$OUT"
echo "##### END #####"
