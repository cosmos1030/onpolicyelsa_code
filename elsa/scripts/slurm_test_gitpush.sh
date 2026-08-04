#!/bin/bash
#SBATCH --job-name=test_gitpush
#SBATCH --partition=RTX3090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/home1/doyoonkim/projects/elsa/logs/test_gitpush_%j.out
exec 2>&1

echo "=== git push test ==="
echo "NODE=$(hostname)  JOB=$SLURM_JOB_ID"

PYTHON=/home1/doyoonkim/miniconda3/envs/rac/bin/python
_GIT_ROOT="/home1/doyoonkim/projects"

# Step 1: rundb status 확인
echo "--- rundb status ---"
cd "$_GIT_ROOT/elsa/scripts"
$PYTHON rundb/cli.py status 2>&1 | head -10

# Step 2: 테스트 파일 생성 → git push
echo "--- git push test (create) ---"
echo "gitpush_test_${SLURM_JOB_ID}" > "$_GIT_ROOT/elsa/scripts/.gitpush_test"
git -C "$_GIT_ROOT" add elsa/scripts/.gitpush_test
git -C "$_GIT_ROOT" commit -m "chore: test git push from compute node (job ${SLURM_JOB_ID})" \
    && git -C "$_GIT_ROOT" push 2>&1 \
    && echo "PUSH OK" \
    || echo "PUSH FAILED"

# Step 3: 테스트 파일 삭제 → 클린업
echo "--- git push test (cleanup) ---"
rm -f "$_GIT_ROOT/elsa/scripts/.gitpush_test"
git -C "$_GIT_ROOT" add elsa/scripts/.gitpush_test
git -C "$_GIT_ROOT" commit -m "chore: cleanup test file (job ${SLURM_JOB_ID})" \
    && git -C "$_GIT_ROOT" push 2>&1 \
    && echo "CLEANUP PUSH OK" \
    || echo "CLEANUP PUSH FAILED"

echo "##### END #####"
