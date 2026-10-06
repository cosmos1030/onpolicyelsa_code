#!/bin/bash
#SBATCH --job-name=pol_clean_resume
#SBATCH --partition=RTX6000ADA,A6000
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3,n42,n46,n51,n52,n54,n55,n58,n60,n76,n77,n80,n91
#SBATCH --output=/local-data/user-data/%u/job_%j/slurm/%x_%j.out

# Resume guard for the clean-pool rebuild. policy_divergence_tsne.py caches each
# model's rollouts in samples_<key>.json and skips re-sampling when the file is
# there, so a run killed at the walltime loses only the encode/analyse tail --
# but nothing restarts it. This is queued with --dependency=afterany so it runs
# whatever happens, and exits immediately if the pool is already complete.
OUT=/home1/doyoonkim/projects/elsa/logs/policy_divergence/n30_k64_clean
if [ -f "$OUT/pooled.npz" ]; then
    echo "pooled.npz already present -- 939992 finished, nothing to resume."
    ls -la "$OUT/pooled.npz"
    exit 0
fi
echo "pooled.npz missing -- resuming from cached rollouts"
ls "$OUT"/samples_*.json 2>/dev/null | wc -l | xargs echo "cached rollout files:"
exec bash /home1/doyoonkim/projects/elsa/scripts/slurm_policy_divergence.sh 30 64 2048 clean core
