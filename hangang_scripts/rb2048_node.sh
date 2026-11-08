#!/bin/bash
# Per-node body of rb2048.sbatch: lines [4*k, 4*k+3] of RB_RUNS ("<name> <sparsity> <extra flags...>"),
# run i on GPU i; k = RB_NODE if set (steps added to another job's nodes), else SLURM_NODEID.
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "[$(hostname)] squashfuse mount failed"; exit 1; }
trap "fusermount -uz $E" EXIT
source $U/jobs/env_common.sh
ulimit -c 0
export NCCL_NET=Socket
K=${RB_NODE:-$SLURM_NODEID}
i=0
while read -r N SP EXTRA; do
  [ -n "$N" ] || continue
  ( CUDA_VISIBLE_DEVICES=$i VLLM_PORT=$((30000 + 100*i)) bash $U/jobs/rb2048_run.sh $N $SP $EXTRA \
      >> $U/logs/rb2048_$N.log 2>&1
    echo "[$(hostname) gpu$i] $N exit=$(grep -ao 'main.py EXIT: [0-9]*' $U/logs/rb2048_$N.log | tail -1)" ) &
  i=$((i+1))
done < <(echo "$RB_RUNS" | sed -n "$((4*K+1)),$((4*K+4))p")
wait
