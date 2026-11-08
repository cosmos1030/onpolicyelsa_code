#!/bin/bash
# Per-node launcher for the multi-node SCOUT 32B layout (one srun task per node):
#   node 0:  global ranks 0..2 on GPU0-2, GPU3 left free for rank 0's vLLM rollout server
#   node k>0: global ranks 3+4(k-1) .. 3+4(k-1)+3 on GPU0-3   (world size 4*nodes-1)
# torchrun can't express the uneven 3+4+... split with rank 0 pinned to node 0, so ranks are spawned here
# with the env:// variables torch.distributed reads (RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT).
U=/scratch/t512a04/users/doyoon
E=/tmp/scout_env
fusermount -uz $E 2>/dev/null  # stale mount left by an earlier job on this node
mkdir -p $E && squashfuse $U/scout_env.sqfs $E || { echo "[node $SLURM_NODEID] squashfuse mount failed"; exit 1; }
# STAGE_ROOT: /tmp is a separate 239G tmpfs; /dev/shm (239G) also holds the torch file_system segments of every vLLM weight sync (~64GB each)
SHM=${STAGE_ROOT:-/dev/shm/scout_stage}
trap "fusermount -uz $E; rm -rf $SHM; find /dev/shm -maxdepth 1 -user $(id -u) -name 'torch_*' -delete 2>/dev/null" EXIT
source $U/jobs/env_common.sh
# STAGE_DIRS="<model dir> ...": copy each checkpoint into node RAM once and point the args at the
# copy -- otherwise every rank (student + teacher load) and the vLLM server re-read 62GB from Lustre
args=("$@")
for d in ${STAGE_DIRS:-}; do
  dst=$SHM/$(basename $d)
  mkdir -p $dst && ls $d/* | xargs -P 8 -I{} cp {} $dst/ || { echo "[node $SLURM_NODEID] staging $d failed"; exit 1; }
  args=("${args[@]//$d/$dst}")
  echo "[node $SLURM_NODEID] staged $d -> $dst ($(du -sh $dst | cut -f1))"
done
if [ "$SLURM_NODEID" = 0 ]; then NLOCAL=3; BASE=0; else NLOCAL=4; BASE=$((3 + 4*(SLURM_NODEID-1))); fi
export WORLD_SIZE=$((4*SLURM_JOB_NUM_NODES - 1)) LOCAL_WORLD_SIZE=$NLOCAL OMP_NUM_THREADS=8
cd $U/code/elsa
pids=()
for ((i=0; i<NLOCAL; i++)); do
  RANK=$((BASE+i)) LOCAL_RANK=$i python main.py "${args[@]}" > $U/logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_rank$((BASE+i)).log 2>&1 &
  pids+=($!)
done
# first rank to fail takes the node down: the survivors would otherwise sit in NCCL collectives until
# the walltime, and a nonzero exit here lets srun --kill-on-bad-exit stop the other node too
rc=0
for ((n=0; n<NLOCAL; n++)); do
  wait -n || { rc=1; echo "[node $SLURM_NODEID] a rank failed -- killing the rest"; kill "${pids[@]}" 2>/dev/null; pkill -f vllm_server_standalone 2>/dev/null; break; }
done
wait
echo "[node $SLURM_NODEID] ranks done rc=$rc"
exit $rc
