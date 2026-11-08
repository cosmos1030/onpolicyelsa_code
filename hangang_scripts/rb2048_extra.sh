#!/bin/bash
# Run part of the roll2048 set on idle nodes of an already running job (4 runs per node), e.g. the
# eval job's finished nodes: rb2048_extra.sh <jobid> <node0>,<node1> <s80|scout|all>
# The steps end with that job; resubmit via rb2048.sbatch to continue from the checkpoints.
U=/scratch/t512a04/users/doyoon
J=$1; NODES=$2
export RB_RUNS=$(bash $U/jobs/rb2048_runs.sh ${3:?s80|scout|all})
k=0
for n in ${NODES//,/ }; do
  RB_NODE=$k srun --jobid=$J --overlap -N1 -n1 -c 288 -w $n --export=ALL bash $U/jobs/rb2048_node.sh </dev/null \
      > $U/logs/rb2048_extra_${J}_$n.out 2>&1 &
  k=$((k+1))
done
wait
