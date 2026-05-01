#!/bin/bash
JOBID=$1
if [ -z "$JOBID" ]; then
    echo "Usage: $0 <JOBID>"
    exit 1
fi
NODE=$(squeue -j $JOBID -h -o "%N" 2>/dev/null)
if [ -z "$NODE" ]; then
    echo "Job $JOBID not found or not running"
    exit 1
fi
LOGFILE=$(ssh $NODE "ls /local-data/user-data/doyoonkim/job_${JOBID}/slurm/*.out 2>/dev/null | head -1")
if [ -z "$LOGFILE" ]; then
    echo "Log file not found for job $JOBID on $NODE"
    exit 1
fi
echo "=== Job $JOBID on $NODE: $LOGFILE ==="
ssh $NODE "tail -f $LOGFILE"
