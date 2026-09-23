#!/bin/bash
# Kill eval jobs that have stopped making progress, and restart the ones whose
# weights are already on local disk.
#
#   nohup bash elsa/scripts/log_cluster/eval_watchdog.sh > ~/eval_watchdog/loop.log 2>&1 &
#
# Why: twice in 24h a batch of jobs sat for 5-11 hours at "loading model
# weights" with the GPU at 0%, because huggingface_hub's NFS cache wedges on a
# lock a killed job left behind and nothing times out. Nobody noticed until a
# human looked. A running eval writes a vLLM progress bar continuously, so a
# log that has not grown in STALL_MIN minutes is not slow -- it is stuck.
set -u
LOGS=/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs
MODELS=/home/doyoonkim/models
STATE=$HOME/eval_watchdog
LAUNCH_LADDER=/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_8b_ladder.sh
LAUNCH_HANDOFF=/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_long_handoff.sh
INTERVAL=${INTERVAL:-300}
# Two thresholds. A wedged download sits at "loading model weights" forever
# with the GPU at 0%, and 20min of that is already certain. But a healthy job
# also goes quiet for long stretches while lighteval SCORES a benchmark
# (math_500's sympy extraction ran 29min silent), and a 25min blanket rule
# killed a perfectly good 1.7B job mid-scoring. So: quick kill only while the
# job is still loading weights, a patient one once it is past that.
STALL_LOAD_MIN=${STALL_LOAD_MIN:-20}
STALL_MIN=${STALL_MIN:-60}
mkdir -p "$STATE"
say () { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# arm name -> (launcher, argument). The ladder arms take their own name; the
# handoff arms are named <arm>_seed<N> in that script's case block.
resubmit () {
    local jobname=$1 arm
    case "$jobname" in
        ours_s*|alpsretrain_s[56]0|alps_s[56]0|sparsegpt_s[56]0)
            arm=$jobname
            [ -f "$MODELS/$arm/.download_complete" ] || { say "  no local copy for $arm -- not resubmitting"; return 1; }
            sbatch -J "$arm" --partition=A100,H200 "$LAUNCH_LADDER" "$arm" ;;
        sparsellm8b_s[567]0)
            sbatch -J "$jobname" --partition=A100,H200 \
                /home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_sparsellm.sh 8b "${jobname#sparsellm8b_s}" ;;
        sparsellm_s[567]0)
            sbatch -J "$jobname" --partition=A100,H200 \
                /home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_sparsellm.sh 1.7b "${jobname#sparsellm_s}" ;;
        sparsellm4b_s[567]0)
            sbatch -J "$jobname" --partition=A100,H200 \
                /home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_sparsellm.sh 4b "${jobname#sparsellm4b_s}" ;;
        norefresh4b_*|wokd4b_*|alpspgdtr_*|alpspgdnotr_*|alpsretrainnoopd_*|oursd003_*|cubicnopgd_*|norefresh8b_*)
            # job names here are already the launcher's arm names
            arm=$(echo "$jobname" | sed 's/_s\([0-9]\+\)$/_seed\1/; s/_seed_seed/_seed/')
            sbatch -J "$jobname" --partition=A100,H200 --time=1-12:00:00 "$LAUNCH_HANDOFF" "$arm" ;;
        *) say "  unknown job name $jobname -- not resubmitting"; return 1 ;;
    esac
}

say "watchdog up (interval ${INTERVAL}s, stall ${STALL_MIN}min / ${STALL_LOAD_MIN}min while loading)"
while true; do
    now=$(date +%s)
    while read -r jid jname; do
        [ -z "$jid" ] && continue
        f=$(ls -t "$LOGS"/*_"$jid".out 2>/dev/null | head -1)
        [ -z "$f" ] && continue
        sz=$(stat -c%s "$f")
        mark="$STATE/.size_$jid"
        if [ -f "$mark" ]; then
            read -r old_sz old_t < "$mark"
            if [ "$sz" -gt "$old_sz" ]; then
                echo "$sz $now" > "$mark"
            else
                mins=$(( (now - old_t) / 60 ))
                last=$(tail -c 4000 "$f" | tr '\r' '\n' | grep -v '^[[:space:]]*$' | tail -1)
                limit=$STALL_MIN
                case "$last" in
                    *"model weights format"*|*"Loading model from scratch"*|*"Starting to load model"*) limit=$STALL_LOAD_MIN ;;
                esac
                if [ "$mins" -ge "$limit" ]; then
                    say "STALLED $jid ($jname): log unchanged ${mins}min (limit ${limit}min) -- cancelling"
                    tail -2 "$f" | tr '\r' '\n' | tail -1 | sed 's/^/    last: /'
                    scancel "$jid"
                    rm -f "$mark"
                    sleep 5
                    resubmit "$jname" && say "  resubmitted $jname"
                elif [ "$mins" -ge 20 ]; then
                    say "quiet  $jid ($jname): ${mins}min without output"
                fi
            fi
        else
            echo "$sz $now" > "$mark"
        fi
    done < <(squeue -u "$USER" -h -t RUNNING -o "%i %j")
    sleep "$INTERVAL"
done
