#!/bin/bash
# Persistent SLURM job monitor — runs via crontab every 5 min
# Detects completions/failures and writes pending_updates.log for Claude to pick up

LOG="/home1/doyoonkim/projects/elsa/logs/monitor_slurm.log"
PENDING="/home1/doyoonkim/projects/elsa/logs/pending_updates.log"
SLURM_LOG_DIR="/home1/doyoonkim/projects/elsa/logs"
mkdir -p "$(dirname "$LOG")"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

# ── Watch list (update job IDs as experiments change) ─────────────────────────
declare -A WATCH
WATCH[642665]="SAFE 1.7B S50"
WATCH[642666]="SAFE 1.7B S60"
WATCH[642667]="SAFE 1.7B S70"
WATCH[642668]="TR-GMP 4B warmup+lasso"
WATCH[642669]="TR-GMP 4B warmup-only"
WATCH[642670]="TR-GMP 4B lasso-only"
WATCH[639726]="GMP 8B S50 eval"
WATCH[642524]="GMP 8B S70 eval"
WATCH[639737]="ELSA 4B sweep"
WATCH[639748]="ELSA 4B sweep"
WATCH[639763]="ELSA 4B sweep"

STATE_DIR="/home1/doyoonkim/projects/elsa/logs/.job_states"
mkdir -p "$STATE_DIR"

echo "[$(ts)] === monitor check ===" >> "$LOG"

for jid in "${!WATCH[@]}"; do
    label="${WATCH[$jid]}"
    state_file="$STATE_DIR/${jid}.state"
    cur_state=$(squeue -j "$jid" -h -o "%T %N" 2>/dev/null)
    prev_state=$(cat "$state_file" 2>/dev/null || echo "UNKNOWN")

    if [ -n "$cur_state" ]; then
        echo "[$(ts)] $jid ($label): $cur_state" >> "$LOG"
        echo "$cur_state" > "$state_file"

        # Early error detection
        logfile=$(ls "${SLURM_LOG_DIR}"/*_${jid}.out 2>/dev/null | head -1)
        if [ -n "$logfile" ]; then
            err=$(grep -oE "OOM|out of memory|CUDA error|ServicePollForTokenError|CUDA unknown error" \
                "$logfile" 2>/dev/null | tail -1)
            if [ -n "$err" ] && [ "$prev_state" != "ERROR:$err" ]; then
                echo "[$(ts)] ALERT $jid ($label) ERROR: $err" >> "$LOG"
                echo "$(ts) | FAIL | $jid | $label | $err" >> "$PENDING"
                echo "ERROR:$err" > "$state_file"
            fi
        fi
    else
        # Job not in queue — ended
        [ "$prev_state" = "ENDED" ] && continue

        logfile=$(ls "${SLURM_LOG_DIR}"/*_${jid}.out 2>/dev/null | head -1)
        if [ -n "$logfile" ]; then
            if grep -q "##### END #####" "$logfile" 2>/dev/null; then
                err=$(grep -oE "OOM|out of memory|CUDA error|ServicePollForTokenError|exit [1-9]" \
                    "$logfile" 2>/dev/null | tail -1)
                if [ -n "$err" ]; then
                    echo "[$(ts)] $jid ($label): ENDED with error: $err" >> "$LOG"
                    echo "$(ts) | FAIL | $jid | $label | $err" >> "$PENDING"
                else
                    math=$(grep -oP "(?<=math500[:= ])[0-9]+\.[0-9]+" "$logfile" 2>/dev/null | tail -1)
                    ppl=$(grep -oP "(?<=C4[^0-9]{0,5})[0-9]+\.[0-9]+" "$logfile" 2>/dev/null | tail -1)
                    echo "[$(ts)] $jid ($label): DONE math=$math ppl_c4=$ppl" >> "$LOG"
                    echo "$(ts) | DONE | $jid | $label | math=$math ppl_c4=$ppl" >> "$PENDING"
                fi
            else
                last=$(tail -3 "$logfile" 2>/dev/null | grep -v "^$" | tail -1 | cut -c1-120)
                echo "[$(ts)] $jid ($label): ENDED no END marker — $last" >> "$LOG"
                echo "$(ts) | FAIL | $jid | $label | no END marker: $last" >> "$PENDING"
            fi
        else
            echo "[$(ts)] $jid ($label): not in queue, no log" >> "$LOG"
            echo "$(ts) | UNKNOWN | $jid | $label | no log" >> "$PENDING"
        fi
        echo "ENDED" > "$state_file"
    fi
done

echo "[$(ts)] === done ===" >> "$LOG"
tail -2000 "$LOG" > "${LOG}.tmp" && mv "${LOG}.tmp" "$LOG"
