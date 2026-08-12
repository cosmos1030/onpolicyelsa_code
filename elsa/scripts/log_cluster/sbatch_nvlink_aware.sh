#!/bin/bash
# NVLink-aware sbatch wrapper for log-node07 (8x H200): GPUs 0-3 are one
# NV6-connected NVLink domain, GPUs 4-7 are a separate domain, and links
# ACROSS the two domains are SYS (PCIe + QPI/UPI, no NVLink) -- confirmed via
# `nvidia-smi topo -m` on that node. SLURM's gres allocator has no topology
# awareness: a plain `--gres=gpu:2` request can land on e.g. IDX 3-4
# (spanning both domains), which then pays PCIe latency for every FSDP
# all-gather/reduce-scatter instead of NVLink for that job's entire runtime.
#
# This wraps a normal `sbatch <script> <args...>` call: submits, waits for
# the job to leave PENDING, reads the physical GPU IDX SLURM actually
# assigned (scontrol show job -d), and if those GPUs span both NVLink
# domains, cancels the job and resubmits (up to MAX_RETRIES times) instead
# of leaving a slower-than-necessary job running. There's no way to pin
# specific GPU indices via sbatch flags on this cluster, so retry-until-
# lucky is the only lever available without cluster-admin access to
# gres.conf.
#
# Only meaningful for <=4-GPU jobs (each NVLink domain has 4 GPUs) -- exactly
# our current FSDP 2-GPU use case. Hardcoded to log-node07's topology; if
# H200 jobs ever land on a different node, re-check `nvidia-smi topo -m`
# there and update GROUP_A/GROUP_B below.
#
# Usage: ./sbatch_nvlink_aware.sh <sbatch-script> [sbatch-script-args...]
# e.g.:  ./sbatch_nvlink_aware.sh slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh \
#          0.5 1e-4 0.02 /path/to/data.jsonl 32 cosine fisher myproj quick

set -euo pipefail

MAX_RETRIES=15
RETRY_SLEEP=20
GROUP_A=(0 1 2 3)
GROUP_B=(4 5 6 7)

in_group() {
    local idx=$1; shift
    for g in "$@"; do [[ "$idx" == "$g" ]] && return 0; done
    return 1
}

parse_idx() {
    # SLURM's IDX field looks like "0-1" or "2,7" or "0-3" -- expand to a
    # space-separated list of individual indices.
    local raw=$1
    local result=()
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then
            local start=${part%-*}
            local end=${part#*-}
            for ((i = start; i <= end; i++)); do result+=("$i"); done
        else
            result+=("$part")
        fi
    done
    echo "${result[@]}"
}

attempt=0
while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    job_id=$(sbatch --parsable "$@")
    echo "[nvlink-aware] attempt $attempt: submitted job $job_id"

    # Wait for it to leave PENDING (either RUNNING or gone/failed).
    state=""
    for _ in $(seq 1 120); do
        state=$(squeue -j "$job_id" -h -o '%T' 2>/dev/null || echo "")
        [[ -z "$state" || "$state" != "PENDING" ]] && break
        sleep 5
    done

    if [[ -z "$state" ]]; then
        echo "[nvlink-aware] job $job_id left the queue before we could check (finished/failed fast) -- leaving as-is"
        exit 0
    fi
    if [[ "$state" != "RUNNING" ]]; then
        echo "[nvlink-aware] job $job_id in state $state (not RUNNING after wait) -- leaving as-is"
        exit 0
    fi

    idx_raw=$(scontrol show job "$job_id" -d 2>/dev/null | grep -oP 'IDX:\K[0-9,-]+' | head -1)
    if [[ -z "$idx_raw" ]]; then
        echo "[nvlink-aware] could not read GPU IDX for job $job_id -- leaving as-is"
        exit 0
    fi
    idx_list=($(parse_idx "$idx_raw"))
    echo "[nvlink-aware] job $job_id got GPU IDX: ${idx_list[*]}"

    all_in_a=true
    all_in_b=true
    for idx in "${idx_list[@]}"; do
        in_group "$idx" "${GROUP_A[@]}" || all_in_a=false
        in_group "$idx" "${GROUP_B[@]}" || all_in_b=false
    done

    if $all_in_a || $all_in_b; then
        echo "[nvlink-aware] job $job_id: GPUs ${idx_list[*]} share one NVLink domain -- keeping it"
        exit 0
    fi

    echo "[nvlink-aware] job $job_id: GPUs ${idx_list[*]} SPAN both NVLink domains -- cancelling and retrying"
    scancel "$job_id"
    sleep "$RETRY_SLEEP"
done

echo "[nvlink-aware] gave up after $MAX_RETRIES attempts -- submitting one last time without checking"
exec sbatch "$@"
