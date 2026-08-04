#!/bin/bash
# Register all Qwen3-1.7B sweeps and save IDs
cd /home1/doyoonkim/projects/elsa
WANDB=/home1/doyoonkim/miniconda3/envs/rac/bin/wandb
OUT=/home1/doyoonkim/projects/elsa/scripts/qwen3_sweep_ids.txt
> "$OUT"

for yaml in sweep_configs/qwen3_1.7b/*.yaml; do
    name=$(basename "$yaml" .yaml)
    result=$($WANDB sweep "$yaml" 2>&1)
    sid=$(echo "$result" | grep "sweep with ID:" | grep -o '[a-z0-9]\{8\}$')
    if [ -n "$sid" ]; then
        echo "$sid $name" | tee -a "$OUT"
    else
        echo "FAILED $name" | tee -a "$OUT"
    fi
done
echo "=== Done: $(wc -l < $OUT) sweeps registered ==="
