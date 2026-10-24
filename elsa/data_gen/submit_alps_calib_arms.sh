#!/bin/bash
# ALPS 1.7B one-shot with the calibration corpus swapped between the three boosting-comparison arms
# (same 8,000 OT3 prompts + same 2,000 FineWeb; only how the OT solutions were written differs),
# s50/s60 x calibration seeds 42/0/1, each chained into the standard long eval.
cd /home1/doyoonkim/projects/elsa/data_gen && /home1/doyoonkim/miniconda3/envs/rac/bin/python build_arm8k.py || exit 1
D=/home1/doyoonkim/projects/elsa/data_gen/ot3
for arm in orig boost8k plain8k; do
  f=$D/$arm.jsonl; [ $arm = orig ] && f=$D/orig.jsonl
  tag=_ot3$arm; [ $arm = orig ] && tag=_ot3orig8k
  for sp in 0.5 0.6; do
    p=$(python3 -c "print(int($sp*100))")
    for sd in 42 0 1; do
      t=n128$tag; [ $sd != 42 ] && t=${t}_cseed$sd
      cd /home1/doyoonkim/projects/ALPS
      J=$(ALPS_DATA=$f CALIB_TAG=$tag SEED=$sd sbatch --parsable --export=ALL --job-name=alps_calib_${t}_s$p slurm_alps_ncalib.sh 1.7b $sp 128)
      cd /home1/doyoonkim/projects/elsa/scripts
      E=$(sbatch --parsable --dependency=afterok:$J --job-name=eval_alps_${t}_s$p slurm_eval_full.sh /home1/doyoonkim/projects/elsa/models/qwen3_1.7b_alps_${t}_s${p}pct s3_1.7b_s${p}_alps_${t} alps $sp reasoning_qwen3_1.7b_nostrip8192 "" 42 1 true true long)
      echo "$arm s$p seed$sd alps=$J eval=$E"
    done
  done
done
