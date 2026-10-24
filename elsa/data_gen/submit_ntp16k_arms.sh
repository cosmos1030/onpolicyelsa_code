#!/bin/bash
# boosted / plain NTP-only arms on the 16,384-doc set, same settings as the orig16k baseline (1179601)
cd /home1/doyoonkim/projects/elsa/data_gen && /home1/doyoonkim/miniconda3/envs/rac/bin/python build_arm16k.py || exit 1
cd /home1/doyoonkim/projects/elsa/scripts
for arm in boost16k plain16k; do
  CKPT_EVERY=256 RUN_TAG=ntponly_ot3${arm} sbatch --export=ALL --job-name=ours_1.7b_s50_ntponly_ot3${arm} slurm_gmp_pgd_grow_to_target_qwen3_1.7b.sh 0.5 0.02 2048 32 cosine 2048 5e-5 /home1/doyoonkim/projects/elsa/data_gen/ot3/${arm}.jsonl 8192 true reasoning_qwen3_1.7b_nostrip8192 fisher global 1,0,0 32 0 4 8 0.35 false true
done
