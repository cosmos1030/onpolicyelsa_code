#!/bin/bash
# SWVr (ii) on the review B200 box (1.7B s70, roll2048 set): only the saliency of the SCOUT mask
# update changes, each at KL budget 0.02 and 0.05. Wanda/OBS are ranked within each layer (uniform
# per-layer allocation); the Adam-Fisher layer-rank run is their allocation-matched control;
# empirical Fisher (rollout response grad^2) stays global like SCOUT. Wanda/OBS statistics:
# 128 x 2048 training-document tokens, refreshed every mask_interval.
# Priority queue: GPU pair (0,1) runs obs -> wanda, pair (2,3) runs fisher_layerrank -> empfisher;
# within a pair GPU 0/2 = KL 0.02, GPU 1/3 = KL 0.05. The next run on a GPU starts when its
# previous one exits.
# usage: rb2048_b200_saliency.sh [DEBUG]   (DEBUG: one 40-step run per saliency, no saving / wandb)
B=/NHNHOME/log-postech/doyoonkim
S=$B/code_rb2048/b200_scripts/rb2048_b200.sh
STATS="--gmp_pgd_refit_calib_source=train --gmp_pgd_nm_refit_seqlen=2048 --gmp_pgd_nm_refit_calib_size=128"
declare -A ARGS=(
  [obs]="--gmp_saliency=obs $STATS"
  [wanda]="--gmp_saliency=wanda $STATS"
  [fisher_layerrank]="--gmp_pgd_layer_rank=true"
  [empfisher]="--gmp_fisher_source=opd_empirical"
)
if [ -n "$1" ]; then
  DBG="--steps=40 --gmp_pgd_nm_refit_calib_size=8 --gmp_fisher_emp_nsamples=4 --save_model=false --wandb=false --gmp_ckpt_every_steps=0"
  g=0
  for sal in empfisher fisher_layerrank obs wanda; do
    CUDA_VISIBLE_DEVICES=$g nohup bash $S s70_sal_${sal}_debug 0.7 ${ARGS[$sal]} $DBG \
        > $B/logs/rb2048_s70_sal_${sal}_debug.log 2>&1 &
    g=$((g+1))
  done
  wait; exit
fi
queue() { local g=$1 klb=$2; shift 2
  for sal in "$@"; do
    n=s70_sal_${sal}_klb${klb}
    CUDA_VISIBLE_DEVICES=$g bash $S $n 0.7 ${ARGS[$sal]} --gmp_pgd_kl_budget=$klb >> $B/logs/rb2048_$n.log 2>&1
  done
}
queue 0 0.02 obs wanda &
queue 1 0.05 obs wanda &
queue 2 0.02 fisher_layerrank empfisher &
queue 3 0.05 fisher_layerrank empfisher &
wait
