#!/bin/bash
# Run lines ("<name> <sparsity> <extra flags...>") of the roll2048 rebuttal set: rb2048_runs.sh <s80|scout|all>
if [ "$1" = s80 ] || [ "$1" = all ]; then
  for SCH in cubic linear; do for E in 128 256 512 1024; do
    echo "s80_${SCH}_end${E}_upd8 0.8 --gmp_pgd=false --gmp_pgd_grow_to_target=false --gmp_sparse_train_steps=0 --gmp_pruning_end_ratio=$(python3 -c "print($E/2048)") --gmp_growth_schedule=$SCH --gmp_schedule_update_interval=8"
  done; done
fi
if [ "$1" = scout ] || [ "$1" = all ]; then
  for C in 2 8 16; do echo "s70_scout_calib$C 0.7 --gmp_pgd_kl_calib_size=$C"; done
  echo "s70_scout_ref 0.7"
  echo "s80_scout_ref 0.8"
  # Hz6h ablations (s70, one change each vs s70_scout_ref): no revival after target / OPD loss off with
  # rollout screening kept / off-policy screening (training doc: whole prompt + teacher CoT cut at 2048) with OPD kept
  echo "s70_scout_freeze 0.7 --gmp_pgd_freeze_after_target=true"
  echo "s70_scout_opdoff 0.7 --gmp_opd_loss_off=true"
  echo "s70_scout_screentrain 0.7 --gmp_pgd_screen_source=train"
fi
