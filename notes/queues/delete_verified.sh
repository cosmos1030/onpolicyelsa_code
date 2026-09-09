#!/bin/bash
# Verified-on-Hub local model dirs. Each was confirmed reachable on the Hub
# with every local *.safetensors present remotely, at generation time.
# Review, then run.  Recover any of these with huggingface-cli download <repo>.
set -u

# pre-opkdfix 8B grow_to_target (superseded by the resweep)
#   20.1 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260901_165041
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260901_153749'

# pre-opkdfix 8B grow_to_target (superseded by the resweep)
#   20.1 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260901_165249
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260901_153400'

# pre-opkdfix 8B grow_to_target (superseded by the resweep)
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260902_230621
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260902_215018'

# pre-opkdfix 8B grow_to_target (superseded by the resweep)
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_225227
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260901_212817'

# pre-opkdfix 8B grow_to_target (superseded by the resweep)
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260902_234454
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260902_221712'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.9 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr1e-4_20260826_230856
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr0.0001_onpol_lmda0.33_20260826_214103'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.9 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr1e-4_20260827_060353
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr0.0001_onpol_lmda0.33_20260827_042748'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.8 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260826_230731
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260826_215416'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.8 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260827_055110
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260827_044056'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.8 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr1e-4_20260826_231619
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr0.0001_onpol_lmda0.33_20260826_215546'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.8 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr1e-4_20260827_060409
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr0.0001_onpol_lmda0.33_20260827_044241'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.9 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260826_232251
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260826_215407'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   11.9 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260827_060349
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260827_043702'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr1e-4_20260828_050544
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr0.0001_onpol_lmda0.33_20260828_034431'

# abandoned klgate variants (duringgrowth / posttargetonly)
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr1e-4_20260827_145722
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr0.0001_onpol_lmda0.33_20260827_133331'

# total 235.9 GB

# ---- stage 2: dirs archived to the Hub on 2026-09-09 (manifest VERIFIED) ----
#   20.1 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260827_133259
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260827_133259'
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260827_230324
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260827_230324'
#   20.2 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260906_105649
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260906_105649'
#   16.6 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260820_033814
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260820_033814'
#   11.9 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260828_073925
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s70pct_lr0.0001_onpol_lmda0.33_20260828_073925'
#   8.1 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260828_125050
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260828_125050'
#   8.1 GB   hub: https://huggingface.co/cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260828_125227
rm -rf -- '/NHNHOME/log-postech/doyoonkim/models/gmp_s50pct_lr5e-05_onpol_lmda0.33_20260828_125227'
# stage-2 total 105.2 GB
