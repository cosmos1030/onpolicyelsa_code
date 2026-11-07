# OLMo-3-7B-Think HF repos (cosmos1030)

Renamed 2026-10-10 from the auto-generated main.py/ALPS ids. HF redirects the old ids.
Full action log (verify / rename / delete, with sizes): b200 container
`/NHNHOME/log-postech/doyoonkim/logs/hf_rename_cleanup_log.tsv`.
Local copies were deleted only after every *.safetensors/*.json file matched the HF repo by name and size.

| run | repo | old id | local copy |
|---|---|---|---|
| SCOUT s70 klb0.03 lr1e-4 roll2048 | olmo3-7b-think-scout-s70-klb0.03-roll2048 | gmp-kd3e-1-7b-s70pct-lr1e-4_20261009_234011 | deleted (model 18G + ckpt 47G) |
| ALPS->NTP+KD+OPD s70 lr1e-4 roll2048 | olmo3-7b-think-alps-ntpkdopd-s70-lr1e-4-roll2048 | gmp-kd3e-1-7b-s70pct-lr1e-4_20261010_044606 | deleted (18G + ckpt 47G) |
| ALPS one-shot s70 | olmo3-7b-think-alps-oneshot-s70 | alps-s70pct_20261008_190657 | deleted (14G) |
| SCOUT s50 klb0.01 lr5e-5 roll2048 | olmo3-7b-think-scout-s50-klb0.01-roll2048 | gmp-kd3e-1-7b-s50pct-lr5e-5_20261010_111943 | deleted (18G + ckpt 47G) |
| ALPS one-shot s50 | olmo3-7b-think-alps-oneshot-s50 | alps-s50pct_20261010_074012 | kept until ALPS-SFT s50 finishes |

Pending (handled automatically by logs/hf_post_cleanup_watch.sh on exit 0):
- ALPS->SFT s50 (old-style id) -> rename to olmo3-7b-think-alps-ntpkdopd-s50-lr5e-5-roll2048, delete local
- SCOUT s60 (old-style id) -> rename to olmo3-7b-think-scout-s60-klb0.01-roll2048, delete local
- SCOUT s70 klb0.02, ALPS one-shot s60, ALPS->SFT s60: launchers now pass --hub_model_id
  (olmo3-7b-think-{scout-s<P>-klb<K>-roll<L>[-jump] | alps-oneshot-s<P> | alps-ntpkdopd-s<P>-lr<LR>-roll<L>}),
  timestamp appended if the name already exists; local copies deleted after verification.
