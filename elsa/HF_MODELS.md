# HF 업로드 모델 목록 (cosmos1030)

생성: 2026-10-11, `wandb` 학습 런의 `hub_model_id` + ALPS 로그의 `Uploaded to` 줄로 대조. 자동 생성(스크립트: 세션 scratchpad `hf_registry.py`/`hf_md.py`).

**sync 열:** OPD 롤아웃용 vLLM 가중치 동기화 상태. `BUG` = 2026-07-21~10-09 단일 GPU in-process 경로(qkv_proj·gate_up_proj가 학습 내내 load 시점 값으로 고정, 856f814에서 수정). `ok` = sidecar / FSDP / 수정 이후. `n/a` = OPD 없음.

**engine 열:** in-proc = 학습 프로세스 안 vLLM, sidecar = 별도 프로세스(같은 GPU), FSDPn = n GPU FSDP + 서버 프로세스.

**w 열:** NTP/KD/OPD loss 가중치. roll = OPD 롤아웃 최대 생성 길이, ri = 롤아웃 갱신 주기(step).

## 1. 학습 모델 (wandb 런과 연결됨)

| 업로드일 | 리포 | 종류 | 크기 | sp | lr | KL budget | w | roll | ri | engine | sync | 시작 모델 | avg4(seed42) | wandb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-09 | `gmp-kd3e-1-4b-s0pct-lr1e-4_20261010_021511` | quant 3bit | 4b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_awq_dense_w3_g128 | 67.4 | 4yqsuqff |
| 2026-10-09 | `gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_145513` | quant 2bit | 4b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_gptq_w2_g128_s0pct | 23.7 | voooo6nx |
| 2026-10-09 | `gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_133459` | quant 2bit | 4b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_gptq_w2_g128_s0pct |  | txuhbhso |
| 2026-10-08 | `gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_053427` | quant 3bit | 4b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_gptq_w3_g128_s0pct | 73.9 | e93cv3q0 |
| 2026-10-08 | `gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_052708` | quant 3bit | 4b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_gptq_w3_g128_s0pct | 74.4 | 3p8ru9uh |
| 2026-10-07 | `gmp-kd3e-1-8b-s70pct-lr1e-4_20261007_105708` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.03 | 0.33/0.33/0.33 | 2048 | 32 | FSDP4 | ok | Qwen3-8B | 61.0 | rknqee47 |
| 2026-10-07 | `gmp-kd3e-1-8b-s60pct-lr5e-5_20261007_094812` | SCOUT(Ours) | 8b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | FSDP4 | ok | Qwen3-8B | 73.0 | 9w762d6y |
| 2026-10-07 | `gmp-kd3e-1-8b-s50pct-lr5e-5_20261007_093446` | SCOUT(Ours) | 8b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | FSDP4 | ok | Qwen3-8B | 78.5 | zal3q33u |
| 2026-10-07 | `gmp-kd3e-1-8b-s0pct-lr1e-4_20261008_013029` | quant 2bit | 8b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | FSDP4 | ok | Qwen3-8B | 51.0 | yalo5jvc |
| 2026-10-07 | `gmp-kd3e-1-8b-quant3b-g128-roll2048-lr1e-4_20261007` | quant 3bit | 8b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | FSDP4 | ok | Qwen3-8B | 75.3 | 61iipz5b |
| 2026-10-07 | `gmp-kd3e-1-4b-s70pct-lr1e-4_20261008_023855` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | 1cfa9a7208912126459214e8b04321603b3df60c | 55.9 | lh9jte1w |
| 2026-10-07 | `gmp-kd3e-1-4b-s70pct-lr1e-4_20261008_015739` | ALPS->train | 4b | 0.7 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_alps_s70pct |  | ma7dalx2 |
| 2026-10-07 | `gmp-kd3e-1-4b-s60pct-lr5e-5_20261007_143047` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | 1cfa9a7208912126459214e8b04321603b3df60c | 65.2 | n85zk0y8 |
| 2026-10-07 | `gmp-kd3e-1-4b-s60pct-lr1e-4_20261008_011535` | ALPS->train | 4b | 0.6 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_alps_s60pct | 67.0 | hw49lx0l |
| 2026-10-07 | `gmp-kd3e-1-4b-s50pct-lr5e-5_20261008_010124` | SCOUT(Ours) | 4b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | 1cfa9a7208912126459214e8b04321603b3df60c | 76.2 | 3bkajmtu |
| 2026-10-07 | `gmp-kd3e-1-4b-s50pct-lr5e-5_20261007_224232` | ALPS->train | 4b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | sidecar | ok | qwen3_4b_alps_s50pct | 74.3 | riuqp40l |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261007_141017` | ALPS->train | 1.7b | 0.7 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s70pct | 22.5 | p8wvevy3 |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261007_104809` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | ycs8ahvs |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s60pct-lr5e-5_20261007_154206` | SCOUT(Ours) | 1.7b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | 0z4jme5y |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s60pct-lr1e-4_20261008_043037` | ALPS->train | 1.7b | 0.6 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s60pct | 48.6 | v4uey7v0 |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s50pct-lr5e-5_20261007_150922` | SCOUT(Ours) | 1.7b | 0.5 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | 9c63ffdw |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s50pct-lr1e-4_20261008_035636` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s50pct | 58.7 | 0yaitdjr |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261008_035937` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_gptq_w2_g128_s0pct |  | 8nptw867 |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_232355` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_gptq_w3_g128_s0pct |  | 896szvn0 |
| 2026-10-07 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_163721` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | 8zeu6irr |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261006_140940` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 1024 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 35.3 | apot0v4r |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261006_125005` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 2048 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | 8vz8taeq |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_025825` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 54.6 | h99wfk5v |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_191852` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 14.4 | dy3dhd2p |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_174408` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 24.4 | oj7d2d62 |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_160442` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | y359h64d |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_151503` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 52.3 | xakdnk4a |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_135840` | quant 2bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | 0ei31r5u |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_131408` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 52.1 | pqa7dq77 |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_122929` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | e5sw329s |
| 2026-10-06 | `gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_105401` | quant 3bit | 1.7b | 0 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e |  | y051wp9o |
| 2026-10-05 | `sparsellm-s0pct_20261005_172741` | GMP/other | 1.7b | 0 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | im0xf1t0 |
| 2026-10-05 | `sparsellm-s0pct_20261005_172314` | GMP/other | 1.7b | 0 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | jxr6i8hb |
| 2026-10-05 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_071136` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 | 47.0 | 3d3vnkj8 |
| 2026-10-05 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_065934` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 |  | to2tbem5 |
| 2026-10-05 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_055538` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 |  | 7xppzuwx |
| 2026-10-02 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261002_160223` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 | 46.4 | ullepzlb |
| 2026-10-02 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261002_153016` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 | 47.3 | aprxceyq |
| 2026-10-01 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_111538` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 | 45.9 | ffmzdhfl |
| 2026-10-01 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_111255` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 | 46.3 | 0cgsbo6a |
| 2026-10-01 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_102449` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 |  | rfovl5wo |
| 2026-10-01 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_095927` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s24 |  | t5brwrfs |
| 2026-09-29 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_072819` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP2 | ok | qwen3_1.7b_alps_s24 | 47.4 | uiyeocv7 |
| 2026-09-29 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_060834` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP2 | ok | qwen3_1.7b_alps_s24 | 47.8 | oueiexlt |
| 2026-09-29 | `gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_033246` | ALPS->train | 1.7b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP2 | ok | qwen3_1.7b_alps_s24 |  | au400obq |
| 2026-09-26 | `gmp-kd5e-1-1.7b-s70pct-lr5e-5_20260926_153440` | ALPS->train | 1.7b | 0.7 | 5e-05 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP2 | n/a(no OPD) | qwen3_1.7b_alps_s70pct | 16.1 | qh0knzoz |
| 2026-09-26 | `gmp-kd3e-1-s60pct-lr5e-5_20260901_071402` | SCOUT(Ours) | 1.7b | 0.6 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 40.5 | hcd9uxpn |
| 2026-09-25 | `alps-s80pct_20260925_235139` | GMP/other | 8b | 80 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | u78g9i5x |
| 2026-09-24 | `gmp-kd5e-1-8b-s70pct-lr1e-4_20260924_101832` | ALPS->train | 8b | 0.7 | 0.0001 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP2 | n/a(no OPD) | qwen3_8b_alps_s70pct | 55.8 | 0yl2g96n |
| 2026-09-24 | `gmp-kd3e-1-s60pct-lr5e-5_20260903_081011` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 512 | 4096 | FSDP8 | ok | Qwen3-4B | 58.0 | u4m6hzoz |
| 2026-09-24 | `gmp-kd3e-1-s60pct-lr5e-5_20260902_173413` | SCOUT(Ours) | 1.7b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 43.2 | 94pwg46m |
| 2026-09-23 | `gmp-kd3e-1-s70pct-lr1e-4_20260913_204305` | ALPS->train | 4b | 0.7 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | qwen3_4b_alps_s70pct | 43.1 | es235y9q |
| 2026-09-23 | `gmp-kd3e-1-s70pct-lr1e-4_20260908_102348` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 45.9 | dkek5q9w |
| 2026-09-23 | `gmp-kd3e-1-s60pct-lr5e-5_20260814_231104` | ALPS->train | 4b | 0.6 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_4b_alps_s60pct | 57.1 | frhabnvh |
| 2026-09-23 | `gmp-kd3e-1-s60pct-lr5e-5_20260811_110729` | GMP/other | 4b | 0.6 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 59.9 | m4xxw7p7 |
| 2026-09-23 | `gmp-kd3e-1-s60pct-lr5e-5_20260811_024343` | GMP/other | 4b | 0.6 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 58.1 | dh0trefz |
| 2026-09-23 | `gmp-kd3e-1-s60pct-lr1e-4_20260811_114747` | GMP/other | 4b | 0.6 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 55.2 | 0paqfgde |
| 2026-09-23 | `gmp-kd3e-1-s60pct-lr1e-4_20260811_111500` | GMP/other | 4b | 0.6 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 58.0 | k9p7a1lp |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr5e-5_20260811_020803` | GMP/other | 4b | 0.5 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 65.3 | 4c4fzhd2 |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr5e-5_20260811_020708` | GMP/other | 4b | 0.5 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 65.5 | umd2oqz8 |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr1e-5_20260812_175503` | ALPS->train | 4b | 0.5 | 1e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_4b_alps_s50pct | 65.8 | ihbalua1 |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr1e-4_20260824_125640` | GMP/other | 4b | 0.5 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 45.9 | eyv3mlfj |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr1e-4_20260811_024904` | GMP/other | 4b | 0.5 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 61.5 | xwz7gs3y |
| 2026-09-23 | `gmp-kd3e-1-s50pct-lr1e-4_20260811_024202` | GMP/other | 4b | 0.5 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 63.9 | 1zq9kdxr |
| 2026-09-22 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_062543` | GMP/other | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | sidecar | ok | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 30.7 | nxry83x2 |
| 2026-09-22 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_033414` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 99999 | 0.33/0.33/0.33 | 512 | 32 | sidecar | ok | qwen3_1.7b_alps_s70pct | 22.1 | dpiwlkkh |
| 2026-09-22 | `gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_033301` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | sidecar | ok | qwen3_1.7b_alps_s70pct | 23.1 | bg86uv4f |
| 2026-09-18 | `gmp-kd3e-1-8b-s80pct-lr1e-4_20260918_090751` | SCOUT(Ours) | 8b | 0.8 | 0.0001 | 0.03 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 40.7 | keuegrrb |
| 2026-09-17 | `gmp-kd5e-1-8b-s80pct-lr1e-4_20260917_105733` | ALPS->train | 8b | 0.8 | 0.0001 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | qwen3_8b_alps_s80pct | 27.7 | 4yuo2b36 |
| 2026-09-17 | `gmp-kd5e-1-8b-s70pct-lr1e-4_20260917_181513` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.03 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | Qwen3-8B | 59.7 | g5htdr2q |
| 2026-09-17 | `gmp-kd5e-1-4b-s80pct-lr1e-4_20260917_102335` | ALPS->train | 4b | 0.8 | 0.0001 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | qwen3_4b_alps_s80pct | 16.4 | crdshw6m |
| 2026-09-17 | `gmp-kd3e-1-8b-s80pct-lr1e-4_20260917_121924` | ALPS->train | 8b | 0.8 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | qwen3_8b_alps_s80pct | 30.5 | 9yyo7mmo |
| 2026-09-17 | `gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952` | ALPS->train | 4b | 0.8 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | qwen3_4b_alps_s80pct | 18.1 | q6vjas4j |
| 2026-09-17 | `gmp-4b-s80pct-lr1e-4_20260918_071932` | ALPS->train | 4b | 0.8 | 0.0001 | 0.01 | 0/0/1 | 512 | 32 | FSDP4 | ok | qwen3_4b_alps_s80pct | 15.0 | n9omc089 |
| 2026-09-17 | `gmp-4b-s70pct-lr1e-4_20260917_213402` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.5/0/0.5 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 50.2 | r8eomj61 |
| 2026-09-16 | `gmp-kd5e-1-s70pct-lr1e-4_20260916_215106` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0/0.5/0.5 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 43.8 | ft4chpqb |
| 2026-09-16 | `gmp-kd5e-1-s70pct-lr1e-4_20260916_183517` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0/0.5/0.5 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 30.1 | eb6j0en8 |
| 2026-09-16 | `gmp-kd5e-1-s60pct-lr5e-5_20260916_225506` | SCOUT(Ours) | 1.7b | 0.6 | 5e-05 | 0.01 | 0.5/0.5/0 | 512 | 32 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 40.6 | jo0mwm4z |
| 2026-09-16 | `gmp-kd5e-1-s50pct-lr5e-5_20260916_221150` | SCOUT(Ours) | 1.7b | 0.5 | 5e-05 | 0.02 | 0.5/0.5/0 | 512 | 32 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 50.4 | cnvme56d |
| 2026-09-16 | `gmp-kd3e-1-s80pct-lr1e-4_20260916_220740` | SCOUT(Ours) | 4b | 0.8 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 29.2 | q0kmiudv |
| 2026-09-16 | `gmp-kd3e-1-s80pct-lr1e-4_20260916_214905` | SCOUT(Ours) | 8b | 0.8 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 38.6 | 3j2qqve1 |
| 2026-09-15 | `gmp-kd5e-1-s70pct-lr5e-5_20260915_140911` | ALPS->train | 1.7b | 0.7 | 5e-05 | 0.01 | 0.5/0.5/0 | 512 | 32 | in-proc | n/a(no OPD) | qwen3_1.7b_alps_s70pct | 16.3 | n7natwjd |
| 2026-09-15 | `gmp-kd5e-1-s70pct-lr1e-4_20260915_192606` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | 1cfa9a7208912126459214e8b04321603b3df60c | 45.9 | vew8x5r5 |
| 2026-09-15 | `gmp-kd5e-1-s70pct-lr1e-4_20260915_140636` | ALPS->train | 1.7b | 0.7 | 0.0001 | 0.01 | 0.5/0.5/0 | 512 | 32 | in-proc | n/a(no OPD) | qwen3_1.7b_alps_s70pct | 18.8 | mo7cfjrw |
| 2026-09-15 | `gmp-kd5e-1-s70pct-lr1e-4_20260915_124804` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.5/0.5/0 | 512 | 32 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 28.0 | f5d0b6pf |
| 2026-09-15 | `gmp-kd5e-1-s60pct-lr5e-5_20260915_181935` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.02 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | 1cfa9a7208912126459214e8b04321603b3df60c | 58.5 | de0nllme |
| 2026-09-15 | `gmp-kd5e-1-s50pct-lr5e-5_20260915_130503` | SCOUT(Ours) | 4b | 0.5 | 5e-05 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | 1cfa9a7208912126459214e8b04321603b3df60c | 67.1 | jwqzoixm |
| 2026-09-15 | `gmp-kd3e-1-s80pct-lr1e-4_20260916_044553` | SCOUT(Ours) | 1.7b | 0.8 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 10.3 | hz4kw1sr |
| 2026-09-15 | `gmp-kd3e-1-s70pct-lr5e-5_20260915_160528` | ALPS->train | 1.7b | 0.7 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s70pct | 19.2 | vuyodo51 |
| 2026-09-14 | `gmp-s70pct-lr1e-4_20260915_051006` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0/0/1 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 23.4 | 5lankkks |
| 2026-09-13 | `gmp-kd5e-1-s70pct-lr1e-4_20260913_193547` | ALPS->train | 4b | 0.7 | 0.0001 | 0.01 | 0.5/0.5/0 | 512 | 32 | FSDP4 | n/a(no OPD) | qwen3_4b_alps_s70pct | 42.2 | 4n7itrc2 |
| 2026-09-10 | `gmp-kd3e-1-s70pct-lr1e-4_20260911_015957` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.03 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 53.1 | ajm5l60w |
| 2026-09-09 | `gmp-kd3e-1-s70pct-lr1e-4_20260909_221717` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 4096 | FSDP8 | ok | Qwen3-4B | 33.3 | izc2311t |
| 2026-09-09 | `gmp-kd3e-1-s70pct-lr1e-4_20260909_213205` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP8 | ok | Qwen3-4B | 38.4 | 3olz82te |
| 2026-09-08 | `gmp-kd3e-1-s70pct-lr1e-4_20260908_132323` | ALPS->train | 8b | 0.7 | 0.0001 | 0.01 | 0.33/0.33/0.33 | 512 | -1 | FSDP4 | ok | qwen3_8b_alps_s70pct | 51.0 | 3zpks6yu |
| 2026-09-07 | `gmp-kd3e-1-s60pct-lr5e-5_20260908_054305` | ALPS->train | 8b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | -1 | FSDP4 | ok | qwen3_8b_alps_s60pct | 62.7 | o90s9lyo |
| 2026-09-07 | `gmp-kd3e-1-s60pct-lr5e-5_20260907_142152` | SCOUT(Ours) | 8b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 63.3 | mgf2ka9s |
| 2026-09-07 | `gmp-kd3e-1-s50pct-lr5e-5_20260908_053513` | ALPS->train | 8b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | -1 | FSDP4 | ok | qwen3_8b_alps_s50pct | 67.5 | tb09czsq |
| 2026-09-07 | `gmp-kd3e-1-s50pct-lr5e-5_20260907_210125` | SCOUT(Ours) | 8b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 68.1 | roq1vwuw |
| 2026-09-06 | `gmp-kd3e-1-s70pct-lr1e-4_20260907_073958` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-8B | 53.1 | uywgha9p |
| 2026-09-04 | `gmp-kd3e-1-s70pct-lr1e-4_20260905_081916` | SCOUT(Ours) | 8b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 4096 | FSDP4 | ok | Qwen3-8B | 50.7 | n69llpef |
| 2026-09-03 | `gmp-kd3e-1-s70pct-lr1e-4_20260903_144159` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP8 | ok | Qwen3-4B | 46.1 | k43jzq69 |
| 2026-09-03 | `gmp-kd3e-1-s60pct-lr5e-5_20260903_142958` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP8 | ok | Qwen3-4B | 58.9 | 1npg0ri2 |
| 2026-09-03 | `gmp-kd3e-1-s50pct-lr5e-5_20260903_142204` | SCOUT(Ours) | 4b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP8 | ok | Qwen3-4B | 66.0 | kp4bd255 |
| 2026-09-02 | `gmp-kd3e-1-s70pct-lr1e-4_20260903_081638` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 4096 | FSDP8 | ok | Qwen3-4B | 47.8 | laphj0g7 |
| 2026-09-02 | `gmp-kd3e-1-s60pct-lr5e-5_20260903_071754` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP8 | ok | Qwen3-4B | 59.3 | 4h0qqsze |
| 2026-09-02 | `gmp-kd3e-1-s50pct-lr5e-5_20260903_080351` | SCOUT(Ours) | 4b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 4096 | FSDP8 | ok | Qwen3-4B | 65.9 | rxsmqs5p |
| 2026-09-01 | `gmp-kd3e-1-s60pct-lr5e-5_20260901_121104` | SCOUT(Ours) | 1.7b | 0.6 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 8 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 42.2 | oui58p86 |
| 2026-08-31 | `gmp-kd3e-1-s70pct-lr1e-4_20260901_084730` | SCOUT(Ours) | 1.7b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 31.2 | q9f33zl7 |
| 2026-08-31 | `gmp-kd3e-1-s70pct-lr1e-4_20260901_080954` | SCOUT(Ours) | 4b | 0.7 | 0.0001 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 48.0 | r5j1uw8d |
| 2026-08-31 | `gmp-kd3e-1-s60pct-lr5e-5_20260901_075817` | SCOUT(Ours) | 4b | 0.6 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 58.8 | 02vfcs1b |
| 2026-08-31 | `gmp-kd3e-1-s50pct-lr5e-5_20260901_071337` | SCOUT(Ours) | 1.7b | 0.5 | 5e-05 | 0.02 | 0.33/0.33/0.33 | 512 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 52.1 | 3wvt5gme |
| 2026-08-31 | `gmp-kd3e-1-s50pct-lr5e-5_20260901_060006` | SCOUT(Ours) | 4b | 0.5 | 5e-05 | 0.01 | 0.33/0.33/0.33 | 512 | 32 | FSDP4 | ok | Qwen3-4B | 67.4 | 544kemgd |
| 2026-08-28 | `gmp-kd5e-1-s70pct-lr5e-5_20260829_033354` | ALPS->train | 1.7b | 0.7 | 5e-05 | 0.01 | 0/0.5/0.5 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s70pct | 22.0 | chbqjyok |
| 2026-08-28 | `gmp-kd5e-1-s70pct-lr1e-4_20260829_053846` | GMP/other | 1.7b | 0.7 | 0.0001 | 0.02 | 0/0.5/0.5 | 256 | 32 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 26.0 | fzu9zers |
| 2026-08-27 | `alps-s70pct_20260827_144621` | GMP/other | 8b | 70 | None | None | None/None/0 | None | None | FSDP2 | n/a(no OPD) |  | 28.3 | 36ro8z91 |
| 2026-08-27 | `alps-s60pct_20260827_141236` | GMP/other | 8b | 60 | None | None | None/None/0 | None | None | FSDP2 | n/a(no OPD) |  | 57.6 | 199k4s03 |
| 2026-08-27 | `alps-s50pct_20260827_154700` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 50.6 | fwfztw1f |
| 2026-08-27 | `alps-s50pct_20260827_153718` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 49.3 | p5591btn |
| 2026-08-27 | `alps-s50pct_20260827_131723` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 50.1 | k557jw64 |
| 2026-08-27 | `alps-s50pct_20260827_131618` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 53.7 | gwiahcw6 |
| 2026-08-27 | `alps-s50pct_20260827_131415` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 52.7 | pvuiuif0 |
| 2026-08-27 | `alps-s50pct_20260827_131320` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 51.1 | hm77whhn |
| 2026-08-27 | `alps-s50pct_20260827_095701` | GMP/other | 8b | 50 | None | None | None/None/0 | None | None | FSDP2 | n/a(no OPD) |  | 66.8 | 5o0p2wqk |
| 2026-08-26 | `alps-s70pct_20260826_223323` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 4.7 | fao5u2ih |
| 2026-08-26 | `alps-s60pct_20260826_223415` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 32.3 | sg1ochko |
| 2026-08-26 | `alps-s50pct_20260826_223028` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 51.0 | 0e7mpjug |
| 2026-08-17 | `gmp-kd5e-1-s70pct-lr5e-5_20260817_135834` | GMP/other | 4b | 0.7 | 5e-05 | None | 0.5/0.5/0 | 256 | 1 | FSDP2 | n/a(no OPD) | 1cfa9a7208912126459214e8b04321603b3df60c | 40.6 | 4hz62cac |
| 2026-08-16 | `gmp-kd5e-1-s70pct-lr1e-4_20260817_000353` | GMP/other | 1.7b | 0.7 | 0.0001 | None | 0.5/0.5/0 | 512 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 23.4 | lotk3fgq |
| 2026-08-16 | `elsa-ntp-cot-s70pct-lr1e-4-lmda1e-2_20260816_110313` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 16.6 | lm35lo0o |
| 2026-08-16 | `elsa-ntp-cot-s50pct-lr5e-5-lmda5e-3_20260816_213357` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 14.4 | tyozh6vq |
| 2026-08-16 | `elsa-ntp-cot-s50pct-lr5e-5-lmda2e-3_20260816_205404` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 16.8 | c34vva8r |
| 2026-08-16 | `elsa-ntp-cot-s50pct-lr5e-5-lmda1e-2_20260816_225415` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 15.0 | uvzewp2k |
| 2026-08-15 | `elsa-offpolicy-kd-s70pct-lr1e-4-lmda5e-3_20260816_045739` | GMP/other | 1.7b | 0.7 | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 4.8 | 065ijzh1 |
| 2026-08-15 | `elsa-offpolicy-kd-s70pct-lr1e-4-lmda2e-3_20260816_050650` | GMP/other | 1.7b | 0.7 | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 3.9 | fqm65p2w |
| 2026-08-15 | `elsa-offpolicy-kd-s70pct-lr1e-4-lmda1e-2_20260816_050639` | GMP/other | 1.7b | 0.7 | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 4.6 | hchh9kpq |
| 2026-08-15 | `elsa-offpolicy-kd-s60pct-lr5e-5-lmda5e-4_20260816_045150` | GMP/other | 1.7b | 0.6 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 9.5 | 362w241g |
| 2026-08-15 | `elsa-offpolicy-kd-s60pct-lr5e-5-lmda2e-3_20260816_050528` | GMP/other | 1.7b | 0.6 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 5.9 | rgfeh8b2 |
| 2026-08-15 | `elsa-offpolicy-kd-s60pct-lr5e-5-lmda1e-3_20260816_044928` | GMP/other | 1.7b | 0.6 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 8.1 | ccr2ti8a |
| 2026-08-15 | `elsa-offpolicy-kd-s50pct-lr5e-5-lmda5e-4_20260816_041339` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 30.0 | xkqf9hjs |
| 2026-08-15 | `elsa-offpolicy-kd-s50pct-lr5e-5-lmda2e-3_20260816_043447` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 23.3 | dd8yaija |
| 2026-08-15 | `elsa-offpolicy-kd-s50pct-lr5e-5-lmda1e-3_20260816_043041` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 25.6 | 0kq8no5i |
| 2026-08-15 | `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260816_031909` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 20.1 | 5icfnslu |
| 2026-08-15 | `elsa-ntp-cot-s70pct-lr1e-4-lmda2e-3_20260816_032722` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 23.0 | p2dab8xi |
| 2026-08-15 | `elsa-ntp-cot-s60pct-lr5e-5-lmda5e-3_20260816_031849` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 31.3 | jds8tfmn |
| 2026-08-15 | `elsa-ntp-cot-s60pct-lr5e-5-lmda2e-3_20260816_031130` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 33.7 | dgw1mu7b |
| 2026-08-15 | `elsa-ntp-cot-s60pct-lr5e-5-lmda1e-2_20260816_032056` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 31.0 | dhp81z91 |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda5e-4_20260816_010750` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 55.6 | m2a9s7z4 |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda5e-3_20260815_101933` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 13.9 | y928qu5k |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda2e-3_20260816_023832` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 54.4 | cfvcjn2l |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda2e-3_20260815_092513` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 15.3 | slb62534 |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda1e-3_20260816_011159` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 56.0 | t9i56pyr |
| 2026-08-15 | `elsa-ntp-cot-s50pct-lr5e-5-lmda1e-2_20260815_092601` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 12.4 | xrhyjru7 |
| 2026-08-15 | `alps-s70pct_20260815_204218` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 3.2 | gt24i2yg |
| 2026-08-15 | `alps-s70pct_20260815_171107` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 3.4 | yp1ylrtn |
| 2026-08-15 | `alps-s70pct_20260815_155353` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | ocpqm8eu |
| 2026-08-15 | `alps-s60pct_20260815_220343` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 31.4 | rzjnobb8 |
| 2026-08-15 | `alps-s60pct_20260815_171309` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | atimoka6 |
| 2026-08-15 | `alps-s60pct_20260815_155320` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | pondkexf |
| 2026-08-15 | `alps-s50pct_20260815_171300` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 50.4 | mfh8kv7h |
| 2026-08-15 | `alps-s50pct_20260815_155037` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  |  | 6a3rfg8a |
| 2026-08-14 | `gmp-kd3e-1-s60pct-lr1e-4_20260814_193400` | ALPS->train | 4b | 0.6 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_4b_alps_s60pct | 58.0 | dqocd3f8 |
| 2026-08-14 | `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260814_214033` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 17.7 | msgqhsqn |
| 2026-08-14 | `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260814_094802` | GMP/other | 1.7b | 0.7 | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 5.1 | fv4daupw |
| 2026-08-14 | `elsa-ntp-cot-s70pct-lr1e-4-lmda2e-3_20260814_210955` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 22.8 | huo5opq9 |
| 2026-08-14 | `elsa-ntp-cot-s70pct-lr1e-4-lmda1e-2_20260814_223556` | GMP/other | 4b |  | 0.0001 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 16.1 | p0unbnsp |
| 2026-08-14 | `elsa-ntp-cot-s60pct-lr5e-5-lmda5e-3_20260814_123337` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 31.6 | 2bojnhsf |
| 2026-08-14 | `elsa-ntp-cot-s60pct-lr5e-5-lmda2e-3_20260814_122418` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 33.8 | jguhmyqu |
| 2026-08-14 | `elsa-ntp-cot-s60pct-lr5e-5-lmda2e-3_20260814_093427` | GMP/other | 1.7b | 0.6 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 11.3 | f7ojh6d8 |
| 2026-08-14 | `elsa-ntp-cot-s60pct-lr5e-5-lmda1e-2_20260814_184859` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 29.8 | s7zw09c9 |
| 2026-08-14 | `elsa-ntp-cot-s50pct-lr5e-5-lmda5e-4_20260814_091641` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 56.1 | wtk47hnc |
| 2026-08-14 | `elsa-ntp-cot-s50pct-lr5e-5-lmda2e-3_20260814_112734` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 55.0 | 8yhin37b |
| 2026-08-14 | `elsa-ntp-cot-s50pct-lr5e-5-lmda1e-3_20260814_110521` | GMP/other | 4b |  | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | Qwen3-4B | 56.4 | 0wspxea1 |
| 2026-08-13 | `gmp-kd3e-1-s70pct-lr1e-4_20260814_035030` | ALPS->train | 4b | 0.7 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_4b_alps_s70pct | 44.4 | 5x4prktp |
| 2026-08-13 | `elsa-ntp-cot-s60pct-lr5e-5-lmda1e-3_20260814_085708` | GMP/other | 1.7b | 0.6 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 11.2 | ktlm7ooj |
| 2026-08-13 | `elsa-ntp-cot-s50pct-lr5e-5-lmda5e-4_20260814_085925` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 34.3 | bv3i3jce |
| 2026-08-13 | `elsa-ntp-cot-s50pct-lr5e-5-lmda2e-3_20260814_085555` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 27.8 | 0e3mt0ge |
| 2026-08-13 | `elsa-ntp-cot-s50pct-lr5e-5-lmda1e-3_20260814_084930` | GMP/other | 1.7b | 0.5 | 5e-05 | None | 1/0/0 | 256 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 32.6 | j9a8kknc |
| 2026-08-12 | `gmp-s0pct-lr5e-5_20260813_034617` | GMP/other | 1.7b | 0 | 5e-05 | None | 0/0/1 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 62.4 | xq6pe7o2 |
| 2026-08-12 | `gmp-s0pct-lr1e-4_20260813_035857` | GMP/other | 1.7b | 0 | 0.0001 | None | 0/0/1 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 50.1 | xrfzkp1d |
| 2026-08-12 | `gmp-kd5e-1-s0pct-lr5e-5_20260813_012417` | GMP/other | 1.7b | 0 | 5e-05 | None | 0/0.5/0.5 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 63.0 | 0wnl4dva |
| 2026-08-12 | `gmp-kd5e-1-s0pct-lr1e-4_20260813_013030` | GMP/other | 1.7b | 0 | 0.0001 | None | 0/0.5/0.5 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 58.4 | oidgmmd4 |
| 2026-08-12 | `gmp-kd3e-1-s50pct-lr5e-5_20260812_132642` | ALPS->train | 4b | 0.5 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_4b_alps_s50pct | 65.9 | miysnxlq |
| 2026-08-12 | `gmp-kd1e0-s0pct-lr5e-5_20260813_041317` | GMP/other | 1.7b | 0 | 5e-05 | None | 0/1/0 | 512 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 62.4 | g1tc6mee |
| 2026-08-12 | `gmp-kd1e0-s0pct-lr1e-4_20260813_041002` | GMP/other | 1.7b | 0 | 0.0001 | None | 0/1/0 | 512 | 1 | in-proc | n/a(no OPD) | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | 61.1 | 3b261pxt |
| 2026-08-12 | `alps-s70pct_20260812_151749` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 3.5 | mjqz2zrc |
| 2026-08-12 | `alps-s60pct_20260812_151806` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 35.1 | 3svtuugl |
| 2026-08-12 | `alps-s50pct_20260812_151711` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 51.8 | pm8urv1l |
| 2026-08-11 | `gmp-kd3e-1-s70pct-lr5e-5_20260811_231644` | ALPS->train | 1.7b | 0.7 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s70pct | 20.5 | zexfluvb |
| 2026-08-11 | `gmp-kd3e-1-s70pct-lr5e-5_20260811_115604` | GMP/other | 4b | 0.7 | 5e-05 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | Qwen3-4B | 46.6 | 657qk0cq |
| 2026-08-11 | `gmp-kd3e-1-s60pct-lr1e-4_20260811_213610` | ALPS->train | 1.7b | 0.6 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s60pct | 44.5 | 53uwdeop |
| 2026-08-11 | `gmp-kd3e-1-s50pct-lr1e-4_20260811_191146` | ALPS->train | 1.7b | 0.5 | 0.0001 | None | 0.33/0.33/0.33 | 512 | 1 | in-proc | BUG(in-proc vLLM sync) | qwen3_1.7b_alps_s50pct | 53.0 | 16gmeadr |
| 2026-08-11 | `alps-s70pct_20260811_224211` | GMP/other | 1.7b | 70 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 4.2 | 0f78dam6 |
| 2026-08-11 | `alps-s60pct_20260811_224112` | GMP/other | 1.7b | 60 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 27.6 | odxdg1j1 |
| 2026-08-11 | `alps-s50pct_20260811_224425` | GMP/other | 1.7b | 50 | None | None | None/None/0 | None | None | in-proc | n/a(no OPD) |  | 47.4 | 26wfgd54 |

## 2. ALPS one-shot 모델 (학습 없음, 로그에서 연결)

| 업로드일 | 리포 | 로그 폴더 | Final PPL |
|---|---|---|---|
| 2026-10-10 | `alps-rollout2048mixsep-qwen3-1.7b-s70pct` | alps_rollout_mixsep_1.7b_s70_1114935 | 12.1365 |
| 2026-10-10 | `alps-rollout2048mix3sep-qwen3-1.7b-s70pct` | alps_rollout_mix3sep_1.7b_s70_1114939 | 11.3883 |
| 2026-10-10 | `alps-rollout2048mix3-qwen3-4b-s24` | alps_rollout_mix3_4b_s24_1114282 | 6.3465 |
| 2026-10-10 | `alps-rollout2048mix3-qwen3-1.7b-s24` | alps_rollout_mix3_1.7b_s24_1114266 | 7.5914 |
| 2026-10-10 | `alps-rollout2048mix-qwen3-4b-s24` | alps_rollout_mix_4b_s24_1114278 | 6.4425 |
| 2026-10-10 | `alps-rollout2048mix-qwen3-1.7b-s24` | alps_rollout_mix_1.7b_s24_1114262 | 7.7179 |
| 2026-10-10 | `alps-rollout2048freshfix3-qwen3-1.7b-s70pct` | alps_rollout_freshfix3_1.7b_s70_1114931 | 10.1081 |
| 2026-10-10 | `alps-rollout2048freshfix-qwen3-1.7b-s70pct` | alps_rollout_freshfix_1.7b_s70_1114927 | 11.1474 |
| 2026-10-10 | `alps-n256-qwen3-4b-s24` | alps_n256_4b_s24_1114270 | 6.2267 |
| 2026-10-10 | `alps-n256-qwen3-1.7b-s24` | alps_n256_1.7b_s24_1114254 | 7.4102 |
| 2026-10-10 | `alps-n128sg128-qwen3-4b-s24` | alps_n128sg128_4b_s24_1114274 | 6.4473 |
| 2026-10-10 | `alps-n128sg128-qwen3-1.7b-s24` | alps_n128sg128_1.7b_s24_1114258 | 7.7522 |
| 2026-10-10 | `alps-n128-qwen3-4b-s24` | alps_n128_4b_s24_1114250 | 6.3218 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-4b-s70pct` | alps_rollout_mix3_4b_s70_1107944 | 7.6651 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-4b-s60pct` | alps_rollout_mix3_4b_s60_1107940 | 5.8117 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-4b-s50pct` | alps_rollout_mix3_4b_s50_1107936 | 5.2311 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-1.7b-s70pct` | alps_rollout_mix3_1.7b_s70_1107932 | 11.3350 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-1.7b-s60pct` | alps_rollout_mix3_1.7b_s60_1107928 | 7.4860 |
| 2026-10-09 | `alps-rollout2048mix3-qwen3-1.7b-s50pct` | alps_rollout_mix3_1.7b_s50_1107924 | 6.3147 |
| 2026-10-09 | `alps-n512-qwen3-4b-s70pct` | alps_n512_4b_s70_1107128 | 7.8293 |
| 2026-10-09 | `alps-n512-qwen3-4b-s60pct` | alps_n512_4b_s60_1107126 | 5.7282 |
| 2026-10-09 | `alps-n512-qwen3-4b-s50pct` | alps_n512_4b_s50_1107124 | 5.2153 |
| 2026-10-09 | `alps-n512-qwen3-1.7b-s70pct` | alps_n512_1.7b_s70_1107122 | 11.7257 |
| 2026-10-09 | `alps-n512-qwen3-1.7b-s60pct` | alps_n512_1.7b_s60_1107120 | 7.1001 |
| 2026-10-09 | `alps-n512-qwen3-1.7b-s50pct` | alps_n512_1.7b_s50_1107118 | 6.1001 |
| 2026-10-09 | `alps-n256-qwen3-4b-s70pct` | alps_n256_4b_s70_1107113 | 7.8579 |
| 2026-10-09 | `alps-n256-qwen3-4b-s60pct` | alps_n256_4b_s60_1107111 | 5.7498 |
| 2026-10-09 | `alps-n256-qwen3-4b-s50pct` | alps_n256_4b_s50_1107109 | 5.2119 |
| 2026-10-09 | `alps-n256-qwen3-1.7b-s70pct` | alps_n256_1.7b_s70_1107107 | 11.8939 |
| 2026-10-09 | `alps-n256-qwen3-1.7b-s60pct` | alps_n256_1.7b_s60_1107105 | 7.1848 |
| 2026-10-09 | `alps-n256-qwen3-1.7b-s50pct` | alps_n256_1.7b_s50_1107103 | 6.0998 |
| 2026-10-09 | `alps-n128sg128-qwen3-4b-s70pct` | alps_n128sg128_4b_s70_1108111 | 8.1724 |
| 2026-10-09 | `alps-n128sg128-qwen3-4b-s60pct` | alps_n128sg128_4b_s60_1108107 | 5.8400 |
| 2026-10-09 | `alps-n128sg128-qwen3-4b-s50pct` | alps_n128sg128_4b_s50_1108103 | 5.2554 |
| 2026-10-09 | `alps-n128sg128-qwen3-1.7b-s70pct` | alps_n128sg128_1.7b_s70_1108099 | 12.1413 |
| 2026-10-09 | `alps-n128sg128-qwen3-1.7b-s60pct` | alps_n128sg128_1.7b_s60_1108095 | 7.5462 |
| 2026-10-09 | `alps-n128sg128-qwen3-1.7b-s50pct` | alps_n128sg128_1.7b_s50_1108091 | 6.2830 |
| 2026-10-08 | `alps-rollout2048rollonly-qwen3-1.7b-s70pct` | alps_rollout_rollonly_1.7b_s70_1104729 | 13.2614 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-4b-s70pct` | alps_rollout_mix_4b_s70_1104728 | 7.9679 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-4b-s60pct` | alps_rollout_mix_4b_s60_1104727 | 5.8109 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-4b-s50pct` | alps_rollout_mix_4b_s50_1104726 | 5.2442 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-1.7b-s70pct` | alps_rollout_mix_1.7b_s70_1104725 | 11.7479 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-1.7b-s60pct` | alps_rollout_mix_1.7b_s60_1104724 | 7.4632 |
| 2026-10-08 | `alps-rollout2048mix-qwen3-1.7b-s50pct` | alps_rollout_mix_1.7b_s50_1104723 | 6.2674 |
| 2026-10-06 | `alps-s70pct_20261006_162912` | alps_1.7b_1087007_debug | 11.8548 |
| 2026-09-28 | `alps-s50pct_20260929_002917` | alps_1.7b_24_1028837_debug | 7.4954 |
| 2026-09-25 | `alps-s80pct_20260925_220218` | alps_4b_selfgen_v3_1003078_debug | 8.4441 |
| 2026-09-11 | `alps-s40pct_20260911_120837` | alps_4b_907465_debug | 4.9835 |
| 2026-09-11 | `alps-s30pct_20260911_120849` | alps_4b_907464_debug | 4.8758 |
| 2026-09-02 | `alps-s80pct_20260903_082836` | alps_8b_839779_debug | 12.6907 |
| 2026-09-02 | `alps-s80pct_20260902_224218` | alps_4b_837518_debug | 16.7340 |
| 2026-08-27 | `alps-s70pct_20260827_111542` | alps_4b_selfgen_v3_794884_debug | 4.2795 |
| 2026-08-27 | `alps-s60pct_20260827_105937` | alps_4b_selfgen_v3_794883_debug | 3.0604 |
| 2026-08-27 | `alps-s50pct_20260827_094710` | alps_4b_selfgen_v3_794882_debug | 2.7702 |
| 2026-08-13 | `alps-s70pct_20260813_204659` | alps_8b_rtx6000ada_719635_debug | 7.1088 |
| 2026-08-13 | `alps-s60pct_20260813_204611` | alps_8b_rtx6000ada_719634_debug | 5.2599 |
| 2026-08-13 | `alps-s50pct_20260813_205938` | alps_8b_24_rtx6000ada_719641_debug | 5.4912 |
| 2026-08-13 | `alps-s50pct_20260813_204618` | alps_4b_24_rtx3090_719640_debug | 6.3242 |
| 2026-08-12 | `alps-s50pct_20260813_013858` | alps_1.7b_24_719639_debug | 7.5054 |
| 2026-08-11 | `alps-s70pct_20260811_112036` | alps_4b_710129_debug | 8.1554 |
| 2026-08-11 | `alps-s60pct_20260811_113141` | alps_4b_710128_debug | 5.7834 |
| 2026-08-11 | `alps-s50pct_20260811_111647` | alps_4b_710127_debug | 5.2300 |
| 2026-08-10 | `alps-s60pct_20260811_021742` | alps_1.7b_709621_debug | 7.2016 |
| 2026-08-10 | `alps-s50pct_20260811_015553` | alps_1.7b_709620_debug | 6.1017 |
| 2026-08-08 | `gmp-s70pct-lr1e-4_20260808_131057` | logs |  |
| 2026-08-08 | `gmp-s70pct-lr1e-4_20260808_113308` | logs |  |
| 2026-08-08 | `gmp-s60pct-lr1e-4_20260808_121032` | logs |  |
| 2026-08-08 | `gmp-s60pct-lr1e-4_20260808_110900` | logs |  |
| 2026-08-08 | `gmp-s50pct-lr1e-4_20260808_143612` | logs |  |
| 2026-08-08 | `gmp-s50pct-lr1e-4_20260808_110050` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s70pct-lr1e-4_20260808_142744` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s70pct-lr1e-4_20260808_101822` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s60pct-lr1e-4_20260808_135612` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s60pct-lr1e-4_20260808_100022` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s50pct-lr1e-4_20260808_125458` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s50pct-lr1e-4_20260808_123031` | logs |  |
| 2026-08-08 | `gmp-kd5e-1-s50pct-lr1e-4_20260808_110942` | logs |  |
| 2026-08-07 | `gmp-kd3e-1-s70pct-lr1e-4_20260808_063213` | logs |  |
| 2026-08-07 | `gmp-kd3e-1-s60pct-lr1e-4_20260808_052714` | logs |  |
| 2026-08-07 | `gmp-kd3e-1-s50pct-lr1e-4_20260808_045211` | logs |  |
| 2026-08-07 | `gmp-kd3e-1-s50pct-lr1e-4_20260808_045134` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr5e-6_20260811_234211` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr5e-5_20260817_163447` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr5e-5_20260814_040006` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr2e-4_20260814_035218` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr1e-5_20260811_232839` | logs |  |
|  | `gmp-kd3e-1-s70pct-lr1e-4_20260811_225808` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr5e-6_20260811_222627` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr5e-5_20260817_053240` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr5e-5_20260812_230422` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr5e-5_20260811_214248` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr2e-4_20260813_072028` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr1e-5_20260813_030730` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr1e-5_20260811_214804` | logs |  |
|  | `gmp-kd3e-1-s60pct-lr1e-4_20260812_212408` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr5e-6_20260811_193103` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr5e-5_20260818_075829` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr5e-5_20260815_213950` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr5e-5_20260812_123156` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr5e-5_20260811_191907` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr1e-5_20260811_193324` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr1e-4_20260817_162507` | logs |  |
|  | `gmp-kd3e-1-s50pct-lr1e-4_20260812_085601` | logs |  |
|  | `alps-s70pct_20260811_015229` | alps_1.7b_709622_debug | 11.5717 |

## 3. HF에 없는 최종 모델 (로컬 전용)

| 로컬 경로 (elsa/models) | 내용 | 비고 |
|---|---|---|
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261008_042227_p1065587` | 4B 3bit 양자화 SCOUT, roll2048, RTN 그리드 (잡 1093104) | 학습 완료 후 zero-shot 평가에서 Half≠float 크래시로 업로드 안 됨. long 평가 wandb aeooconf |
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261008_032809_p1675546` | 4B 2bit 양자화 SCOUT, roll2048 (잡 1096416, 1792 재개) | 같은 이유로 업로드 안 됨. 평가 wandb io88ox1o 등 |

## 4. 출처 미확인 리포 (342개)

wandb 학습 런·ALPS 로그와 자동으로 연결되지 않은 리포 (주로 2026-07 이전, SparseGPT/ELSA/양자화 베이스라인 등). 사용처 확인 전 삭제 금지.

- 2026-10-10 `olmo3-7b-think-scout-s50-klb0.01-roll2048`
- 2026-10-10 `olmo3-7b-think-alps-ntpkdopd-s50-lr5e-5-roll2048`
- 2026-10-10 `olmo3-7b-think-scout-s60-klb0.01-roll2048`
- 2026-10-10 `olmo3-7b-think-alps-oneshot-s60`
- 2026-10-09 `olmo3-7b-think-scout-s70-klb0.03-roll2048`
- 2026-10-09 `olmo3-7b-think-alps-ntpkdopd-s70-lr1e-4-roll2048`
- 2026-10-09 `olmo3-7b-think-alps-oneshot-s50`
- 2026-10-08 `sparsellm-s0pct_20261008_123705`
- 2026-10-08 `sparsellm-s0pct_20261008_123842`
- 2026-10-08 `sparsellm-s0pct_20261008_131030`
- 2026-10-08 `sparsellm-s0pct_20261008_133444`
- 2026-10-08 `olmo3-7b-think-alps-oneshot-s70`
- 2026-10-06 `alps-s70pct_20261006_132528`
- 2026-10-05 `sparsellm-s0pct_20261005_185921`
- 2026-10-05 `sparsellm-s0pct_20261005_185922`
- 2026-09-23 `gmp-4b-s80pct-frozenpool-20260922-062919`
- 2026-09-23 `gmp-8b-s70pct-jump-klb003-20260923-052514`
- 2026-09-23 `gmp-4b-s80pct-cubicnopgd-20260922-181742`
- 2026-09-23 `gmp-4b-s70pct-jump-noopd-20260923-202338`
- 2026-09-23 `alps-selfgenv3-qwen3-8b-s50pct`
- 2026-09-23 `alps-selfgenv3-qwen3-8b-s60pct`
- 2026-09-23 `alps-selfgenv3-qwen3-8b-s70pct`
- 2026-09-23 `sgpt-selfgenv3-qwen3-8b-s50pct`
- 2026-09-23 `sgpt-selfgenv3-qwen3-8b-s60pct`
- 2026-09-23 `sgpt-selfgenv3-qwen3-8b-s70pct`
- 2026-09-23 `gmp-8b-s70pct-jump-noopd-20260924-043024`
- 2026-09-23 `gmp-8b-s70pct-alpssft-noopd-20260924-073338`
- 2026-09-23 `qwen3-1.7b-tr-gmp-ntpkdopkd-s70pct-lr1e-04-mi16-nostrip8192`
- 2026-09-23 `qwen3-1.7b-tr-gmp-ntpkdopkd-s70pct-lr1e-04-mi32-nostrip8192`
- 2026-09-21 `gmp-4b-s70pct-lr0.0001-onpol-lmda0.33-20260921-212708-p265824`
- 2026-09-21 `gmp-4b-s70pct-lr0.0001-onpol-lmda0.33-20260922-002405-p291199`
- 2026-09-21 `gmp-4b-s70pct-lr0.0001-onpol-lmda0.33-20260922-023308-p310907`
- 2026-09-20 `sparsegpt-s80pct_20260920_235157`
- 2026-09-20 `sparsegpt-s60pct_20260921_012025`
- 2026-09-20 `sparsegpt-s50pct_20260921_012025`
- 2026-09-20 `sparsegpt-s80pct_20260921_052911`
- 2026-09-18 `gmp-4b-s70pct-lr1e-4_20260919_041114`
- 2026-09-16 `alps-qwen3-4b-s80pct`
- 2026-09-16 `alps-qwen3-8b-s80pct`
- 2026-09-14 `alps4b-s70-2term-step000512`
- 2026-09-14 `alps4b-s70-2term-step001024`
- 2026-09-14 `alps4b-s70-2term-step001536`
- 2026-09-14 `alps4b-s70-3term-step000512`
- 2026-09-14 `alps4b-s70-3term-step001024`
- 2026-09-14 `alps4b-s70-3term-step001536`
- 2026-09-11 `sparsegpt-s40pct_20260911_103940`
- 2026-09-11 `sparsegpt-s30pct_20260911_103959`
- 2026-09-11 `sparsegpt-s70pct_20260911_113325`
- 2026-09-08 `qwen3-8b-alps-2to4`
- 2026-09-08 `qwen3-8b-alps-s60pct`
- 2026-09-08 `qwen3-8b-alps-s70pct`
- 2026-09-08 `backup-gmp-s50pct-lr5e-05-onpol-lmda0p33-20260827-133259`
- 2026-09-08 `backup-gmp-s50pct-lr5e-05-onpol-lmda0p33-20260828-125050`
- 2026-09-08 `backup-gmp-s50pct-lr5e-05-onpol-lmda0p33-20260828-125227`
- 2026-09-08 `backup-gmp-s50pct-lr5e-05-onpol-lmda0p33-20260901-105142`
- 2026-09-08 `backup-gmp-s60pct-lr5e-05-onpol-lmda0p33-20260820-033814`
- 2026-09-08 `backup-gmp-s70pct-lr0p0001-onpol-lmda0p33-20260827-230324`
- 2026-09-08 `backup-gmp-s70pct-lr0p0001-onpol-lmda0p33-20260828-073925`
- 2026-09-08 `backup-gmp-s70pct-lr0p0001-onpol-lmda0p33-20260906-105649`
- 2026-09-06 `sparsegpt-s60pct_20260906_231228`
- 2026-09-06 `sparsegpt-s50pct_20260906_231231`
- 2026-09-06 `sparsegpt-s70pct_20260906_231301`
- 2026-09-06 `sparsegpt-s70pct_20260906_231631`
- 2026-09-06 `sparsegpt-s50pct_20260906_232827`
- 2026-09-06 `sparsegpt-s70pct_20260906_233030`
- 2026-09-06 `sparsegpt-s60pct_20260906_233523`
- 2026-09-04 `ipo-alpssft-s70-ultrafeedback-fullft-lr5e-6`
- 2026-09-04 `ipo-alpssft-s70-ultrafeedback-fullft-lr1e-5`
- 2026-09-04 `grpo-alpssft-s70-overthinking`
- 2026-09-04 `grpo-alpssft-s70-ablation-noLen`
- 2026-09-03 `sparsellm-s80pct_20260903_113201`
- 2026-09-02 `sparsellm-s80pct_20260903_020850`
- 2026-09-01 `sparsellm-s50pct_20260901_090527`
- 2026-08-31 `sparsellm-s50pct_20260831_133750`
- 2026-08-31 `sparsellm-s50pct_20260831_162611`
- 2026-08-31 `sparsellm-s50pct_20260831_204119`
- 2026-08-31 `sparsellm-s70pct_20260831_204148`
- 2026-08-31 `sparsellm-s60pct_20260831_204149`
- 2026-08-31 `sparsellm-s50pct_20260831_204946`
- 2026-08-31 `sparsellm-s60pct_20260831_210736`
- 2026-08-31 `sparsellm-s70pct_20260831_211530`
- 2026-08-31 `sparsellm-s50pct_20260831_220245`
- 2026-08-31 `sparsellm-s50pct_20260901_050129`
- 2026-08-31 `sparsellm-s60pct_20260901_062412`
- 2026-08-31 `sparsellm-s70pct_20260901_073704`
- 2026-08-20 `sparsegpt-qwen3-4b-24`
- 2026-08-20 `sparsegpt-qwen3-8b-s50pct`
- 2026-08-20 `sparsegpt-qwen3-8b-s60pct`
- 2026-08-20 `sparsegpt-qwen3-8b-s70pct`
- 2026-08-20 `sparsegpt-qwen3-8b-24`
- 2026-08-19 `qwen3-8b-alps-s50pct`
- 2026-08-12 `sparsegpt-qwen3-1.7b-s50pct`
- 2026-08-12 `sparsegpt-qwen3-4b-s50pct`
- 2026-08-12 `sparsegpt-qwen3-4b-s70pct`
- 2026-08-12 `sparsegpt-qwen3-1.7b-s60pct`
- 2026-08-12 `sparsegpt-qwen3-1.7b-s70pct`
- 2026-08-12 `sparsegpt-qwen3-4b-s60pct`
- 2026-08-12 `alps-s50pct_20260813_025953`
- 2026-08-12 `alps-s70pct_20260813_063546`
- 2026-08-12 `alps-s50pct_20260813_065449`
- 2026-08-10 `qwen3-1.7b-tr-gmp-ntpkdopkd-s50pct-lr5e-05-mi32-nostrip8192`
- 2026-08-10 `qwen3-1.7b-tr-gmp-ntpkdopkd-s60pct-lr1e-04-mi32-nostrip8192`
- 2026-08-08 `gmp-s50pct-lr3e-3_20260808_114716`
- 2026-08-08 `gmp-s50pct-lr1e-3_20260808_115131`
- 2026-08-08 `gmp-kd3e-1-s70pct-lr1e-4_20260808_162458`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260807_171220`
- 2026-08-07 `gmp-kd5e-1-s50pct-lr1e-4_20260807_180625`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260807_182827`
- 2026-08-07 `gmp-kd5e-1-s70pct-lr1e-4_20260807_184606`
- 2026-08-07 `gmp-kd5e-1-s50pct-lr1e-4_20260807_190643`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260807_191912`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260807_210247`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260807_211520`
- 2026-08-07 `gmp-kd5e-1-s70pct-lr1e-4_20260808_021102`
- 2026-08-07 `gmp-kd5e-1-s70pct-lr1e-4_20260808_035846`
- 2026-08-07 `gmp-kd3e-1-s50pct-lr1e-4_20260808_055921`
- 2026-08-06 `elsa-offpolicy-kd-s70pct-lr1e-4-lmda1e-3_20260806_090614`
- 2026-08-06 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260806_091235`
- 2026-08-06 `alps-sparse-sft-qwen3-4b-s70pct`
- 2026-08-06 `elsa-offpolicy-kd-s60pct-lr1e-4-lmda1e-3_20260806_213447`
- 2026-08-06 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260806_221109`
- 2026-08-06 `elsa-offpolicy-kd-s70pct-lr1e-4-lmda1e-3_20260806_233920`
- 2026-08-06 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260807_084818`
- 2026-08-05 `gmp-s60pct-lr1e-4_20260805_091422`
- 2026-08-05 `gmp-s70pct-lr1e-4_20260805_091646`
- 2026-08-05 `gmp-s70pct-lr1e-4_20260805_143819`
- 2026-08-05 `gmp-kd5e-1-s50pct-lr1e-4_20260805_154837`
- 2026-08-05 `gmp-kd5e-1-s60pct-lr1e-4_20260805_190033`
- 2026-08-05 `gmp-kd5e-1-s70pct-lr1e-4_20260805_223842`
- 2026-08-05 `gmp-kd5e-1-s60pct-lr1e-4_20260806_015108`
- 2026-08-05 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260806_045245`
- 2026-08-05 `elsa-offpolicy-kd-s60pct-lr1e-4-lmda1e-3_20260806_074612`
- 2026-08-04 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_091450`
- 2026-08-04 `elsa-ntp-cot-s50pct-lr1e-4-lmda5e-4_20260804_095627`
- 2026-08-04 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_175804`
- 2026-08-04 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_184323`
- 2026-08-04 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260804_203618`
- 2026-08-04 `gmp-kd5e-1-s50pct-lr1e-4_20260805_021915`
- 2026-08-04 `gmp-kd5e-1-s60pct-lr1e-4_20260805_023634`
- 2026-08-04 `gmp-kd5e-1-s70pct-lr1e-4_20260805_030323`
- 2026-08-04 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260805_030546`
- 2026-08-04 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260805_042034`
- 2026-08-04 `elsa-offpolicy-kd-s50pct-lr1e-4-lmda1e-3_20260805_042628`
- 2026-08-04 `gmp-kd5e-1-s50pct-lr1e-4_20260805_070639`
- 2026-08-04 `gmp-s50pct-lr1e-4_20260805_084102`
- 2026-08-03 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260803_182759`
- 2026-08-03 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_082443`
- 2026-08-03 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_082603`
- 2026-08-03 `elsa-ntp-cot-s50pct-lr1e-4-lmda1e-3_20260804_085652`
- 2026-08-02 `alps-s70pct_20260802_101527`
- 2026-08-02 `alps-s60pct_20260802_102302`
- 2026-08-02 `qwen3-1.7b-sgpt-s50pct-ot80fw20`
- 2026-08-02 `qwen3-1.7b-sgpt-s60pct-ot80fw20`
- 2026-08-02 `qwen3-1.7b-sgpt-s70pct-ot80fw20`
- 2026-08-02 `qwen3-1.7b-sgpt-2to4-ot80fw20`
- 2026-08-02 `qwen3-4b-sgpt-s50pct-ot80fw20`
- 2026-08-02 `qwen3-4b-sgpt-2to4-ot80fw20`
- 2026-08-02 `qwen3-8b-sgpt-2to4-ot80fw20`
- 2026-08-02 `alps-s50pct_20260802_110310`
- 2026-08-02 `sparsellm-s70pct_20260802_115255`
- 2026-08-02 `sparsellm-s60pct_20260802_120045`
- 2026-08-02 `sparsellm-s50pct_20260802_120337`
- 2026-08-02 `sparsellm-s70pct_20260802_120402`
- 2026-08-02 `sparsellm-s50pct_20260802_120853`
- 2026-08-02 `sparsellm-s50pct_20260802_122059`
- 2026-08-02 `sparsellm-s60pct_20260802_124745`
- 2026-08-02 `sparsellm-s50pct_20260802_125438`
- 2026-08-02 `sparsellm-s60pct_20260802_125804`
- 2026-08-02 `sparsellm-s50pct_20260802_130036`
- 2026-08-02 `sparsellm-s70pct_20260802_134428`
- 2026-08-02 `qwen3-4b-alps-2to4-ot80fw20`
- 2026-08-02 `qwen3-4b-sparsellm-s70pct-ot80fw20`
- 2026-08-02 `sparsellm-s50pct_20260803_004532`
- 2026-08-01 `gmp-kd1e0-s50pct-lr1e-4_20260801_235019`
- 2026-08-01 `smoketest-qwen3-0.6b-sgpt-s50`
- 2026-08-01 `alps-s50pct_20260802_054359`
- 2026-08-01 `alps-s70pct_20260802_054634`
- 2026-08-01 `alps-s50pct_20260802_055049`
- 2026-08-01 `alps-s70pct_20260802_055131`
- 2026-08-01 `alps-s60pct_20260802_055152`
- 2026-08-01 `alps-s50pct_20260802_055618`
- 2026-08-01 `alps-s50pct_20260802_071835`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260731_093118`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260731_100053`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda1e-3_20260731_234207`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260731_234336`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-4_20260731_235212`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260801_000827`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260801_012136`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda1e-3_20260801_013939`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-4_20260801_014106`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda1e-3_20260801_074258`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-4_20260801_075234`
- 2026-07-31 `elsa-ntp-cot-s70pct-lr1e-4-lmda5e-3_20260801_082004`
- 2026-07-30 `qwen3-4b-gmp-tr-dwup-s70pct`
- 2026-07-30 `gmp-kd1e0-s70pct-lr1e-4_20260730_164216`
- 2026-07-30 `gmp-kd1e0-s70pct-lr1e-4_20260730_180740`
- 2026-07-25 `safe-qwen3-1.7b-s60pct`
- 2026-07-25 `safe-qwen3-1.7b-s70pct`
- 2026-07-24 `qwen3-4b-gmp-ntp-kd-opkd-s50pct`
- 2026-07-24 `qwen3-4b-gmp-ntp-kd-opkd-s60pct`
- 2026-07-24 `qwen3-4b-gmp-ntp-kd-opkd-s70pct`
- 2026-07-24 `qwen3-4b-gmp-lasso-s50pct`
- 2026-07-24 `qwen3-4b-gmp-lasso-s60pct`
- 2026-07-24 `qwen3-4b-gmp-lasso-s70pct`
- 2026-07-24 `qwen3-4b-gmp-tr-dwup-s50pct`
- 2026-07-24 `qwen3-4b-gmp-tr-dwup-lasso-s50pct`
- 2026-07-24 `qwen3-4b-gmp-tr-dwup-lasso-s60pct-v1`
- 2026-07-24 `qwen3-4b-gmp-tr-dwup-lasso-s60pct-v2`
- 2026-07-24 `elsa-ntp-admm-qwen3-4b-s50pct`
- 2026-07-24 `elsa-ntp-admm-qwen3-4b-s60pct`
- 2026-07-24 `elsa-ntp-admm-qwen3-4b-s70pct`
- 2026-07-24 `alps-qwen3-4b-s50pct`
- 2026-07-24 `alps-qwen3-4b-s60pct`
- 2026-07-24 `alps-qwen3-4b-s70pct`
- 2026-07-24 `sparsellm-qwen3-4b-s50pct`
- 2026-07-24 `sparsellm-qwen3-4b-s60pct`
- 2026-07-24 `sparsellm-qwen3-4b-s70pct`
- 2026-07-24 `qwen3-8b-gmp-ntp-kd-s50pct`
- 2026-07-24 `safe-qwen3-1.7b-s50pct`
- 2026-07-22 `qwen3-8b-gmp-ntp-kd-s60pct`
- 2026-07-20 `elsa-ntp-admm-qwen3-1.7b-s50pct`
- 2026-07-20 `elsa-ntp-admm-qwen3-1.7b-s60pct`
- 2026-07-20 `elsa-ntp-admm-qwen3-1.7b-s70pct`
- 2026-07-20 `sparsellm-qwen3-1.7b-s50pct`
- 2026-07-20 `sparsellm-qwen3-1.7b-s60pct`
- 2026-07-20 `sparsellm-qwen3-1.7b-s70pct`
- 2026-07-20 `alps-qwen3-1.7b-s50pct`
- 2026-07-20 `alps-qwen3-1.7b-s60pct`
- 2026-07-20 `alps-qwen3-1.7b-s70pct`
- 2026-07-19 `qwen3-8b-sgpt-s50pct-ot80fw20`
- 2026-07-19 `qwen3-8b-sgpt-s60pct-ot80fw20`
- 2026-07-19 `qwen3-8b-sgpt-s70pct-ot80fw20`
- 2026-07-18 `qwen3-4b-sgpt-s60pct-ot80fw20`
- 2026-07-18 `qwen3-4b-sgpt-s70pct-ot80fw20`
- 2026-07-18 `qwen3-4b-gmp-tr-ntpkd-s50pct-kl001`
- 2026-07-18 `qwen3-4b-gmp-tr-ntpkd-s50pct-kl002`
- 2026-07-18 `qwen3-4b-gmp-tr-ntpkd-s60pct-kl001`
- 2026-07-18 `qwen3-4b-gmp-tr-ntpkd-s60pct-kl002`
- 2026-07-17 `Qwen3-4B-SparseGPT-S50`
- 2026-07-17 `Qwen3-4B-SparseGPT-S60`
- 2026-07-17 `Qwen3-4B-SparseGPT-S70`
- 2026-07-17 `Qwen3-4B-GMP-NTP-KD-S50`
- 2026-07-17 `Qwen3-4B-GMP-NTP-KD-S60`
- 2026-07-14 `gmp-kd1e0-s50pct-lr1e-4_20260714_111310`
- 2026-07-14 `gmp-kd1e0-s70pct-lr1e-4_20260714_112752`
- 2026-07-14 `gmp-kd1e0-s60pct-lr1e-4_20260714_113414`
- 2026-07-14 `gmp-kd1e0-s50pct-lr1e-4_20260714_113553`
- 2026-07-14 `gmp-kd1e0-s50pct-lr1e-4_20260715_053051`
- 2026-07-14 `gmp-kd1e0-s70pct-lr1e-4_20260715_054019`
- 2026-07-14 `gmp-kd1e0-s60pct-lr1e-4_20260715_055304`
- 2026-07-13 `gmp-kd1e0-s70pct-lr1e-4_20260713_153610`
- 2026-07-13 `gmp-kd1e0-s70pct-lr1e-4_20260713_154053`
- 2026-07-13 `gmp-kd1e0-s60pct-lr1e-4_20260713_154219`
- 2026-07-13 `gmp-kd1e0-s60pct-lr1e-4_20260714_045718`
- 2026-07-13 `gmp-kd1e0-s50pct-lr1e-4_20260714_055623`
- 2026-07-13 `gmp-kd1e0-s70pct-lr1e-4_20260714_061104`
- 2026-07-13 `gmp-kd1e0-s70pct-lr1e-4_20260714_062056`
- 2026-07-13 `gmp-kd1e0-s60pct-lr1e-4_20260714_064907`
- 2026-07-13 `gmp-kd1e0-s70pct-lr1e-4_20260714_083512`
- 2026-07-12 `gmp-kd1e0-s70pct-lr1e-4_20260712_092549`
- 2026-07-12 `gmp-kd1e0-s50pct-lr1e-4_20260712_100353`
- 2026-07-12 `gmp-kd1e0-s70pct-lr1e-4_20260712_101701`
- 2026-07-12 `gmp-kd1e0-s60pct-lr1e-4_20260712_124610`
- 2026-07-12 `gmp-kd1e0-s70pct-lr1e-4_20260712_144947`
- 2026-07-12 `gmp-kd1e0-s50pct-lr1e-4_20260712_183815`
- 2026-07-12 `gmp-kd1e0-s60pct-lr1e-4_20260712_193204`
- 2026-07-12 `gmp-kd1e0-s50pct-lr1e-4_20260712_203917`
- 2026-07-12 `gmp-kd1e0-s60pct-lr1e-4_20260712_221513`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260711_092116`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260711_094953`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260711_103056`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260711_132542`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260711_133212`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260711_135950`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_175036`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260711_181703`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_190919`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_195221`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_204836`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_223422`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260711_234224`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260712_023504`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260712_062009`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260712_072156`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260712_073213`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260712_073935`
- 2026-07-11 `gmp-kd1e0-s70pct-lr1e-4_20260712_080022`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260712_082547`
- 2026-07-11 `gmp-kd1e0-s50pct-lr1e-4_20260712_082548`
- 2026-07-11 `gmp-kd1e0-s60pct-lr1e-4_20260712_084045`
- 2026-07-10 `gmp-kd1e0-s50pct-lr1e-4_20260710_112207`
- 2026-07-10 `gmp-kd1e0-s50pct-lr1e-4_20260710_144853`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_165553`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_165658`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_170350`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_170933`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_171018`
- 2026-07-10 `gmp-kd1e0-s50pct-lr1e-4_20260710_171528`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_174108`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_174446`
- 2026-07-10 `gmp-kd1e0-s50pct-lr1e-4_20260710_184120`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_203741`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_210736`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260710_211955`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_214929`
- 2026-07-10 `gmp-kd1e0-s70pct-lr1e-4_20260710_220516`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260711_071148`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260711_081102`
- 2026-07-10 `gmp-kd1e0-s60pct-lr1e-4_20260711_085206`
- 2026-07-09 `gmp-kd1e0-s50pct-lr1e-4_20260710_064104`
- 2026-07-09 `gmp-kd1e0-s50pct-lr1e-4_20260710_064323`
- 2026-07-09 `gmp-kd1e0-s50pct-lr1e-4_20260710_071332`
- 2026-07-08 `gmp-kd1e0-s50pct-lr1e-4_20260708_120030`
- 2026-07-08 `gmp-kd1e0-s50pct-lr1e-4_20260708_120835`
- 2026-07-08 `gmp-kd1e0-s60pct-lr1e-4_20260708_122951`
- 2026-07-08 `gmp-kd1e0-s50pct-lr1e-4_20260708_123251`
- 2026-07-08 `gmp-kd1e0-s70pct-lr1e-4_20260708_123408`
- 2026-07-08 `gmp-kd1e0-s70pct-lr1e-4_20260708_130808`
- 2026-07-08 `gmp-kd1e0-s60pct-lr1e-4_20260708_131025`
- 2026-07-08 `gmp-kd1e0-s70pct-lr1e-4_20260708_150124`
- 2026-07-08 `gmp-kd1e0-s60pct-lr1e-4_20260708_150506`
- 2026-07-07 `gmp-kd1e0-s50pct-lr1e-4_20260707_114011`
- 2026-07-07 `gmp-kd1e0-s50pct-lr1e-4_20260707_114525`
- 2026-07-07 `gmp-kd1e0-s60pct-lr1e-4_20260707_115022`
- 2026-07-07 `gmp-kd1e0-s60pct-lr1e-4_20260707_120906`
- 2026-07-07 `gmp-kd1e0-s50pct-lr1e-4_20260707_121943`
- 2026-07-07 `gmp-kd1e0-s70pct-lr1e-4_20260707_122404`
- 2026-07-07 `gmp-kd1e0-s60pct-lr1e-4_20260707_124907`
- 2026-07-07 `gmp-kd1e0-s70pct-lr1e-4_20260707_130931`
- 2026-07-07 `gmp-kd1e0-s70pct-lr1e-4_20260707_145811`
- 2026-07-06 `gmp-kd1e0-s50pct-lr1e-4_20260707_010755`
- 2026-07-06 `gmp-kd1e0-s50pct-lr1e-4_20260707_011305`
- 2026-07-06 `gmp-kd1e0-s60pct-lr1e-4_20260707_012832`
- 2026-07-06 `gmp-kd1e0-s50pct-lr1e-4_20260707_013118`
- 2026-07-06 `gmp-kd1e0-s70pct-lr1e-4_20260707_013419`
- 2026-07-06 `gmp-kd1e0-s70pct-lr1e-4_20260707_020027`
- 2026-07-06 `gmp-kd1e0-s60pct-lr1e-4_20260707_020505`
- 2026-07-06 `gmp-kd1e0-s60pct-lr1e-4_20260707_031928`
- 2026-07-05 `deepseek-r1-distill-qwen-1.5b-sgpt-s30pct-ot80fw20`
- 2026-07-05 `deepseek-r1-distill-qwen-1.5b-sgpt-s40pct-ot80fw20`
- 2026-07-05 `deepseek-r1-distill-qwen-1.5b-sgpt-s50pct-ot80fw20`
- 2026-07-05 `deepseek-r1-distill-qwen-1.5b-sgpt-s60pct-ot80fw20`
- 2026-07-05 `deepseek-r1-distill-qwen-1.5b-sgpt-s70pct-ot80fw20`
- 2026-07-05 `gmp-kd1e0-s50pct-lr7e-5_20260705_130521`
- 2026-07-05 `gmp-kd1e0-s50pct-lr1e-4_20260705_130752`
- 2026-07-05 `gmp-kd1e0-s60pct-lr2e-4_20260705_131031`
- 2026-07-05 `gmp-kd1e0-s60pct-lr1e-4_20260705_131614`
- 2026-07-05 `gmp-kd1e0-s60pct-lr7e-5_20260705_131722`
- 2026-07-05 `gmp-kd1e0-s70pct-lr1e-4_20260705_132250`
- 2026-07-05 `gmp-kd1e0-s70pct-lr2e-4_20260705_132637`
- 2026-07-05 `gmp-kd1e0-s50pct-lr2e-4_20260705_151224`
- 2026-07-05 `gmp-kd1e0-s70pct-lr7e-5_20260705_154717`
- 2026-07-05 `gmp-kd1e0-s50pct-lr1e-4_20260706_020934`
- 2026-07-05 `gmp-kd1e0-s60pct-lr1e-4_20260706_022219`
- 2026-07-05 `gmp-kd1e0-s70pct-lr1e-4_20260706_023531`
- 2026-07-01 `gmp-s70pct-lr1e-4_20260701_121916`
- 2026-07-01 `gmp-kd1e0-s60pct-lr1e-4_20260701_122400`
- 2026-07-01 `gmp-kd1e0-s50pct-lr1e-4_20260701_122423`
- 2026-07-01 `gmp-s60pct-lr1e-4_20260701_122545`
- 2026-07-01 `gmp-s50pct-lr1e-4_20260701_123528`
- 2026-07-01 `gmp-kd1e0-s70pct-lr1e-4_20260701_124145`
- 2026-07-01 `gmp-kd1e0-s70pct-lr1e-4_20260701_124550`
- 2026-07-01 `gmp-kd1e0-s60pct-lr1e-4_20260701_125332`
- 2026-07-01 `gmp-kd1e0-s50pct-lr1e-4_20260701_130140`

## 5. 로컬 사본을 지운 모델 (HF에만 있음)

정리일: 2026-10-12. 디스크 쿼터 확보를 위해, HF 리포의 safetensors 총 크기가 로컬과 정확히 같고 마지막 수정이 3일 넘은 로컬 폴더만 삭제했습니다. ALPS 기준 모델(qwen3_*_alps_s*pct, *_s24)과 체크포인트 폴더는 대상에서 뺐습니다. 필요하면 `huggingface-cli download <리포> --local-dir elsa/models/<로컬 폴더>`로 복원합니다. 각 리포의 방법·설정·sync 상태는 위 1·2절에서 리포 이름으로 찾으면 됩니다.

| 로컬 폴더 (elsa/models) | HF 리포 | 크기 GB | 런 (wandb 이름 또는 로그 헤더) | 로그 |
|---|---|---|---|---|
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20260929_220212_p1287040` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_033246` | 3.4 | pending | alps24_scout_wanda_1.7b_1030947.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20260929_220655_p3299667` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_060834` | 3.4 | pending | alps24_scout_fisher_1.7b_1030946.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20260929_232731_p1089940` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20260930_072819` | 3.4 | pending | alps24_training_1.7b_1031704.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261001_030945_p3878065` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_111255` | 3.4 | pending | alps24scout_ste_d1e3_klb002_n24_1.7b_1038486.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261001_031649_p3573175` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_095927` | 3.4 | pending | alps24scout_ste_d1e3_klb001_n24_1.7b_1038485.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261001_031741_p1322283` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_111538` | 3.4 | pending | alps24scout_ste_d0_klb002_n24_1.7b_1038488.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261001_033100_p3575874` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261001_102449` | 3.4 | pending | alps24scout_ste_d0_klb001_n24_1.7b_1038487.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261002_073442_p1580146` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261002_153016` | 3.4 | pending | alps24scout_tgate_m0_c8_n24_1.7b_1048667.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261002_074429_p2173971` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261002_160223` | 3.4 | pending | alps24scout_tgate_m0_c16_n24_1.7b_1048669.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261005_230331_p2524277` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_071136` | 3.4 | pending | alps24scout_comp_klb002_n24_1.7b_1078267.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261005_232718_p1578318` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_065934` | 3.4 | pending | alps24scout_comp_klb001_n24_1.7b_1078268.out |
| `gmp_1.7b_n24_s50pct_lr0.0001_onpol_lmda0.33_20261005_234019_p2546564` | `cosmos1030/gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261006_055538` | 3.4 | pending | alps24scout_comp_tgate_n24_1.7b_1078266.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_065604_p3544763` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_105401` | 3.4 | pending | quant_1.7b_3bit_notr_1080791.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_070128_p2305192` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_131408` | 3.4 | pending | quant_1.7b_3bit_01_1080790.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_072146_p1265318` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_160442` | 3.4 | pending | quant_1.7b_2bit_01_1080786.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_072319_p1265307` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_174408` | 3.4 | pending | quant_1.7b_2bit_002_1080785.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_072748_p1630678` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_135840` | 3.4 | pending | quant_1.7b_2bit_notr_1080787.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_073707_p2447102` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_191852` | 3.4 | pending | quant_1.7b_2bit_0005_1080784.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_073724_p1630679` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_122929` | 3.4 | pending | quant_1.7b_3bit_0005_1080788.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_081629_p3058928` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261006_151503` | 3.4 | pending | quant_1.7b_3bit_notr_fixedgrid_1081111.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261006_202928_p499895` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_025825` | 3.4 | pending | quant_1.7b_3bit_002_1084751.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261007_084716_p3307628` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_163721` | 3.4 | pending | quant_1.7b_2bit_notr_fixedgrid_1088171.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261007_192701_p3371813` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261007_232355` | 3.4 | pending | gptq_train_1.7b_3bit_1090727.out |
| `gmp_1.7b_s0pct_lr0.0001_onpol_lmda0.33_20261007_193305_p3371812` | `cosmos1030/gmp-kd3e-1-1.7b-s0pct-lr1e-4_20261008_035937` | 3.4 | pending | gptq_train_1.7b_2bit_1090728.out |
| `gmp_1.7b_s50pct_lr0.0001_onpol_lmda0.33_20261007_223216_p3373423` | `cosmos1030/gmp-kd3e-1-1.7b-s50pct-lr1e-4_20261008_035636` | 3.4 | pending | alpstrain_1.7b_s50_lr1e-4_roll2048_1090751.out |
| `gmp_1.7b_s50pct_lr5e-05_onpol_lmda0.33_20261007_115015_p2669171` | `cosmos1030/gmp-kd3e-1-1.7b-s50pct-lr5e-5_20261007_150922` | 3.4 | pending | ours_1.7b_s50_klb002_pgdi8_lr5e-5_roll2048_1088189.out |
| `gmp_1.7b_s60pct_lr0.0001_onpol_lmda0.33_20261007_215158_p2924548` | `cosmos1030/gmp-kd3e-1-1.7b-s60pct-lr1e-4_20261008_043037` | 3.4 | pending | alpstrain_1.7b_s60_lr1e-4_roll2048_1090752.out |
| `gmp_1.7b_s60pct_lr5e-05_onpol_lmda0.33_20261007_114751_p85593` | `cosmos1030/gmp-kd3e-1-1.7b-s60pct-lr5e-5_20261007_154206` | 3.4 | pending | ours_1.7b_s60_klb001_pgdi16_lr5e-5_roll2048_1088190.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20260923_011001_p3316810` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_033301` | 3.4 | pending | alps_pgd_1.7b_984411.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20260923_011723_p663887` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_033414` | 3.4 | pending | alps_pgd_1.7b_984412.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20260923_041708_p2558591` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20260923_062543` | 3.4 | pending | gmp_pgd_grow_1.7b_984413.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20261006_044623_p2427664` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261006_140940` | 3.4 | pending | gmp_pgd_grow_1.7b_s70_roll1024_1079836.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20261006_064818_p4032810` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261006_125005` | 3.4 | pending | gmp_pgd_grow_1.7b_s70_roll2048_1079837.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20261007_033733_p2510172` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261007_141017` | 3.4 | pending | wallclock_B_alpstrain_s70_1087287.out |
| `gmp_1.7b_s70pct_lr0.0001_onpol_lmda0.33_20261007_042125_p1826726` | `cosmos1030/gmp-kd3e-1-1.7b-s70pct-lr1e-4_20261007_104809` | 3.4 | pending | wallclock_A_ours_s70_1087286.out |
| `gmp_1.7b_s70pct_lr5e-05_20260926_034710_p3335307` | `cosmos1030/gmp-kd5e-1-1.7b-s70pct-lr5e-5_20260926_153440` | 3.4 | pending | alps_sft_1.7b_2term_1003077.out |
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261009_014811_p258219` | `cosmos1030/gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_052708` | 8.0 | pending | gmp_pgd_grow_4b_1100203.out |
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261009_015438_p258220` | `cosmos1030/gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_053427` | 8.0 | pending | gmp_pgd_grow_4b_1100204.out |
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261009_061148_p11285` | `cosmos1030/gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_133459` | 8.0 | pending | gmp_pgd_grow_4b_1100256.out |
| `gmp_4b_s0pct_lr0.0001_onpol_lmda0.33_20261009_061621_p11289` | `cosmos1030/gmp-kd3e-1-4b-s0pct-lr1e-4_20261009_145513` | 8.0 | pending | gmp_pgd_grow_4b_1100257.out |
| `gmp_4b_s50pct_lr5e-05_onpol_lmda0.33_20261007_193302_p1421715` | `cosmos1030/gmp-kd3e-1-4b-s50pct-lr5e-5_20261007_224232` | 8.0 | pending | alps_sft_ntpkd_opkd_4b_1091208.out |
| `gmp_4b_s50pct_lr5e-05_onpol_lmda0.33_20261007_213041_p79349` | `cosmos1030/gmp-kd3e-1-4b-s50pct-lr5e-5_20261008_010124` | 8.0 | pending | gmp_pgd_grow_4b_1091254.out |
| `gmp_4b_s60pct_lr0.0001_onpol_lmda0.33_20261007_205426_p1006017` | `cosmos1030/gmp-kd3e-1-4b-s60pct-lr1e-4_20261008_011535` | 8.0 | pending | alps_sft_ntpkd_opkd_4b_1091209.out |
| `gmp_4b_s60pct_lr5e-05_onpol_lmda0.33_20261007_101002_p12534` | `cosmos1030/gmp-kd3e-1-4b-s60pct-lr5e-5_20261007_143047` | 8.0 | pending | gmp_pgd_grow_4b_1088478.out |
| `gmp_4b_s70pct_lr0.0001_onpol_lmda0.33_20261007_204839_p1006019` | `cosmos1030/gmp-kd3e-1-4b-s70pct-lr1e-4_20261008_015739` | 8.0 | pending | alps_sft_ntpkd_opkd_4b_1091210.out |
| `gmp_4b_s70pct_lr0.0001_onpol_lmda0.33_20261007_213705_p1015076` | `cosmos1030/gmp-kd3e-1-4b-s70pct-lr1e-4_20261008_023855` | 8.0 | pending | gmp_pgd_grow_4b_1091256.out |
| `gmp_8b_s70pct_lr0.0001_onpol_lmda1.0_20260919_015431_p476020` | `cosmos1030/gmp-8b-s70pct-lr1e-4_20260919_041114` | 8.0 | pending | gmp_pgd_grow_4b_fsdp2_961503.out |
| `gmp_s50pct_lr5e-05_20260915_115020_p3476415` | `cosmos1030/gmp-kd5e-1-s50pct-lr5e-5_20260915_130503` | 8.0 | pending | gmp_pgd_grow_4b_fsdp4_927233.out |
| `gmp_s50pct_lr5e-05_20260916_191943_p3732968` | `cosmos1030/gmp-kd5e-1-s50pct-lr5e-5_20260916_221150` | 3.4 | pending | gmp_pgd_grow_1.7b_938290.out |
| `gmp_s50pct_lr5e-05_onpol_lmda0.33_20260901_045703` | `cosmos1030/gmp-kd3e-1-s50pct-lr5e-5_20260901_071337` | 3.4 | pending | gmp_pgd_grow_1.7b_823372.out |
| `gmp_s60pct_lr5e-05_20260915_165740_p3530899` | `cosmos1030/gmp-kd5e-1-s60pct-lr5e-5_20260915_181935` | 8.0 | pending | gmp_pgd_grow_4b_fsdp4_927234.out |
| `gmp_s60pct_lr5e-05_20260916_192621_p1310166` | `cosmos1030/gmp-kd5e-1-s60pct-lr5e-5_20260916_225506` | 3.4 | pending | gmp_pgd_grow_1.7b_938291.out |
| `gmp_s60pct_lr5e-05_onpol_lmda0.33_20260901_043855` | `cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260901_071402` | 3.4 | pending | gmp_pgd_grow_1.7b_823373.out |
| `gmp_s60pct_lr5e-05_onpol_lmda0.33_20260901_094041` | `cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260901_121104` | 3.4 | pending | gmp_pgd_grow_1.7b_823379.out |
| `gmp_s60pct_lr5e-05_onpol_lmda0.33_20260902_150611` | `cosmos1030/gmp-kd3e-1-s60pct-lr5e-5_20260902_173413` | 3.4 | pending | gmp_pgd_grow_1.7b_830941.out |
| `gmp_s70pct_lr0.0001_20260915_104210_p470883` | `cosmos1030/gmp-kd5e-1-s70pct-lr1e-4_20260915_124804` | 3.4 | pending | gmp_pgd_grow_1.7b_927227.out |
| `gmp_s70pct_lr0.0001_20260915_114445_p3610577` | `cosmos1030/gmp-kd5e-1-s70pct-lr1e-4_20260915_140636` | 3.4 | pending | alps17_s70_2term_w55_lr1e4_928072.out |
| `gmp_s70pct_lr0.0001_20260915_173630_p908241` | `cosmos1030/gmp-kd5e-1-s70pct-lr1e-4_20260915_192606` | 8.0 | pending | gmp_pgd_grow_4b_fsdp4_927235.out |
| `gmp_s70pct_lr0.0001_onpol_lmda0.33_20260901_055527` | `cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_084730` | 3.4 | pending | gmp_pgd_grow_1.7b_823374.out |
| `gmp_s70pct_lr0.0001_onpol_lmda0.5_20260829_022919` | `cosmos1030/gmp-kd5e-1-s70pct-lr1e-4_20260829_053846` | 3.4 | pending | gmp_pgd_klgate_1.7b_805475.out |
| `gmp_s70pct_lr0.0001_onpol_lmda0.5_20260916_140558_p3630227` | `cosmos1030/gmp-kd5e-1-s70pct-lr1e-4_20260916_183517` | 3.4 | pending | gmp_pgd_grow_1.7b_935954.out |
| `gmp_s70pct_lr0.0001_onpol_lmda1.0_20260915_014615_p571857` | `cosmos1030/gmp-s70pct-lr1e-4_20260915_051006` | 3.4 | pending | gmp_pgd_grow_1.7b_925055.out |
| `gmp_s70pct_lr5e-05_20260915_114518_p852066` | `cosmos1030/gmp-kd5e-1-s70pct-lr5e-5_20260915_140911` | 3.4 | pending | alps17_s70_2term_w55_lr5e5_928073.out |
| `gmp_s70pct_lr5e-05_onpol_lmda0.33_20260915_134705_p3610576` | `cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260915_160528` | 3.4 | pending | alps17_s70_3term_lr5e5_928074.out |
| `gmp_s70pct_lr5e-05_onpol_lmda0.5_20260829_001532` | `cosmos1030/gmp-kd5e-1-s70pct-lr5e-5_20260829_033354` | 3.4 | pending | alps_sft_ntpkd_opkd_1.7b_805476.out |
| `gmp_s80pct_lr0.0001_onpol_lmda0.33_20260916_020553_p441981` | `cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_044553` | 3.4 | pending | gmp_pgd_grow_1.7b_930596.out |
| `qwen3_1.7b_alps_rollout2048mix_s50pct` | `cosmos1030/alps-rollout2048mix-qwen3-1.7b-s50pct` | 3.4 | ALPS rollout-refine Qwen3-1.7b s50% mode=mix rollout_max_new=2048 | slurm_1104723.out |
| `qwen3_1.7b_alps_rollout2048mix_s60pct` | `cosmos1030/alps-rollout2048mix-qwen3-1.7b-s60pct` | 3.4 | ALPS rollout-refine Qwen3-1.7b s60% mode=mix rollout_max_new=2048 | slurm_1104724.out |
| `qwen3_1.7b_alps_rollout2048mix_s70pct` | `cosmos1030/alps-rollout2048mix-qwen3-1.7b-s70pct` | 3.4 | ALPS rollout-refine Qwen3-1.7b s70% mode=mix rollout_max_new=2048 | slurm_1104725.out |
| `qwen3_1.7b_alps_rollout2048rollonly_s70pct` | `cosmos1030/alps-rollout2048rollonly-qwen3-1.7b-s70pct` | 3.4 | ALPS rollout-refine Qwen3-1.7b s70% mode=rollonly rollout_max_new=2048 | slurm_1104729.out |
| `qwen3_1.7b_alps_selfgen_v3_s50pct` | `cosmos1030/alps-s50pct_20260826_223028` | 3.4 | alps_selfgen_v3_s50pct | slurm_794294.out |
| `qwen3_1.7b_alps_selfgen_v3_s60pct` | `cosmos1030/alps-s60pct_20260826_223415` | 3.4 | alps_selfgen_v3_s60pct | slurm_794295.out |
| `qwen3_1.7b_alps_selfgen_v3_s70pct` | `cosmos1030/alps-s70pct_20260826_223323` | 3.4 | alps_selfgen_v3_s70pct | slurm_794296.out |
| `qwen3_1.7b_alps_seqlenabl_fixed_seq2048_s50pct` | `cosmos1030/alps-s50pct_20260827_131618` | 3.4 | alps_seqlenabl_fixed_seq2048_s50pct | slurm_795375.out |
| `qwen3_1.7b_alps_seqlenabl_fixed_seq4096_s50pct` | `cosmos1030/alps-s50pct_20260827_131723` | 3.4 | alps_seqlenabl_fixed_seq4096_s50pct | slurm_795376.out |
| `qwen3_1.7b_alps_seqlenabl_fixed_seq8192_s50pct` | `cosmos1030/alps-s50pct_20260827_154700` | 3.4 | alps_seqlenabl_fixed_seq8192_s50pct | slurm_795778.out |
| `qwen3_1.7b_alps_seqlenabl_selfgen_seq2048_s50pct` | `cosmos1030/alps-s50pct_20260827_131320` | 3.4 | alps_seqlenabl_selfgen_seq2048_s50pct | slurm_795372.out |
| `qwen3_1.7b_alps_seqlenabl_selfgen_seq4096_s50pct` | `cosmos1030/alps-s50pct_20260827_131415` | 3.4 | alps_seqlenabl_selfgen_seq4096_s50pct | slurm_795373.out |
| `qwen3_1.7b_alps_seqlenabl_selfgen_seq8192_s50pct` | `cosmos1030/alps-s50pct_20260827_153718` | 3.4 | alps_seqlenabl_selfgen_seq8192_s50pct | slurm_795777.out |
| `qwen3_1.7b_sparsellm_s24` | `cosmos1030/sparsellm-s50pct_20260831_204119` | 3.4 | sparsellm_s24 | slurm_825469.out |
| `qwen3_1.7b_sparsellm_s50pct` | `cosmos1030/sparsellm-s50pct_20260831_162611` | 3.4 | sparsellm_s50pct | slurm_823386.out |
| `qwen3_1.7b_sparsellm_s60pct` | `cosmos1030/sparsellm-s60pct_20260831_204149` | 3.4 | sparsellm_s60pct | slurm_825467.out |
| `qwen3_1.7b_sparsellm_s70pct` | `cosmos1030/sparsellm-s70pct_20260831_204148` | 3.4 | sparsellm_s70pct | slurm_825468.out |
| `qwen3_4b_alps_rollout2048mix_s50pct` | `cosmos1030/alps-rollout2048mix-qwen3-4b-s50pct` | 8.0 | ALPS rollout-refine Qwen3-4b s50% mode=mix rollout_max_new=2048 | slurm_1104726.out |
| `qwen3_4b_alps_rollout2048mix_s60pct` | `cosmos1030/alps-rollout2048mix-qwen3-4b-s60pct` | 8.0 | ALPS rollout-refine Qwen3-4b s60% mode=mix rollout_max_new=2048 | slurm_1104727.out |
| `qwen3_4b_alps_rollout2048mix_s70pct` | `cosmos1030/alps-rollout2048mix-qwen3-4b-s70pct` | 8.0 | ALPS rollout-refine Qwen3-4b s70% mode=mix rollout_max_new=2048 | slurm_1104728.out |
| `qwen3_4b_alps_selfgen_v3_s50pct` | `cosmos1030/alps-s50pct_20260827_094710` | 8.0 | alps_selfgen_v3_s50pct | slurm_794882.out |
| `qwen3_4b_alps_selfgen_v3_s60pct` | `cosmos1030/alps-s60pct_20260827_105937` | 8.0 | alps_selfgen_v3_s60pct | slurm_794883.out |
| `qwen3_4b_alps_selfgen_v3_s70pct` | `cosmos1030/alps-s70pct_20260827_111542` | 8.0 | alps_selfgen_v3_s70pct | slurm_794884.out |
| `qwen3_4b_alps_selfgen_v3_s80pct` | `cosmos1030/alps-s80pct_20260925_220218` | 8.0 | alps_selfgen_v3_s80pct | slurm_1003078.out |
| `qwen3_4b_gptq_w2_g128_s0pct` | `cosmos1030/sparsellm-s0pct_20261008_133444` | 8.0 | SparseLLM Pruning Qwen3-4B s0% | slurm_1100255.out |
| `qwen3_4b_gptq_w2_pc_s0pct` | `cosmos1030/sparsellm-s0pct_20261008_123842` | 8.0 | SparseLLM Pruning Qwen3-4B s0% | slurm_1099996.out |
| `qwen3_4b_gptq_w3_g128_s0pct` | `cosmos1030/sparsellm-s0pct_20261008_131030` | 8.0 | SparseLLM Pruning Qwen3-4B s0% | slurm_1100201.out |
| `qwen3_4b_gptq_w3_pc_s0pct` | `cosmos1030/sparsellm-s0pct_20261008_123705` | 8.0 | SparseLLM Pruning Qwen3-4B s0% | slurm_1099995.out |
| `qwen3_4b_sparsellm_s24` | `cosmos1030/sparsellm-s50pct_20260831_220245` | 8.0 | sparsellm_s24 | slurm_825473.out |
| `qwen3_4b_sparsellm_s50pct` | `cosmos1030/sparsellm-s50pct_20260831_204946` | 8.0 | sparsellm_s50pct | slurm_825470.out |
| `qwen3_4b_sparsellm_s60pct` | `cosmos1030/sparsellm-s60pct_20260831_210736` | 8.0 | sparsellm_s60pct | slurm_825471.out |
| `qwen3_4b_sparsellm_s70pct` | `cosmos1030/sparsellm-s70pct_20260831_211530` | 8.0 | sparsellm_s70pct | slurm_825472.out |
| `qwen3_4b_sparsellm_s80pct` | `cosmos1030/sparsellm-s80pct_20260903_113201` | 8.0 | sparsellm_s80pct | slurm_837519.out |
| `qwen3_8b_selfgen_v3_alps_s50pct` | `cosmos1030/alps-s50pct_20260827_095701` | 16.4 | alps_8b_selfgen_v3_s50pct | slurm_794887.out |
| `qwen3_8b_selfgen_v3_alps_s60pct` | `cosmos1030/alps-s60pct_20260827_141236` | 16.4 | alps_8b_selfgen_v3_s60pct | slurm_794888.out |
| `qwen3_8b_selfgen_v3_alps_s70pct` | `cosmos1030/alps-s70pct_20260827_144621` | 16.4 | alps_8b_selfgen_v3_s70pct | slurm_794889.out |
| `qwen3_8b_selfgen_v3_alps_s80pct` | `cosmos1030/alps-s80pct_20260925_235139` | 16.4 | alps_8b_selfgen_v3_s80pct | slurm_1003079.out |
| `qwen3_8b_sparsellm_s24` | `cosmos1030/sparsellm-s50pct_20260901_090527` | 16.4 | sparsellm_s24 | slurm_825477.out |
| `qwen3_8b_sparsellm_s50pct` | `cosmos1030/sparsellm-s50pct_20260901_050129` | 16.4 | sparsellm_s50pct | slurm_825474.out |
| `qwen3_8b_sparsellm_s60pct` | `cosmos1030/sparsellm-s60pct_20260901_062412` | 16.4 | sparsellm_s60pct | slurm_825475.out |
| `qwen3_8b_sparsellm_s70pct` | `cosmos1030/sparsellm-s70pct_20260901_073704` | 16.4 | sparsellm_s70pct | slurm_825476.out |
| `qwen3_8b_sparsellm_s80pct` | `cosmos1030/sparsellm-s80pct_20260903_020850` | 16.4 | sparsellm_s80pct | slurm_839780.out |

## 6. 1.7B dense→2:4 개선 탐색 (2026-10-15~16, 로컬 삭제 전 기록)

전부 Qwen3-1.7B dense 출발, 2:4, PGD grow_to_target, 롤아웃 2048 sidecar, NTP/KD/OPD 0.33씩, 2048 step, saliency fisher(별도 표기 제외). avg4 = math500/IFEval/LCB/GSM8K (seed42 in-job).

| HF repo (cosmos1030/...) | 잡 | arm | lr | klb | avg4 |
|---|---|---|---|---|---|
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261015_231527 | 1150711 | 기준 | 5e-5 | 0.02 | 34.6 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_010544 | 1150712 | 그룹 보정(ridge 1e-6) | 5e-5 | 0.02 | 23.1 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261015_234949 | 1150715 | lasso shrink 3e-3 | 5e-5 | 0.02 | 32.5 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_001249 | 1150716 | lasso shrink 1e-2 | 5e-5 | 0.02 | 29.0 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_005049 | 1150717 | 그룹 보정 + shrink 1e-2 | 5e-5 | 0.02 | 25.4 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_002513 | 1150814 | wanda saliency (layer scope) | 5e-5 | 0.02 | 29.7 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_023705 | 1150815 | wanda + 그룹 보정 | 5e-5 | 0.02 | 17.2 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_013829 | 1151063 | 행 단위 refit | 5e-5 | 0.02 | 32.8 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_035858 | 1151064 | refit + shrink 1e-2 | 5e-5 | 0.02 | 34.1 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_052213 | 1151065 | wanda + refit | 5e-5 | 0.02 | 26.3 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr5e-5_20261016_054628 | 1151101 | 그룹 보정(ridge 1e-2) | 5e-5 | 0.02 | 26.2 |
| gmp-kd3e-1-1.7b-n24-s50pct-lr1e-4_20261016_161122 | 1156449 | 행 단위 refit | 1e-4 | 0.02 | 36.0 |
