# TR x OPD 곡선 — 런 출처

| 파일 | wandb id | 설명 |
|---|---|---|
| `4b_scout_r5j1uw8d.csv` | [r5j1uw8d](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b_nostrip8192/runs/r5j1uw8d) | TR + grow2target, klb=0.02, step 1-2048 |
| `4b_jump_hkshqtev.csv` | [hkshqtev](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b_nostrip8192/runs/hkshqtev) | one-step jump, klb=0.02, step 1-748 (crashed; 이후 3olz82te 가 513- 이어받음) |
| `4b_jump_tail_3olz82te.csv` | [3olz82te](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_4b_nostrip8192/runs/3olz82te) | one-step jump 이어받은 런, step 513-2048 |
| `8b_scout_0z5sc32y.csv` | [0z5sc32y](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b_nostrip8192/runs/0z5sc32y) | TR + grow2target, klb=0.03, step 1-520 (crashed; 4ebzii1b -> ajm5l60w 로 이어짐) |
| `8b_scout_mid_4ebzii1b.csv` | [4ebzii1b](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b_nostrip8192/runs/4ebzii1b) | SCOUT 체인 2번째, step 513-1469 |
| `8b_scout_tail_ajm5l60w.csv` | [ajm5l60w](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b_nostrip8192/runs/ajm5l60w) | SCOUT 체인 3번째, step 1281-2048 (최종 체크포인트) |
| `8b_jump_iuwf08pf.csv` | [iuwf08pf](https://wandb.ai/dyk6208-gwangju-institute-of-science-and-technology/reasoning_qwen3_8b_nostrip8192/runs/iuwf08pf) | one-step jump, klb=0.03, step 1-2048 (끊김 없음) |

`train/grad_norm` 은 **전체 손실**(NTP+KD+OPD) 기준이다. OPD 단독이 아니다.
jump 런도 step 1-7 은 dense 이고 step 8 에 한 번에 목표 희소도로 간다.
