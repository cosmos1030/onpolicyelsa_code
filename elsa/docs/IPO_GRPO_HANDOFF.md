안녕하세요, IPO/GRPO 실험 검수 관련해서 정리해서 공유드립니다.

## 정정할 부분
지금까지 돌린 IPO/GRPO는 **TR-GMP를 "ours"로 놓고** 실험한 것이었고, 비교 대상이었던 "ALPS+SFT"는 사실 **ALPS+recovery**입니다 (SFT가 아니라 NTP+KD+OPD 신호로 복구한 것).

## 앞으로의 기준
"ours"가 바뀝니다 — 4B S70 최고 성능 모델:
**https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_080954**

앞으로 실험은 **ALPS+recovery vs 이 ours 모델**에서 GRPO/IPO 효과를 비교하는 구도로 봐주시면 됩니다.

## 환경 설정
레포 clone 후:
```bash
bash elsa/scripts/setup_ipo_grpo_env.sh
```
→ `rac`(reasoning eval), `rac_vllm084`(IPO/GRPO 학습) 두 conda env 자동 생성됩니다.

## 스크립트
- IPO 학습: [`elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh)
  - config: [`config_ipo_ultrafeedback_s70_trgmp_fullft.yaml`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_trgmp_fullft.yaml) / [`config_ipo_ultrafeedback_s70_alpssft_fullft.yaml`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_alpssft_fullft.yaml)
- GRPO 학습: [`elsa/scripts/slurm_grpo_overthinking_s70.sh`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/elsa/scripts/slurm_grpo_overthinking_s70.sh)
  - config: [`config_overthinking_s70.yaml`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/grpo/config_overthinking_s70.yaml) / [`_ablation_noLen`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/grpo/config_overthinking_s70_ablation_noLen.yaml) / [`_alpssft`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/grpo/config_overthinking_s70_alpssft.yaml) / [`_alpssft_ablation_noLen`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/RAC/open-r1-main/recipes/Qwen3-4B/grpo/config_overthinking_s70_alpssft_ablation_noLen.yaml)
- Reasoning eval (math500/lcb/gpqa/ifeval/gsm8k): [`elsa/scripts/slurm_gmp_eval_only.sh`](https://github.com/cosmos1030/onpolicyelsa_code/blob/master/elsa/scripts/slurm_gmp_eval_only.sh) `<model_path> <wandb_run_id> 0.7 quick`

## 체크포인트 + Reasoning 벤치마크 (HF, quick=8192 프로파일)

| 모델 | Reas.avg | Math500 | LCB | GPQA | IFEval | GSM8K | HF |
|---|---|---|---|---|---|---|---|
| **새 ours 베이스 (4B S70 best, δ=0.02)** | **45.58** | 72.4 | 8.21 | 35.86 | 39.93 | 71.49 | [link](https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260901_080954) |
| (기존) TR-GMP s70 베이스 | 43.1 | 71.0 | 4.9 | 28.8 | 37.3 | 73.4 | [link](https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr5e-5_20260811_115604) |
| (기존) ALPS+recovery s70 베이스 | 40.9 | 66.2 | 9.0 | 26.8 | 31.4 | 70.9 | [link](https://huggingface.co/cosmos1030/gmp-kd3e-1-s70pct-lr1e-4_20260814_035030) |
| TR-GMP+IPO lr5e-6 | 44.4 | 69.6 | 5.22 | 30.3 | 41.96 | 74.75 | [link](https://huggingface.co/cosmos1030/ipo-trgmp-s70-ultrafeedback-fullft-lr5e-6) |
| TR-GMP+IPO lr1e-5 | **46.6** | 72.4 | 5.22 | 33.3 | 47.87 | 74.0 | [link](https://huggingface.co/cosmos1030/ipo-trgmp-s70-ultrafeedback-fullft-lr1e-5) |
| ALPS+recovery+IPO lr5e-6 | 41.8 | 66.8 | 8.21 | 23.74 | 35.67 | 74.75 | [link](https://huggingface.co/cosmos1030/ipo-alpssft-s70-ultrafeedback-fullft-lr5e-6) |
| ALPS+recovery+IPO lr1e-5 | 43.6 | 69.4 | 8.21 | 23.23 | 40.67 | 76.57 | [link](https://huggingface.co/cosmos1030/ipo-alpssft-s70-ultrafeedback-fullft-lr1e-5) |
| TR-GMP+GRPO(overthinking) | 43.9 | 68.6 | 6.0 | 33.3 | 38.3 | 73.3 | [link](https://huggingface.co/cosmos1030/grpo-trgmp-s70-overthinking) |
| TR-GMP+GRPO(noLen ablation) | 44.0 | 69.8 | 4.5 | 31.8 | 39.6 | 74.5 | [link](https://huggingface.co/cosmos1030/grpo-trgmp-s70-ablation-noLen) |
| ALPS+recovery+GRPO(overthinking) | 40.0 | 65.4 | 7.8 | 24.2 | 31.2 | 71.5 | [link](https://huggingface.co/cosmos1030/grpo-alpssft-s70-overthinking) |
| ALPS+recovery+GRPO(noLen ablation) | 40.4 | 66.2 | 6.3 | 28.8 | 31.4 | 69.1 | [link](https://huggingface.co/cosmos1030/grpo-alpssft-s70-ablation-noLen) |

아래 8개(IPO 4개 + GRPO 4개)는 **옛 TR-GMP 베이스 기준**으로 돌린 것들이라, 새 ours 베이스(45.58)로는 아직 IPO/GRPO를 안 돌렸습니다 — 검수 끝나면 새 베이스로 다시 돌릴 예정입니다.

감사합니다!
