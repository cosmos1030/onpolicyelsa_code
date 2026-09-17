# IPO on s80 — 실험 핸드오프 (Qwen3-4B)

2026-09-17 · 대상: 오민제 · 이전 자료: `elsa/docs/IPO_GRPO_HANDOFF.md`

---

## 0. 요청

80% 희소도 체크포인트에 **IPO(UltraFeedback)** 를 걸고 **MATH-500**을 재주시면 됩니다.
70%로는 가속 효과가 작아서 80%로 다시 만들었습니다.

- **① ours(SCOUT) s80 먼저**, ② ALPS+recovery s80 은 시간이 되시면
- 학습: 기존 S70 레시피 그대로, `model_name_or_path`만 교체
- 평가: MATH-500, `--profile long`, seed 42, 두 모델 동일 예산
- 지표: 정확도 + **평균 토큰수와 절단율을 각각 전체 / 정답 / 오답** (§4)

길이를 같이 보는 이유는 토큰당 생성이 빨라져도 답이 길어지면 end-to-end 이득이
줄기 때문입니다.

---

## 1. 체크포인트

| | HF repo |
|---|---|
| **① ours (SCOUT) s80** | `cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740` |
| **② ALPS+recovery s80** | `cosmos1030/gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952` |

---

## 2. 실행

### 환경

```bash
bash elsa/scripts/setup_ipo_grpo_env.sh
```

env 두 개가 만들어집니다 — 학습용 `rac_vllm084`(vllm 0.8.4), 평가용 `rac`(vllm 0.10.0).
버전이 달라야 해서 합칠 수 없고, 런처가 알아서 전환합니다. 미리 준비하실 건
`~/.hf_token`과 `WANDB_API_KEY`뿐입니다.

### 데이터

`trl-lib/ultrafeedback_binarized` — config에 지정돼 있어 자동으로 받습니다.
새로 만드실 것 없습니다.

### 학습

```bash
sbatch elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh \
  <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]
```

config는 `RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_{trgmp,alpssft}_fullft.yaml`
를 복사해서 `model_name_or_path`와 `output_dir`만 바꾸시면 됩니다.

유지할 값:

```yaml
dataset_name: trl-lib/ultrafeedback_binarized
loss_type: ipo
beta: 0.1
sparse_optimizer: MaskedAdam     # ★
learning_rate: 1.0e-5            # S70에서 5e-6보다 좋았음
num_train_epochs: 1
max_length: 2048
max_prompt_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

**주의 두 가지**

- `sparse_optimizer: MaskedAdam`이 빠지면 IPO가 0인 가중치를 다시 채워 **희소도가
  풀립니다.** 학습 후 zero fraction이 0.80인지 한 번 확인해주세요.
- 반드시 `_fullft` 런처를 쓰세요. 이전 LoRA 버전은 `merge_and_unload()`가 프루닝
  마스크를 몰라 같은 이유로 희소도가 풀립니다.

### 평가

```bash
python b200_scripts/resume_eval_lighteval.py \
  --model_dir <경로 또는 HF repo> \
  --tasks math500 \
  --tp_size 1 --profile long --seed 42 --no_hub
```

`--profile long`의 MATH-500 설정: `max_new_tokens=16,384`, `max_model_length=17,408`,
`temperature=0.6, top_p=0.95, top_k=20`. 1회 약 47~76분 (4B, B200 1장).
**두 모델에 같은 예산을 쓰세요.**

---

## 3. 참고 — 기존 S70 IPO 결과

**참고용입니다.** `TR-GMP`는 deprecated된 이전 방법이고, 이때 MATH-500을 8,192
토큰으로 쟀습니다 (지금은 16,384).

| 시작 체크포인트 | lr | Reas.avg | Math500 | LCB | GPQA | IFEval | GSM8K | wandb |
|---|---|---|---|---|---|---|---|---|
| TR-GMP s70 | 5e-6 | 44.4 | 69.6 | 5.22 | 30.3 | 41.96 | 74.75 | vpwc73p4 |
| TR-GMP s70 | 1e-5 | **46.6** | 72.4 | 5.22 | 33.3 | 47.87 | 74.0 | lqsauwzq |
| ALPS-SFT s70 | 5e-6 | 41.8 | 66.8 | 8.21 | 23.74 | 35.67 | 74.75 | qrwaqpum |
| ALPS-SFT s70 | 1e-5 | 43.6 | 69.4 | 8.21 | 23.23 | 40.67 | 76.57 | sc4tgsm1 |

| 시작점 | pre-IPO | post-IPO (5e-6) | post-IPO (1e-5) |
|---|---|---|---|
| TR-GMP | 43.1 | 44.4 (+1.3) | **46.6 (+3.5)** |
| ALPS-SFT | 40.9 | 41.8 (+0.9) | 43.6 (+2.7) |

두 arm 모두 **lr=1e-5가 5e-6보다 좋았습니다.**

---

## 4. 길이 지표

평가 시 아래가 자동 기록됩니다 (MATH-500이면 `{bench}` = `math500`). 여섯 칸을
모두 보고해주세요.

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | `{bench}_avg_output_tokens` | `{bench}_correct_avg_output_tokens` | `{bench}_wrong_avg_output_tokens` |
| 절단율 | `{bench}_truncation_rate` | `{bench}_correct_truncation_rate` | `{bench}_wrong_truncation_rate` |

### IPO 전 값 (이미 측정됨 — 다시 재지 않으셔도 됩니다)

**ours s80** — 정확도 48.0% (정답 240 / 오답 260)

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | 11,421 | **7,265** | 15,257 |
| 절단율 | 52.0% | **16.7%** | 84.6% |

**ALPS+recovery s80** — 정확도 17.8% (정답 89 / 오답 411)

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | 15,831 | **14,587** | 16,101 |
| 절단율 | 93.6% | **82.0%** | 96.1% |

### 해석 주의

baseline은 **정답의 82%가 예산 상한에 걸려** 있습니다. 14,587은 "이만큼 쓴다"가
아니라 "천장에 붙었다"에 가깝습니다. 그래서 IPO로 실제로 짧아져도 **평균 토큰수는
안 움직이고 절단율만 내려갈 수 있습니다** — 그걸 "효과 없음"으로 읽으시면 안 됩니다.

오답은 양쪽 다 천장이라 정보가 없으니, 주된 비교는 **정답일 때** 기준입니다.
