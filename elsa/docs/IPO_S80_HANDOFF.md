# IPO on s80 — 실험 핸드오프 (Qwen3-4B)

작성: 2026-09-17 · 대상: 오민제
이전 자료: `elsa/docs/IPO_GRPO_HANDOFF.md` (환경 설정 / 실행 방법)

---

## 0. 요청 요약

80% 희소도 체크포인트에 **IPO(UltraFeedback)를 걸고, MATH-500을 long 프로파일로**
재주시면 됩니다. 70%로는 가속 효과가 작아서 80%로 다시 만들었습니다.

- 대상: **① ours(SCOUT) s80 을 먼저**, **② ALPS+recovery s80 은 시간이 되시면**
- 학습: 기존 S70 IPO 레시피 그대로, `model_name_or_path`만 교체
- 평가: **MATH-500만**, `--profile long`, seed 42, **두 모델 동일 예산**
- 지표: 정확도에 더해 **평균 토큰수와 절단율을 각각 전체 / 정답 / 오답으로** (§6)

**왜 길이를 같이 보는가**: 희소 커널로 토큰당 생성이 빨라져도, 프루닝된 모델이
답을 더 길게 쓰면 **end-to-end 추론 시간의 이득이 줄어듭니다.** IPO 계열로 길이가
줄어드는 경향이 관찰된 적이 있어서, 실제 가속을 주장하려면 "토큰당 속도"와
"총 생성 토큰수"를 따로 재고 곱해서 보여줘야 합니다. 총 시간만 보면 두 요인이
상쇄되어 원인이 안 보입니다.

---

## 1. 참고 — 기존 S70 IPO 결과

**아래 표는 참고용입니다. 그대로 비교하지 마세요.** 두 가지 이유입니다:

1. `TR-GMP`는 **deprecated된 이전 방법**입니다. 현재 "ours"는 TR-GMP가 아니라
   PGD grow-to-target(SCOUT)입니다.
2. 이때 MATH-500을 **8,192 토큰**으로 쟀습니다. 지금 기준(long, 16,384)과 다르고,
   8,192에서는 절단이 심해 점수가 눌립니다.

IPO full-FT (UltraFeedback, MaskedAdam), S70:

| 시작 체크포인트 | lr | Reas.avg | Math500 | LCB | GPQA | IFEval | GSM8K | wandb |
|---|---|---|---|---|---|---|---|---|
| TR-GMP s70 | 5e-6 | 44.4 | 69.6 | 5.22 | 30.3 | 41.96 | 74.75 | vpwc73p4 |
| TR-GMP s70 | 1e-5 | **46.6** | 72.4 | 5.22 | 33.3 | 47.87 | 74.0 | lqsauwzq |
| ALPS-SFT s70 | 5e-6 | 41.8 | 66.8 | 8.21 | 23.74 | 35.67 | 74.75 | qrwaqpum |
| ALPS-SFT s70 | 1e-5 | 43.6 | 69.4 | 8.21 | 23.23 | 40.67 | 76.57 | sc4tgsm1 |

pre-IPO 기준선:

| 체크포인트 | Reas.avg | Math500 | LCB | GPQA | IFEval | GSM8K | wandb |
|---|---|---|---|---|---|---|---|
| TR-GMP s70 (pre-IPO) | 43.1 | 71.0 | 4.9 | 28.8 | 37.3 | 73.4 | 657qk0cq |
| ALPS-SFT s70 (pre-IPO) | 40.9 | 66.2 | 9.0 | 26.8 | 31.4 | 70.9 | 5x4prktp |

전후:

| 시작점 | pre-IPO | post-IPO (5e-6) | post-IPO (1e-5) |
|---|---|---|---|
| TR-GMP | 43.1 | 44.4 (+1.3) | **46.6 (+3.5)** |
| ALPS-SFT | 40.9 | 41.8 (+0.9) | 43.6 (+2.7) |

**여기서 가져갈 것 하나**: 두 arm 모두 **lr=1e-5가 5e-6보다 좋았습니다.** s80에서도
1e-5부터 시작하시는 걸 권합니다.

---

## 2. 체크포인트 (전부 Qwen3-4B, 80% unstructured)

| 역할 | HF repo |
|---|---|
| **ours (SCOUT) s80** | `cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740` |
| **ALPS+recovery s80** (NTP+KD+OPD) | `cosmos1030/gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952` |
| (참고) ALPS+recovery s80, OPD 제거 | `cosmos1030/gmp-kd5e-1-4b-s80pct-lr1e-4_20260917_102335` |
| (참고) ALPS 원샷 s80, 회복 없음 | `cosmos1030/alps-qwen3-4b-s80pct` |
| (참고) dense | `Qwen/Qwen3-4B` |

---

## 3. 환경 / 데이터 / 스크립트

### 환경 — conda env 두 개가 필요합니다

```bash
bash elsa/scripts/setup_ipo_grpo_env.sh
```

| env | 용도 | 핵심 핀 |
|---|---|---|
| `rac_vllm084` | **IPO 학습** | torch 2.6.0, **vllm 0.8.4** |
| `rac` | **평가**(lighteval) | torch 2.7.1, **vllm 0.10.0** |

**하나로 합치면 안 됩니다.** vllm ≥ 0.10은 trl의 서버 모드 `update_named_param`에서
업스트림 데드락(huggingface/trl#3608)에 걸려서 학습용은 0.8.4로 핀돼 있고, 평가
경로(lighteval)는 0.10.0이 필요합니다. 런처 스크립트가 학습엔 `rac_vllm084`,
평가엔 `rac`로 알아서 전환하므로 두 이름을 그대로 쓰시면 수동 전환은 없습니다.

PYTHONPATH 등 나머지는 런처가 잡습니다. 미리 준비하실 건 시크릿 두 개뿐입니다 —
`~/.hf_token` 파일과 `WANDB_API_KEY` (README의 "Environment variables / secrets").

### 데이터

| 용도 | 데이터셋 | 비고 |
|---|---|---|
| IPO 학습 | **`trl-lib/ultrafeedback_binarized`** | config에 지정, 자동 다운로드. 새로 만들 것 없음 |
| 평가 | lighteval이 Hub에서 task별로 가져옴 | `HF_HUB_OFFLINE` 등이 켜져 있으면 해제 필요 |

### 스크립트

| 단계 | 파일 |
|---|---|
| 환경 구축 | `elsa/scripts/setup_ipo_grpo_env.sh` |
| IPO 학습 (SLURM) | `elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh` |
| 학습 config | `RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_{trgmp,alpssft}_fullft.yaml` |
| 내부 진입점 | `accelerate launch --config_file recipes/plain_1gpu.yaml src/open_r1/dpo.py --config <yaml>` |
| 평가 | `b200_scripts/resume_eval_lighteval.py` |

실행 형식:

```bash
sbatch elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh \
  <CONFIG_YAML> <OUTPUT_DIR> <TRAIN_WBID> [MAX_STEPS] [LR]
```

GPU 1장(A100-80GB), `--time=24:00:00` 기준으로 잡혀 있습니다.

### ⚠ 반드시 이 `_fullft` 런처를 쓰세요

이전 LoRA 버전(`slurm_ipo_ultrafeedback_s70.sh`)은 `merge_and_unload()`가 프루닝
마스크를 몰라서 희소도가 풀립니다(zero_frac 0.70 → ~0.0001). `_fullft`는 merge 단계가
없어 **저장된 체크포인트가 곧 평가 체크포인트**입니다.

---

## 4. IPO 학습 — 기존 레시피, 모델만 교체

새로 pair를 모을 필요 없습니다. S70에서 쓰던 UltraFeedback 레시피가 그대로 있습니다.

기존 config:
- `RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_trgmp_fullft.yaml`
- `RAC/open-r1-main/recipes/Qwen3-4B/dpo/config_ipo_ultrafeedback_s70_alpssft_fullft.yaml`
- 실행: `elsa/scripts/slurm_ipo_ultrafeedback_s70_fullft.sh`

유지할 값:

```yaml
dataset_name: trl-lib/ultrafeedback_binarized
loss_type: ipo
beta: 0.1
sparse_optimizer: MaskedAdam     # ★ 반드시 유지
learning_rate: 1.0e-5            # S70에서 5e-6보다 좋았음
lr_scheduler_type: cosine
num_train_epochs: 1
max_length: 2048
max_prompt_length: 1024
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

바꿀 곳:

```yaml
# ① ours  ← 이것부터
model_name_or_path: cosmos1030/gmp-kd3e-1-s80pct-lr1e-4_20260916_220740
output_dir: <새 경로>

# ② baseline (ALPS+recovery) ← 시간이 되시면
model_name_or_path: cosmos1030/gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952
output_dir: <다른 경로>
```

**순서**: ours 한 arm만으로도 "s80에서 IPO가 길이를 줄이는가"는 답이 나옵니다.
baseline은 그 효과가 방법에 특유한 것인지 프루닝 모델 일반의 성질인지를 가르는
용도라, 시간이 없으면 뒤로 미루셔도 됩니다. ours 쪽이 길이 측정도 훨씬 잘 되고요
(정답의 83%가 예산 안에서 끝남 — §6).

**`sparse_optimizer: MaskedAdam`이 빠지면 실험이 무의미해집니다.** 프루닝으로 0이 된
가중치를 매 스텝 0으로 고정하는 역할인데, 일반 Adam이면 IPO가 그 자리를 다시 채워
**희소도가 풀립니다.** 그러면 "프루닝 모델에 IPO"가 아니라 "dense로 되돌리며 IPO"가
되어 가속 주장이 무너집니다. 학습 후 zero fraction이 0.80으로 유지되는지 한 번만
확인해주시면 안전합니다.

---

## 5. 평가 — MATH-500, 동일 예산

```bash
python b200_scripts/resume_eval_lighteval.py \
  --model_dir <경로 또는 HF repo> \
  --tasks math500 \
  --tp_size 1 --profile long --seed 42 --no_hub
```

`--tasks math500`으로 MATH-500만 돌립니다 (1회 약 47~76분). 나머지 4종은 pre-IPO
값이 §3에 있고, IPO 후 필요해지면 `--tasks`를 빼면 5종 전부 돌아갑니다 (약 6시간).

`--profile long`의 벤치별 예산 (`elsa/lib/lighteval_bench.py`의 `_LONG_BENCHMARKS`):

| 벤치 | max_new_tokens | max_model_length |
|---|---|---|
| MATH-500 / GPQA / IFEval | 16,384 | 17,408 |
| LCB | 32,768 | 33,792 |
| GSM8K | 8,192 | 9,216 |

- `temperature=0.6, top_p=0.95, top_k=20`, seed 42
- `tp_size`는 1로. 이 코드베이스에서 tp=4는 vLLM cleanup에서 행이 걸립니다.
- `max_model_length = max_new_tokens + 1024`인 게 의도된 설계입니다. 둘을 같게 두면
  프롬프트가 생성 예산을 잠식해서 상한이 8,192인데 7,925에서 잘리는 식이 됩니다.

**모든 모델에 같은 예산을 쓰세요.** 모델마다 예산을 바꾸면 점수도 길이도 비교가
불가능해집니다.

참고 소요 시간 (4B, B200 1장): MATH-500 47~76분. (참고: GPQA 30분, IFEval 80분,
LCB 180분, GSM8K 55분 → 5종 전부면 약 6시간.)

---

## 6. 길이 지표 — 전체 / 정답 / 오답을 각각

정확도만이 아니라 **평균 토큰수와 절단율을 세 갈래로 나눠서** 보고해주세요.
평가 시 벤치마크별로 아래가 자동 기록됩니다 (wandb 연결 시). MATH-500이면
`{bench}` = `math500`입니다.

| | 전체 | 정답일 때 | 오답일 때 |
|---|---|---|---|
| 평균 토큰수 | `{bench}_avg_output_tokens` | `{bench}_correct_avg_output_tokens` | `{bench}_wrong_avg_output_tokens` |
| 절단율 | `{bench}_truncation_rate` | `{bench}_correct_truncation_rate` | `{bench}_wrong_truncation_rate` |

`{bench}_avg_gen_cap`, `{bench}_max_output_tokens`도 같이 찍힙니다.

wandb 없이 돌리셨다면 details parquet에서 직접 계산됩니다 —
`model_response.output_tokens` 길이와 `metric`의 정오답으로 위 6칸이 나옵니다.

### IPO 전 실측 (MATH-500, long 16,384)

아래는 이미 측정된 pre-IPO 값입니다. **다시 재지 않으셔도 됩니다.**

**ours s80** — 정확도 48.0% (정답 240 / 오답 260)

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | 11,421 | **7,265** | 15,257 |
| 절단율 | 52.0% | **16.7%** | 84.6% |

**ALPS+recovery s80 (NTP+KD+OPD)** — 정확도 17.8% (정답 89 / 오답 411)

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | 15,831 | **14,587** | 16,101 |
| 절단율 | 93.6% | **82.0%** | 96.1% |

**ALPS+recovery s80 (NTP+KD)** — 정확도 17.0% (정답 85 / 오답 415)

| | 전체 | 정답 | 오답 |
|---|---|---|---|
| 평균 토큰수 | 15,522 | **13,258** | 15,986 |
| 절단율 | 91.0% | **71.8%** | 94.9% |

### 해석할 때 주의

같은 예산인데 **ours는 정답의 83%가 예산 안에서 끝나고, baseline은 정답의 18%만
끝납니다.** baseline의 14,587은 "이만큼 쓴다"가 아니라 **"천장에 붙었다"** 에
가깝습니다.

그래서 IPO 전후 비교에서 이런 일이 생길 수 있습니다: IPO로 baseline이 실제로
짧아져도 여전히 천장에 걸려 있으면 **평균 토큰수가 거의 안 움직입니다.** 이때
움직이는 건 **절단율**입니다. 그러니 두 값을 항상 같이 보고, 절단율이 내려갔는데
평균 길이가 그대로면 "효과 없음"이 아니라 "이제 막 측정 범위에 들어왔음"으로
읽으셔야 합니다.

오답은 양쪽 다 천장(15,257 / 16,101)이라 정보가 거의 없습니다. 주된 비교는
**정답일 때** 기준입니다.
