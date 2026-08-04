# OT80/FW20 재실행 계획

**배경**: `math_220k_cot.jsonl`(ELSA ADMM) / `ot3_fineweb_20k.jsonl`(GMP·ALPS·SparseLLM, DeepSeek 토크나이저로 렌더링됨)을 잘못 써서 Qwen3 계열 실험 전체를 OT80/FW20 통일 데이터로 재실행함.

**새 데이터**: `/home1/doyoonkim/projects/elsa/data/ot3_fineweb_200k_qwen3.jsonl` (OpenThoughts3 80% + FineWeb-Edu 20%, 20만 샘플, Qwen3 토크나이저 렌더링, seed=42). Qwen3-1.7B/4B/8B 전부 동일 토크나이저라 이 파일 하나로 공용.

**새 학습 예산 (ELSA + GMP 공통)**: `steps=4096`, **global batch = 16**, `seqlen=2048` (예: `batch_size=1 × grad_accum=4 × world_size=4`, 또는 GPU 개수에 맞게 accum/world_size 조합만 바꾸고 곱은 16 유지) → 4096 × 16 × 2048 ≈ **0.134B(1.34억) 토큰**.

**ELSA lmda_schedule**: 전부 **constant**로 통일 (기존 plain 계열은 cosine을 썼었는데 이번 재실행부터는 constant. cubic/trust-region z-schedule은 이미 이번 세션에서 constant로 테스트해왔으므로 그대로 유지).

**LR warmup (ELSA + GMP 공통)**: 비율(ratio) 기반 대신 **스텝 기준 256 스텝**으로 통일 (`lr_warmup_steps=256`).

**LR scheduling (ELSA + GMP 공통)**: 기본값을 **constant**로 통일 — `lr_scheduler=constant_with_warmup` (256스텝 warmup 후 flat, decay 없음). `lr_scheduler='constant'`(warmup 없는 순수 constant)로 하면 HF가 warmup_steps를 무시해버려서 `constant_with_warmup`으로 명시함.

**플래그 통일**: `lr`, `steps`, `lr_scheduler`, `lr_warmup_steps`, `seqlen` — ELSA(`admm_lr`/`admm_steps`/`admm_lr_scheduler`/`admm_warmup_steps`)와 GMP(`gmp_lr`/`gmp_steps`/`gmp_lr_schedule`/`gmp_lr_warmup_steps`/`gmp_max_seq_len`)가 어차피 같은 잡에서 동시에 안 돌길래 각각 따로 갖고 있던 flag를 공용 하나로 합침 (`main.py`/`lib/prune.py`/`lib/gkd_admm.py`/`lib/gmp_trainer.py`/`lib/grpo_opkd.py` 반영 완료). `gmp_dense_warmup_steps`는 cubic ramp 지연까지 겸하도록 통일(이전엔 `gmp_cubic_warmup_steps`를 따로 만들었다가 다시 제거).

**재실행 제외 (그대로 유지)**: SparseGPT, Wanda — `prune_and_eval.py`의 `build_calib_dataset()`이 매번 원본에서 fresh 렌더링하므로 이 문제와 무관.

**하이퍼파라미터 출처**: 아래 값들은 기존 wandb 기록(`reasoning_qwen3_1.7b`/`elsa_qwen3_4b`/`reasoning_qwen3_8b` 프로젝트)에서 실제 쓰인 값을 그대로 가져온 것. Step 수/global batch만 새 예산으로 교체.

---

## Qwen3-1.7B (`reasoning_qwen3_1.7b`)

### ELSA NTP-ADMM (plain)
| sparsity | lr | lmda | lmda_schedule | interval | 비고 (기존 wbid) |
|---|---|---|---|---|---|
| 50% | 1e-4 | 1e-3 | **constant** | 32 | ef2zr9qy (기존 cosine → constant로 통일) |
| 60% | 1e-4 | 1e-3 | **constant** | 32 | e6a16chc (기존 cosine → constant로 통일) |
| 70% | 1e-4 | 5e-3 | **constant** | 32 | c2pp4ud8 (기존 cosine → constant로 통일) |

→ cubic / trust-region z-schedule 버전도 같은 lr/lmda 그리드로 s70 기준 lmda={0.0005, 0.001, 0.005} 추가 스윕 권장 (이번 세션 cubic-vs-plain 비교에서 쓰던 값).

### GMP NTP+KD (plain, no OPKD)
| sparsity | lr | 비고 |
|---|---|---|
| 50/60/70% | 1e-4 | o65xo2z5 계열 |

### TR-GMP + NTP+KD (KL-gated mask growth)
| sparsity | kl_threshold 후보 | lr | 비고 |
|---|---|---|
| 50/60/70% | 0.005, 0.01, 0.02 | 1e-4 | 기존에 3개 kl 다 테스트했음 — 동일 그리드 유지 |

### TR-GMP + NTP+KD + OPD
(옛 "OPKD (Dense)" 표기를 OPD로 통일. Prev Mask / Dual 변형은 폐기, Empirical Fisher 변형도 폐기.)
| sparsity | kl_threshold | lr |
|---|---|---|
| 50% | 0.005, 0.01, 0.02 | 1e-4 |
| 60% | 0.005, 0.01, 0.02 | 1e-4 |
| 70% | 0.005, 0.01, 0.02 | 1e-4 |

### GMP NTP+KD+OPD (cubic schedule)
| sparsity | 비고 |
|---|---|
| 50% | cubic schedule, lr=1e-4 (기존 32768 step → 이번엔 4096 step 예산으로 축소) |

### ALPS / SparseLLM (재빌드 데이터로 재실행만 하면 됨, 하이퍼파라미터 동일)
| sparsity | ALPS rho/nsamples | SparseLLM |
|---|---|---|
| 50/60/70%, 2:4 | one-shot, 기존과 동일 설정 | one-shot, 기존과 동일 설정 |

---

## Qwen3-4B (`elsa_qwen3_4b` / `reasoning_qwen3_4b`)

### ELSA NTP-ADMM (plain)
| sparsity | lr | lmda | lmda_schedule | interval | 비고 |
|---|---|---|---|---|---|
| 50% | 5e-5 | 1e-3 | **constant** | 32 | ng28it84 (기존 cosine → constant로 통일) |
| 60% | 5e-5 | 5e-3 | **constant** | 32 | fqcfkllx (기존 cosine → constant로 통일) |
| 70% | 1e-4 | 5e-3 | **constant** | 32 | 0uqv2ogw (기존 cosine → constant로 통일) |

### GMP NTP+KD (plain)
| sparsity | lr |
|---|---|
| 50/60/70% | 1e-4 |

### TR-GMP NTP+KD
| sparsity | kl_threshold | lr |
|---|---|---|
| 50% | 0.01, 0.02 | 1e-4 |
| 60% | 0.01, 0.02 | 1e-4 |
| 70% | 0.01, 0.02 | 1e-4 |

### TR-GMP NTP+KD + OPD
(옛 "OPKD (Dense)" 표기를 OPD로 통일. dwup/dwup+lasso 계열은 폐기.)
| sparsity | kl_threshold | lr |
|---|---|---|
| 50% | 0.005, 0.01, 0.02 | 1e-4 |
| 60% | 0.01, 0.02 | 1e-4 |
| 70% | 0.01, 0.02 | 1e-4 |

### GMP NTP+KD+OPD (plain schedule)
| sparsity | lr |
|---|---|
| 50/60/70% | 1e-4 (기존 8192 step) |

### SparseGPT + Sparse SFT (NTP, fixed mask)
| sparsity | 비고 |
|---|---|
| 50/60% | 4096 step, fixed mask — SparseGPT 자체는 안 돌려도 되지만 이 SFT 단계는 OT80/FW20 학습 데이터를 쓰므로 재실행 필요 |

### ALPS / SparseLLM
| sparsity | 비고 |
|---|---|
| 50/60/70%, 2:4 | one-shot, 기존과 동일 설정으로 재빌드 데이터만 교체 |

---

## Qwen3-8B (`reasoning_qwen3_8b`... 실제로는 `reasoning_qwen3_4b` project로 기록됨)

### GMP NTP+KD (pure)
| sparsity | lr |
|---|---|
| 50/60/70% | 1e-4 |

### ALPS
| sparsity | 설정 |
|---|---|
| 50/60/70% | rho=300, nsamples=128 |

### SparseLLM
| sparsity | 비고 |
|---|---|
| 50/60/70% | one-shot, 기존과 동일 설정 |

(8B는 기존 기록이 대부분 단일 조합만 테스트돼있어서 별도 스윕 없이 그 설정 그대로 4096 step/global batch 16으로 재실행)

---

## 공통 재실행 우선순위 제안
1. **ELSA NTP-ADMM plain** (1.7B/4B, s50/60/70) — 가장 기본 baseline, 먼저 확정
2. **GMP NTP+KD plain** (1.7B/4B/8B) — 마찬가지로 기본 baseline
3. **TR-GMP / TR-GMP+OPD** — kl 그리드 (0.005/0.01/0.02) 재확인, 이번에 고친 `admm_tr_gate_at_target` 반영
4. **ALPS / SparseLLM** — 데이터만 바뀐 거라 빠르게 재실행 가능 (one-shot)
5. **cubic z-schedule (ELSA)** — 이번 세션에서 새로 만든 기능, best lmda(0.005)로 우선 확인

## 폐기된 변형 (재실행 대상 아님)
- TR-GMP + NTP+KD + OPKD의 **Prev Mask**, **Dual** 변형 (Dense만 남기고 **OPD**로 표기 통일)
- **Empirical Fisher** 변형
- **GMP lasso-only** (on-policy lasso 적용)
- **dwup-only / dwup+lasso** (S70 체크포인트 파생 계열)

## 확인 필요 사항
- [x] **SAFE(ADMM+SAM)는 우선 돌리지 않음** (이전에도 collapse, PPL 수백만~수천만) — 보류
- [ ] GMP OPD cubic 32768-step 계열은 새 4096-step 예산에 맞게 스케줄(cubic ramp 길이 `T`, 예: 4096의 절반인 ~2048 정도) 재설계 필요
- [x] **GMP cubic sparsity ramp 시작 시점 버그 수정 완료**: `_cubic_sparsity` 호출 두 군데가 LR warmup 변수를 재사용하면서 `gmp_dense_warmup_steps`를 한쪽 경로에서만 체크하던 불일치를 수정. (중간에 별도 `gmp_cubic_warmup_steps` flag를 만들었다가, LR warmup과 별개로 존재해야 할 이유가 없어서 다시 없애고) 최종적으로 `gmp_dense_warmup_steps` 하나로 cubic ramp 시작 지연까지 통일, 두 호출부 모두 이 값으로 일관되게 게이트.

