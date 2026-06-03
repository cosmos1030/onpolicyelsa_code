# 2:4 Semi-Structured Sparsity Experiments

**Model**: DeepSeek-R1-Distill-Qwen-1.5B  
**Sparsity**: 50% (2:4 — 각 group-of-4에서 최대 2개 pruning)  
**Eval**: MATH-500 500샘플, max_new_tokens=8192  
**wandb project**: `gmp_24_semi_structured`

---

## 구현

- **GMP 2:4**: `elsa/lib/gmp_trainer.py`의 `GradualMaskManager`에 N:M masking 추가
  - `competition codebase` (`log_efficient_qwen_competition/lib/gmp.py`)에서 이식
  - `sparsity_type="2:4"` → `prune_n=2, prune_m=4`
  - Cubic gradual schedule: step 0 (0%) → step T (50%), 각 update마다 per-layer N:M mask 재계산
  - Top-N per group 보호 → 나머지에서 global threshold pruning
- **SparseGPT 2:4**: RAC `grpo.py`의 `--prune_N=2 --prune_M=4` 플래그 사용 (기존 지원)
- **SparseGPT 2:4 + Retrain**: SparseGPT pruned model → `gmp_fixed_mask=true`로 mask 고정 후 retrain
- **GMP 2:4 structured L1**: training 중 bottom-2 per group에 L1 penalty — `gmp_l1_lambda` 플래그, mean-normalized, already-pruned 제외
- **On-policy KD (MiniLLM-style)**: reverse KL, on-policy rollout, `gmp_onpolicy_pg=true`, 8 steps마다 1회

---

## 실험 현황

| # | Method | Sweep | 상태 | Best MATH-500 |
|---|--------|-------|------|--------------|
| 1 | GMP 2:4 NTP (lr sweep) | `9e1nt3qg` | ✅ 완료 | 0.344 (lr=2e-4) |
| 2 | SparseGPT 2:4 one-shot | `ee91pxth` | ✅ 완료 | 0.370 |
| 3 | SparseGPT 2:4 + Retrain NTP | `41nql157` | ✅ 완료 | 0.542 (lr=2e-4) |
| 4 | SparseGPT 2:4 + Retrain OPKD | `pmcsozwk` | ✅ 완료 | 0.550 (lr=2e-4, kd=1) |
| 5 | GMP 2:4 NTP + L1 reg | `cv3f7v9l` | ✅ 완료 | 0.334 (l1=1e-4) |
| 6 | GMP 2:4 NTP + OPKD | `oxuyfsii` | ✅ 완료 | 0.390 (lr=1e-4, kd=4) |
| 7 | SparseGPT 2:4 + Retrain chunk GRPO | `870vg461` | ✅ 완료 | 0.552 (lr=2e-4, grpo=0.1) |

---

## 결과 상세

### 1. GMP 2:4 NTP (sweep `9e1nt3qg`)

| lr | MATH-500 |
|----|----------|
| 1e-5 | 0.220 |
| 5e-5 | 0.288 |
| 1e-4 | 0.332 |
| **2e-4** | **0.344** |
| 5e-4 | 0.180 (발산) |

### 2. SparseGPT 2:4 one-shot (sweep `ee91pxth`)

calibration: math_cot 1M tokens

| MATH-500 | PPL (wikitext2) | PPL (c4) |
|----------|-----------------|----------|
| **0.370** | 199.6 | 201.9 |

### 3. SparseGPT 2:4 + Retrain NTP (sweep `41nql157`)

| lr | MATH-500 |
|----|----------|
| 1e-5 | 0.424 |
| 5e-5 | 0.492 |
| 1e-4 | 0.536 |
| **2e-4** | **0.542** |

### 4. SparseGPT 2:4 + Retrain OPKD (sweep `pmcsozwk`)

| lr \ kd_lambda | 1 | 2 | 4 |
|---|---|---|---|
| 5e-5 | 0.502 | 0.526 | 0.500 |
| 1e-4 | 0.530 | 0.548 | 0.510 |
| **2e-4** | **0.550** | 0.538 | 0.518 |

### 5. GMP 2:4 NTP + L1 reg (sweep `cv3f7v9l`)

lr=2e-4 고정, l1_lambda 탐색 — bottom-2 per group에 structured L1 penalty

| l1_lambda | MATH-500 |
|-----------|----------|
| 0.0 (baseline) | 0.304 |
| 1e-5 | 0.332 |
| **1e-4** | **0.334** |
| 1e-3 | 0.302 |

→ 미미한 개선. NTP only 0.344보다도 낮음 (baseline variance 영향 있음)

### 6. GMP 2:4 NTP + OPKD (sweep `oxuyfsii`)

dense model에서 GMP 2:4 pruning + NTP + MiniLLM-style on-policy KD  
lr × kd_lambda grid (3×3=9 runs)

| lr \ kd_lambda | 1 | 2 | 4 |
|---|---|---|---|
| 5e-5 | 0.336 | 0.324 | 0.324 |
| 1e-4 | 0.368 | 0.358 | **0.390** |
| 2e-4 | 0.366 | 0.386 | 0.362 |

→ best 0.390. NTP only (0.344)보다 +0.046 개선이지만 SparseGPT retrain 대비 크게 낮음

---

## 비교 (완료된 실험 기준, 1024 steps)

| Method | MATH-500 |
|--------|----------|
| GMP 2:4 NTP (best) | 0.344 |
| SparseGPT 2:4 one-shot | 0.370 |
| SparseGPT 2:4 + Retrain NTP (best) | 0.542 |
| SparseGPT 2:4 + Retrain OPKD (best) | **0.550** |
| GMP 2:4 NTP + L1 reg | 🔄 |
| GMP 2:4 NTP + OPKD (best) | 0.390 |
| SparseGPT 2:4 + Retrain OPKD (best) | 0.550 |
| SparseGPT 2:4 + Retrain chunk GRPO (best) | **0.552** |

---

### 7. SparseGPT 2:4 + Retrain chunk GRPO (sweep `870vg461`) 🔄

SparseGPT 2:4 pruned model + `gmp_fixed_mask=true`, NTP + chunk GRPO (8 steps마다)  
lr × grpo_lambda grid (2×2=4 runs), chunk_size=16

| lr \ grpo_lambda | 0.1 | 0.5 |
|---|---|---|
| 1e-4 | 0.542 | 0.542 |
| **2e-4** | **0.552** | 0.462 |

→ best 0.552. SparseGPT + OPKD retrain (0.550)과 거의 동일

---

## Pruned Model 경로

- **SparseGPT 2:4**:  
  `/home1/doyoonkim/projects/RAC/open-r1-main/models/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562_pruned_50_all_tokens1000000_prunemethod_SparseGPT_thirds_1_2_3_N2_M4_dataset_DeepSeek-R1-Distill-Qwen-1_trace_OpenR1-Math-220k_.jsonl`

---

## 요약

- SparseGPT one-shot (0.370)이 GMP NTP (0.344)보다 높음 — pruning quality 차이
- SparseGPT + retrain: NTP (0.542) → OPKD (0.550) ≈ chunk GRPO (0.552)
- GMP + OPKD (0.390): NTP only보다 +0.046이지만 SparseGPT retrain 대비 크게 낮음
- L1 reg: 효과 미미 (0.334), 구조적 개선보다 pruning 초기점이 더 중요
- **결론**: SparseGPT로 먼저 pruning 후 retrain하는 것이 GMP보다 압도적으로 유리
