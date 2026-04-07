# RAC 실험 기록

## 목표
DeepSeek-R1-Distill-Qwen-1.5B을 50% sparse로 프루닝한 후 성능을 회복시키는 것.
평가: MATH-500 pass@1

---

## 베이스라인

| 모델 | MATH-500 | 비고 |
|------|----------|------|
| Dense (원본) | **0.832** | deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B |
| SparseGPT pruned (50%) | **0.692** | 프루닝 후 fine-tuning 없음, trace 263샘플 calibration |

---

## GKD 실험 (On-policy Distillation)

학습 설정 공통:
- Student: SparseGPT 50% sparse 모델
- Teacher: Dense 원본 모델
- Loss: Reverse KL (student‖teacher)
- max_new_tokens=4096, temperature=1.0

| 실험 | 옵티마이저 | 데이터 | LR | 스텝 | MATH-500 | 비고 |
|------|-----------|--------|-----|------|----------|------|
| gkd_trace_lr2e5 | MaskedAdam | trace 263샘플 | 2e-5 | 1 epoch | **0.626** | baseline보다 하락 |
| gkd_lr1e-6_adam | Adam (dense) | 220k | 1e-6 | 1 epoch | **0.714** | sparsity 소실, 성능은 향상 |
| gkd_lr1e-3_ProjectedMuon | ProjectedMuon | trace 263샘플 | 1e-3 | 1 epoch | **0.702** | sparsity 유지됨 (train_sparsity≈0.434) |
| gkd_220k_ProjectedMuon | ProjectedMuon | 220k | 1e-3 | 300 steps | **진행 중 (job 247118)** | ~71h 예상, step 85/300 완료 |

### 핵심 발견
- **MaskedAdam**: sparsity는 유지되지만 성능 오히려 하락 → 최적화 문제
- **Adam (dense)**: sparsity 소실되지만 0.714로 GKD 자체는 동작함을 확인
- **ProjectedMuon**: sparsity 유지 + 0.702로 baseline(0.692) 대비 성능 향상 확인
- DeepSpeed ZeRO-2와 custom optimizer 충돌 → `SparseMaskCallback`으로 해결 (on_step_end에서 마스크 재적용)

---

## ELSA Baseline (ADMM Pruning)

- 방법: ADMM 기반 constrained optimization으로 pruning + training 동시 진행
- 데이터: math_trace JSONL (528샘플, `completion` 컬럼 사용)
- 설정: sparsity=0.5, admm_steps=2000, lr=2e-4, lmda=0.01, interval=32, seqlen=2048
- 모델 저장: `elsa/models/DeepSeek-R1-Distill-Qwen-1.5B_pruned0.5_admm_lr0.0002_20260323_2032`
- 평가: **진행 중 (job 247997)** — config.json architecture 이름 버그 수정 후 재평가

---

## 현재 진행 중인 작업

| Job | 내용 | 상태 |
|-----|------|------|
| 247118 | GKD 220k + ProjectedMuon 300 steps | RUNNING (n49, ~50h 남음) |
| 247997 | ELSA pruned 모델 MATH-500 평가 | RUNNING |

---

## 비교 요약 (완료된 실험)

```
Dense:              0.832  ████████████████████████████████
SparseGPT (50%):    0.692  ██████████████████████████
GKD+ProjectedMuon:  0.702  ███████████████████████████  ← sparsity 유지
GKD+Adam(dense):    0.714  ████████████████████████████  ← sparsity 소실
GKD+MaskedAdam:     0.626  █████████████████████████
ELSA(ADMM):         TBD
```
