# Experiment Log

모델: **Qwen3-0.6B**, 태스크: **MATH-500**, 방법: **ELSA (ADMM-based joint pruning+training)**

---

## Baselines (from RAC paper)

| 모델 | Sparsity | 방법 | MATH-500 |
|---|---|---|---|
| DS-R1-Distill-1.5B | 0% | Dense | 0.832 |
| DS-R1-Distill-1.5B | 50% | SparseGPT (C4) | 0.356 |
| DS-R1-Distill-1.5B | 50% | SparseGPT (prompt-only) | 0.496 |
| DS-R1-Distill-1.5B | 50% | RAC (prompt+CoT) | 0.664 |
| DS-R1-Distill-7B | 0% | Dense | 0.936 |
| DS-R1-Distill-7B | 50% | RAC (prompt+CoT) | 0.900 |
| Qwen3-0.6B | 0% | Dense | **0.726** (job 299607, 2026-04-06) |

---

## 2026-04-06 | ELSA NTP-CoT Sweep

**목적:** CoT-only 데이터(`math_220k_cot.jsonl`)로 NTP 훈련 시 성능 확인.
기존 elsa NTP 실험은 prompt+CoT 전체를 썼는데, CoT 부분만 쓰면 더 dense한 reasoning 신호 기대.

**설정:**
- 모델: Qwen3-0.6B (50% unstructured sparsity)
- 데이터: `elsa/data/math_220k_cot.jsonl` (CoT만 추출)
- ADMM: steps=1024, interval=32, BS=8, ACCUM=1
- Sweep grid: lr ∈ {0.1, 0.01} × λ ∈ {5e-5, 5e-4}

**wandb sweep:** `igmwmiyd` (project: elsa_qwen3_0.6b)
**slurm jobs:** 309627, 309628, 309629, 309630

**이전 실패:**
- sweep 9flioyoy: `command` 블록에서 `${env}` 변수 처리 안됨 → python path 못찾아 crash
- sweep ynnbj9tq: `/usr/bin/env python`으로 실행 → numpy 없음 crash
- sweep 1ts6fpg5: HF ID `Qwen/Qwen3-0.6B` → transformers 버전 문제로 Unrecognized model error
- **Fix:** `command`에 절대경로 python 지정 + model을 로컬 캐시 경로로 변경

**결과:** (TBD)

---

## 2026-04-05 | KD-ADMM (On-Policy KD) 속도 문제

**문제:** generate() 8번/step (BS=1, ACCUM=8) → 584s/step, ETA 166h
**원인 1:** model.config.use_cache=False 전역 설정이 generate() 시에도 KV cache 비활성화
**원인 2:** gradient checkpointing과 KV cache 충돌 (Qwen3DecoderLayer)
**Fix:**
- `gkd_admm_trainer.py`: generate() 전후 gc_disable/enable, use_cache=True/False
- BS=8, ACCUM=1로 변경 시 generate() 1번/step → ~8× 속도 개선 예상

**KD-ADMM job 288458:** cancelled (위 문제로)

---

## TODO / 다음 실험

- [ ] NTP-CoT sweep 결과 확인 후 best hp로 longer run
- [ ] On-Policy KD sweep (KD-ADMM, BS=8) — NTP 대비 성능 비교
- [ ] 다양한 sparsity (20/30/40/50%) ablation
- [ ] DS-R1-Distill-1.5B / 7B 동일 실험 (RAC 논문 Table 직접 비교)