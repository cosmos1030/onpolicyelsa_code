# GMP Unstructured Sparsity Experiments

**Model**: DeepSeek-R1-Distill-Qwen-1.5B  
**Eval**: MATH-500 500샘플, max_new_tokens=8192  
**wandb project**: `gmp_qwen3_1.5b_v2`

---

## Baselines

| Method | Sparsity | MATH-500 | Sweep |
|--------|----------|----------|-------|
| Dense (no pruning) | 0% | 0.816 | — |
| Dense fine-tuned (NTP, seq=2048) | 0% | 0.766 | — |
| **GMP NTP** | **50%** | **0.704** | `9002rmn3` |
| **GMP NTP** | **70%** | **0.410** | `9ryq4acj` |

---

## 실험 결과 요약

### s50% 실험

| Method | MATH-500 | Sweep | 비고 |
|--------|----------|-------|------|
| NTP baseline | 0.704 | `9002rmn3` | lr=1e-4 |
| On-policy KD (MiniLLM) | 0.694~0.724 | `k7phv4lt` | lr=1e-4, kd_lam=2 |
| OPKD + DPO | **0.728** | v7 sweep | NTP baseline 넘어섬 |

### s70% 실험

| Method | MATH-500 | Sweep | 비고 |
|--------|----------|-------|------|
| NTP baseline | 0.410 | `9ryq4acj` | lr=1e-4 |
| On-policy KD (MiniLLM) | 0.444~0.486 | v6/`jctpv3gs` | lr=1e-4 |
| OPKD + DPO | 0.470 | v7 sweep | OPKD 단독보다 -0.016 |
| OPKD + plain L1 (1e-4) | 0.524 | `zjoe9q6l` | OPKD best + L1 추가 |
| NTP + inv_fisher_sqrt L1 | **0.448** | `m319jrn1` | NTP only +0.038 |

---

## L1 Regularization 실험 (s70 NTP 기준)

### inv_fisher_sqrt L1 (sweep `m319jrn1`)

$$R = \lambda \cdot \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \frac{|w_i|}{\sqrt{\text{clamp}(\tilde{f}_i, 0.1, 10.0)}}$$

Fisher 작은 weight (pruning 후보) → penalty 강, Fisher 큰 weight (중요) → penalty 약

| lambda | MATH-500 |
|--------|----------|
| **1e-5** | **0.448** |
| 1e-4 | 0.406 |
| 5e-4 | 0.424 |
| 1e-3 | 0.380 |

→ NTP baseline (0.410) 대비 best +0.038. lambda 작을수록 좋음.

### plain L1 비교 (s70)
- OPKD + plain L1 1e-4: 0.524 (`zjoe9q6l`) — OPKD 위에 L1 추가
- NTP + plain L1: 미실험 (비교용 run 필요)

---

## On-Policy KD 상세 (MiniLLM-style)

- **구조**: NTP 매 step + reverse KL on-policy rollout 8 step마다 1회
- **s50 best config**: lr=1e-4, kd_lambda=2, onpol_interval=8, temp=1, topk=100
- **s70 best config**: lr=1e-4, kd_lambda=1

## DPO 상세 (Pruning-aware DPO)

- **아이디어**: ref = mask update 이전 snapshot, rejected = mask update 이후 generation
- **s50**: +0.034 (OPKD 위에서 추가 개선, NTP baseline 넘어섬)
- **s70**: -0.016 (OPKD보다 소폭 하락)
- **DPO 유효 구간**: sparsity 25~60% (Δ_T 분석, inverted-U)

---

## TODO

- [ ] NTP + plain L1 s70 단독 run (inv_fisher_sqrt와 fair 비교)
- [ ] inv_fisher_sqrt L1 lambda=1e-5로 OPKD 위에 붙이기
- [ ] s50에서 inv_fisher_sqrt L1 실험
