# GMP + DPO Version History

## 공통 설정 (전 버전 동일)
- 모델: DeepSeek-R1-Distill-Qwen-1.5B
- chosen: dense model 생성 (pre-cached)
- rejected: post-mask sparse model 생성 (mask update마다 갱신)
- ref: pre-mask model snapshot (pruning-aware)
- 학습: `gmp_steps=1024`, `lr=1e-4`, `mask_interval=32`
- 평가: math500 pass@1
- NTP baseline: s50=**0.704**, s70=**0.410** (sweep `9002rmn3`/`9ryq4acj`)

---

## v1 — 초기 구현 (실패)
**sweep**: `n20yennc` (s50), `bz9q3x01` (s70)

| 버그 | 내용 |
|------|------|
| ref model = dense | ref를 학습 시작 전 한 번만 deepcopy → ref ≈ dense model |
| EOS 강제 비활성화 | `min_new_tokens=512` → 자연 종료 없이 쓰레기 토큰 생성 |
| sum logprob | `beta=0.1 × sum(512 tokens) ≈ 150` → sigmoid 포화 |

**결과**: s50 best=0.682, s70 best=0.428 (baseline 이하)

---

## v2 — 3개 버그 수정
**sweep**: `le1koc0e` (s50), `uva7dlgt` (s70)

| 수정 | 내용 |
|------|------|
| ref 갱신 | mask_interval마다 ref deepcopy |
| EOS 허용 | `eos_token_id` 복원, 실제 cont_mask 사용 |
| average logprob | `sum(logp) / seq_len` |

**남은 버그**: pad==eos mask 버그 (DeepSeek tokenizer: pad_id==eos_id=151643 → EOS 토큰 mask에서 0으로 처리 → NaN DPO loss)

**결과**: dpo_acc=0 throughout (NaN 스킵으로 DPO 미적용)

---

## v3 — pad==eos 버그 수정 + Pruning-aware DPO
**sweep**: `yi0rn47u` (s50), `o204zd6i` (s70)

| 수정 | 내용 |
|------|------|
| pad==eos 버그 | cumsum trick으로 EOS 이후 padding 처리 |
| Pruning-aware ref | ref = mask update **이전** deepcopy (π_{k-1}) |
| rejected | mask update **이후** 생성 (π̃_k) |

**파라미터**: `beta=0.1`, `lambda=[0.01, 0.03, 0.1]`, `max_new_tokens=512`

**실패 원인**: `dpo_loss ≈ log(2) = 0.693` 상수, `std ≈ 0.0002`
→ average logprob 사용하면 margin scale이 seq-sum 대비 ~T배 (T≈512) 작아짐
→ `beta=0.1, lambda=0.01`일 때 DPO gradient가 NTP 대비 **~5만배 약함**

**결과**: s50 best=0.692, s70 best=0.444 (NTP와 사실상 동일, DPO 미작동)

---

## v4 — beta/lambda scale 검증 (256-step sanity)
**sweep**: `ytqqhwpm` (s70, 256 steps, math500 없음)

| 변경 | 내용 |
|------|------|
| beta | 0.1 → {3, 10} |
| lambda | 0.01 → {0.1, 0.3} |

**결과**: margin ±0.4~0.9 확인 (v3: ±0.001) → scale 문제 해결 확인  
최종 margin은 ~0으로 수렴 (sawtooth + pruning confounder)  
math500 없어서 성능 판단 불가

---

## v5 — beta/lambda 수정 + full sweep
**sweep**: `95ybfbqx` (s50), `g4szl8ck` (s70)

| 변경 | 내용 |
|------|------|
| beta | {3, 10} |
| lambda | {0.1, 0.3} |
| max_new_tokens | **512** (v3와 동일) |
| dpo_start_step | 0 (처음부터 DPO 적용) |

**결과**:

| sparsity | beta | lambda | math500 | vs baseline |
|----------|------|--------|---------|------------|
| s50 | 3 | 0.3 | **0.692** | -0.012 |
| s50 | 10 | 0.3 | 0.676 | -0.028 |
| s50 | 10 | 0.1 | 0.666 | -0.038 |
| s50 | 3 | 0.1 | 0.664 | -0.040 |
| s70 | 10 | 0.1 | **0.444** | **+0.034** ✅ |
| s70 | 3 | 0.1 | 0.428 | +0.018 |
| s70 | 10 | 0.3 | 0.396 | -0.014 |
| s70 | 3 | 0.3 | 0.394 | -0.016 |

**실패 원인 (Δ_T diagnostic 결과)**:
- `max_new_tokens=512`는 짧아서 고sparsity에서 rejected가 일찍 EOS → avg logprob 오히려 높아짐
- pair separability (pos_rate) inverted-U: tok=512에서 sparsity>65% 구간 pair 품질 붕괴
- `dpo_start_step=0`으로 sparsity<15% 구간(pos_rate<0.5)에서도 노이즈 pair로 학습

---

## v6 — max_new_tokens 수정 (진행 중)
**sweep**: `oa9x4zhu` (s50), `1c3vwx0g` (s70)

| 변경 | 내용 |
|------|------|
| max_new_tokens | 512 → **1024** |
| beta | {3, 10} |
| lambda | {0.1, 0.3} |
| dpo_start_step | 0 |

**근거 (Δ_T v2 diagnostic)**:
- tok=1024: 전 sparsity 구간에서 pos_rate ≥ 0.5 (평균 0.61), 안정적
- tok=512: sparsity>65%에서 pos_rate 0.35~0.45로 급락
- tok=2048: 비슷하지만 compute 2배 → tok=1024가 최적

**결과**:

| sparsity | beta | lambda | math500 | vs baseline |
|----------|------|--------|---------|------------|
| s50 | 10 | 0.3 | 0.700 | -0.004 |
| s50 | 3 | 0.3 | 0.676 | -0.028 |
| s50 | 3 | 0.1 | 0.666 | -0.038 |
| s50 | 10 | 0.1 | **0.084** | -0.620 ⚠️ 붕괴 |
| s70 | 3 | 0.1 | **0.444** | **+0.034** ✅ best |
| s70 | 10 | 0.1 | 0.418 | +0.008 ✅ |
| s70 | 3 | 0.3 | 0.414 | +0.004 ✅ |
| s70 | 10 | 0.3 | **0.168** | -0.242 ⚠️ 붕괴 |

**패턴**: beta=10은 sparsity에 따라 붕괴 위치 예측 불가 → beta=3만 안전  
DPO가 s70에서만 baseline 이상. s50은 전부 미달.  
**다음**: beta=3, lam=0.1 고정으로 lr sweep (s70)

---

## 남은 실험 후보
- **v6 lr sweep**: v6 결과 확인 후 최적 beta/lambda로 lr={5e-5, 1e-4, 2e-4} 추가
- **soft/gated DPO**: Δ_T 기반 pair confidence weight q_T = σ(α·Δ_T) 적용
