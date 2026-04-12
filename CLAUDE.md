# Claude Code Guidelines

## 프로젝트 개요

- **elsa/**: ADMM 기반 LLM pruning + training (NTP-CoT, Hybrid KD)
- **RAC/open-r1-main/**: SparseGPT pruning + GRPO baseline
- 모델: Qwen3-0.6B, 평가: MATH-500, 로깅: wandb (project: elsa_qwen3_0.6b)
- SLURM 클러스터: A100-80GB, partition=A100-80GB, qos=hpgpu
- wandb entity: `dyk6208-gwangju-institute-of-science-and-technology`
- HF 계정: `cosmos1030` (토큰: `~/.hf_token`)

---

## Git 커밋 규칙

- **서로 다른 성격의 변경사항은 반드시 커밋을 분리한다.**
  - 예: sweep config 수정 + grpo.py 버그 픽스 → 커밋 2개
  - `elsa/`와 `RAC/` 파일이 섞이면 반드시 분리
- 커밋 메시지 prefix:
  - `fix:` 버그 픽스
  - `feat:` 새 기능
  - `chore:` 설정/config 변경
  - `docs:` 문서
- `git add -A` / `git add .` 사용 금지 — 파일 명시적으로 지정
- push 전 secret (API 키, 토큰 등) 포함 여부 확인
- 사용자가 요청할 때만 커밋/push (자의적으로 하지 말 것)

---

## SLURM / wandb sweep 규칙

- sweep yaml name에 method + sparsity 포함 (e.g. `sweep_ntp_cot_qwen3_50pct`)
- sweep 제출 전 yaml에서 반드시 확인:
  - `sparsity_ratio` 값이 의도한 값인지
  - `admm_lr`, `admm_lmda` range가 맞는지
  - `push_to_hub: true` 포함 여부
- HF 토큰 스크립트 하드코딩 금지 → `export HF_TOKEN=$(cat ~/.hf_token 2>/dev/null || echo "")`
- sweep 에이전트 수 = grid 크기 (3 lr × 3 lmda = 9)
- 이미 실행 중인 sweep을 수정해야 하면 → 기존 sweep 취소 후 새로 생성

---

## 코드 관련 주의사항

### main.py / flags
- sweep에서 wandb.config로 FLAGS를 override할 때 기본값이 0이 아닌 flag는 체크가 필요
  - 예: `admm_final_lmda` 기본값 0.01 → sweep에서 `admm_lmda`만 넘기면 덮어쓰기 안됨
  - 현재 fix: `FLAGS.admm_final_lmda = FLAGS.admm_lmda` 항상 적용

### gkd_admm_trainer.py (Hybrid NTP+KD)
- `MathCotKDDataset`: `text` 필드에서 `<think>` 태그 기준으로 split
  - `\n\n` 기준 split은 문제 텍스트 안에도 `\n\n`이 있어서 잘못 split될 수 있음
- `prompt_len`은 batch 내 최대값 (batch_size=1이라 현재 문제 없음)
- Top-K KL: student가 teacher top-k 바깥에 확률 몰아도 gradient 없음 (알려진 한계)

### optimizers.py (ADMM)
- cosine schedule: `init_lmda=0` → `final_lmda=admm_lmda`로 스케줄
- `lmda_default`는 non-constant schedule에서 `init_lmda`로 설정됨 (정상)

### grpo.py (RAC)
- vLLM eval 전 `del model; gc.collect(); torch.cuda.empty_cache()` 필수 (OOM 방지)
- `from pathlib import Path` import 필수

---

## 데이터 경로

- CoT 데이터: `/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl`
- 모델 캐시: `/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca`
- 모델 저장: `/home1/doyoonkim/projects/elsa/models/`

---

## 디스크 관리

- 개인 쿼터: 5000G (현재 ~1000G 사용 중)
- `/home1` 전체: 76% 사용 중 (90% 초과 시 위험)
- 모델 저장이 run마다 ~2-3G씩 쌓이므로 sweep 완료 후 불필요한 모델 정리 필요

---

## 현재 진행 중인 실험 (2026-04-12 기준)

### ELSA sweeps (elsa_qwen3_0.6b project)
| Method | Sparsity | Sweep ID | 상태 |
|--------|----------|----------|------|
| NTP-CoT | 30% | `3cwctrth` | 완료 |
| NTP-CoT | 40% | `4uomhpoi` | 실행 중 |
| NTP-CoT | 50% | `3cwctrth` | 완료 |
| NTP-CoT | 60% | `pxdzadaf` | 실행 중 |
| NTP-CoT | 70% | `ff244k9g` | 실행 중 |
| Hybrid KD | 30% | `1qnq3a6p` | 완료 (admm_lmda 버그 있던 버전) |
| Hybrid KD | 30% | `1qnq3a6p` (재실행) | 실행 중 |
| Hybrid KD | 40% | `2084ue9q` | 실행 중 |
| Hybrid KD | 50% | `x8v8ms0a` | 실행 중 |
| Hybrid KD | 60% | `9ege62mf` | 실행 중 |
| Hybrid KD | 70% | `ujbxnhpl` | 실행 중 |

### RAC sweeps
- CoT calibration 30-70%: 완료
- Prompt-only calibration 30-70%: 완료

### 알려진 버그 (수정 완료)
- `admm_final_lmda` sweep에서 항상 0.01로 고정되던 버그 → fix 커밋 `060c388`
- `math_cot` 데이터에서 `\n\n` 대신 `<think>` 기준으로 split → fix 완료
- wandb sparse eval metrics hybrid run에서 안 나오던 버그 → `eval_strategy="steps"` fix 완료
