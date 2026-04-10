# Claude Code Guidelines

## Git 커밋 규칙

- **서로 다른 성격의 변경사항은 반드시 커밋을 분리한다.**
  - 예: sweep config 수정 + grpo.py 버그 픽스 → 커밋 2개
  - 파일 경로가 다른 레포(elsa/, RAC/)에 걸쳐있으면 특히 분리
- 커밋 메시지는 변경 내용을 정확히 반영할 것
  - `fix:` 버그 픽스
  - `feat:` 새 기능
  - `chore:` 설정/config 변경
  - `docs:` 문서
- `git add -A` 또는 `git add .` 사용 금지 — 파일을 명시적으로 지정할 것

## SLURM / wandb sweep 규칙

- sweep yaml에 sparsity 정보를 name에 포함할 것 (e.g. `sweep_ntp_cot_qwen3_50pct`)
- sweep 제출 전 yaml의 핵심 하이퍼파라미터 (sparsity_ratio, admm_lr, admm_lmda) 반드시 확인
- HF 토큰은 스크립트에 하드코딩 금지 → `~/.hf_token`에서 읽을 것

## 코드 변경 규칙

- 변경사항이 있으면 사용자가 요청 시 즉시 GitHub에 push
- push 전 secret (API 키, 토큰 등) 포함 여부 확인
