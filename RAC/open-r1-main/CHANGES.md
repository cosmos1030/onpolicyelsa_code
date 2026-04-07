# Change Log

## 2026-03-16

### Bug: Prompt+CoT pruning이 실제로는 Prompt-only로 동작하던 문제 수정

**배경**

README의 "Prompt + CoT pruning" 섹션대로 trace 데이터셋을 `--dataset_name`으로 넘겨도
CoT가 calibration에 포함되지 않아 논문 결과 재현이 안 되는 문제 발견.

원인은 두 가지:
- Arrow 데이터셋(`dataset_..._trace_...`)의 `prompt` 컬럼은 문제 텍스트만 담긴 chat dict 형식이라
  `make_calib_loader` → `_row_to_prompt()`가 문제만 추출하고 CoT(`generations` 컬럼)는 무시함.
- JSONL(`dataset_..._trace_..._.jsonl`)의 `prompt` 컬럼은 문제+CoT가 합쳐진 string이라
  이걸 쓰면 CoT가 포함되지만, 로드 자체가 안 됐고 로드되더라도 `to_conversation()`이
  이중 포맷팅을 일으켜서 동작하지 않았음.

**수정 내용**

#### `src/open_r1/utils/data.py` — `_safe_load_dataset()`
- JSONL/JSON 파일 경로를 직접 주면 `load_dataset('json', data_files=...)` 로 처리하도록 추가
- 기존 try/except가 `ValueError`만 잡았는데 `FileNotFoundError`도 catch하도록 수정

#### `src/open_r1/grpo.py` — `to_conversation()` 호출부
- `if not training_args.prune:` 조건 추가
- pruning 시에는 `to_conversation()`을 건너뜀
- JSONL의 `prompt`(이미 완성된 string)가 user 메시지로 이중 포맷팅되는 문제 방지
- 학습/trace 동작에는 영향 없음

**사용법**

Prompt+CoT pruning 시 `--dataset_name`에 Arrow가 아닌 JSONL 경로를 넘길 것:
```bash
--dataset_name /path/to/dataset_..._trace_OpenR1-Math-220k_.jsonl
--dataset_prompt_column prompt
```