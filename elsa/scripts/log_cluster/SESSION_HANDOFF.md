# log_cluster 세션 핸드오프 (2026-08-07 ~ 08-08)

이 서버(log-node01-07 / log-master, 파티션: 3090/A100/A6000/PRO6000/H200)에서 진행한 작업 전체 정리. 다음 세션은 이 파일부터 읽고 시작할 것.

## 이 서버 자체에 대해 알아야 할 것

- 다른 클러스터(`/home1/doyoonkim/projects/elsa/...` 경로를 쓰는 기존 스크립트들)와는 별개의 서버. 이 repo(`onpolicyelsa_code`)는 **여러 서버에서 동시에 git push 중** — 커밋 전 항상 `git fetch && git log HEAD..origin/master`로 확인하고, 있으면 `git pull --rebase`부터 할 것. 충돌 잦음.
- 이 서버용 스크립트는 전부 `elsa/scripts/log_cluster/`에 모아둠 (기존 `elsa/scripts/*.sh`는 다른 클러스터용, 경로 안 맞음).
- conda env `rac`가 이미 세팅되어 있고, requirements.txt와 동기화됨. **반드시 `conda activate rac`로 활성화한 뒤 `python`을 호출할 것** — env의 python 바이너리를 절대경로로 직접 부르면 flash-attn 컴파일 확장이 이 노드의 시스템 glibc(GLIBC_2.32 없음)를 잡아서 즉시 크래시남.
- 공유 HF 캐시: `/home/shared/huggingface` (여러 유저가 같이 씀, 일부 Qwen3 모델 이미 캐시돼 있음). `HF_HOME`을 여기로 잡아두면 좋음.
- `/local-data` 스크래치 없음. 대신 노드 로컬 `/tmp/${USER}/job_${SLURM_JOB_ID}`에 wandb 등 임시파일 저장.
- **PRO6000은 못 씀** — 이 conda env의 PyTorch/vLLM 빌드가 sm_120(Blackwell) 커널을 안 갖고 있어서 `RuntimeError: CUDA error: no kernel image is available for execution on the device`로 무조건 죽음. 워닝이 아니라 실제 실패임.
- A100 두 장 사이는 PCIe(PXB)만 있고 NVLink 없음. FSDP 2-GPU 쓸 때 통신이 PCIe로 감 (H200 단일카드보다 느림).
- 이 세션에서 GPU 파티션별로 자주 꽉 찼음 — H200은 다른 유저(jinseokchung, "ppstudy")가 5개를 계속 점유 중이라 3개만 우리가 쓸 수 있었음.

## 이번 세션에서 고친 코드 버그 (전부 커밋/푸시됨)

1. **`elsa/lib/gkd_admm_trainer.py`**: 데이터셋 pickle 캐시 경로가 `/home1/doyoonkim/...`로 하드코딩돼 있어서 이 서버에서 `PermissionError`. 모듈 파일 위치 기준 상대경로로 고침.
2. **`elsa/main.py` `use_fast=False` → `True`**: 제일 중요한 버그. 토크나이저를 슬로우(순수 파이썬) 모드로 로드하고 있었음. 이것 때문에 (a) 20만 샘플 토큰화가 원래도 극도로 느렸고 (b) `_tokenizer_identity()`가 fast tokenizer에서만 있는 `backend_tokenizer` 속성이 없어서 콘텐츠 해시 대신 `name_or_path`로 캐시 키를 잡아버려서, 모델 크기(1.7B/4B)별로 캐시가 갈라지고 재사용이 전혀 안 됐음. fast/slow 토크나이저가 실제 데이터셋 300개 샘플 기준 동일 출력 내는 것 확인 후 수정.
3. **`elsa/lib/gmp_trainer.py` `candidate_masks()` 메모리 버그**: non-FSDP global-pruning 분기에서 전체 모델 파라미터의 Fisher importance score를 `torch.cat`으로 한번에 이어붙였음 (4B 모델 기준 ~13.5GB 단일 할당). A100(80GB)에서 이미 baseline 메모리(모델+optimizer+vLLM)가 68GB 넘게 차 있는 상태라 OOM. 바로 위 FSDP 분기가 이미 하던 대로(딕셔너리 그대로 순회하며 chunk binary search) 고쳐서 이 burst 자체를 없앰.
4. 그 외 다른 서버 세션이 고친 것들(리베이스로 같이 받음): `lr_scheduler` 기본값 constant_with_warmup→cosine, GMP/OPD teacher가 `--model`과 동일해서 자기 자신을 teacher로 쓰던 버그, `gmp_post_target_steps` 신규 플래그(TR-GMP가 목표 희소도 도달 후 `gmp_mask_interval` 스텝만큼만 더 돌고 멈춤, 기본 -1=mask_interval에 연동).

## 이번 세션에서 만든 스크립트 (전부 이 서버 repo 경로 기준 절대경로: `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/`)

- `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_alps_sparse_ntp_qwen3_1.7b.sh` — ALPS s50pct 체크포인트 → NTP-only, AdamW, fixed mask.
- `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_alps_sparse_ntp_qwen3_1.7b_pgd.sh` — 위와 동일하되 optimizer만 `ActivationMetricProjectedSGD`(신규, 아래 설명). `<LR> [SPARSITY] [LR_SCHEDULER] [GRAD_CKPT] [BATCH_SIZE] [GRAD_ACCUM]` 인자.
- `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_gmp_tr_ntpkd_opd_qwen3_4b.sh` — TR-GMP NTP+KD+OPD(0.33/0.33/0.33 loss mix), dense Qwen3-4B에서 시작, cosine LR. `<SPARSITY> [LR_SCHEDULER] [MASK_INTERVAL]` 인자. **A100에서는 메모리 부족, H200에서만 안정적으로 돌아감** (모델+AdamW+vLLM OPD 엔진 합쳐서 80GB 근처까지 찬 상태이므로).
- `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/slurm_eval_lighteval_only.sh` — 학습+저장은 끝났는데 `eval_full_bench` 도중 멈춘 job을 위한 eval 전용 재실행 스크립트. `HF_HUB_DISABLE_XET=1` 필수 (아래 xet 이슈 참고). 내부에서 `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/eval_full.py`(이 서버 전용 아님, repo 공통)를 호출함.
- `/home/doyoonkim/projects/onpolicyelsa_code/elsa/scripts/log_cluster/SESSION_HANDOFF.md` — 이 파일 자체.
- 이번 세션에서 만들었다가 지운 임시 스크립트: `build_cache_fast_tmp.py`(캐시 미리 빌드용, 1회성이라 삭제함), `db.json`/`db_updated.json` 류(아티팩트 DB 편집용 스크래치, `/tmp/claude-1031/.../scratchpad/`에 있었고 세션 종료 시 사라짐 — git에 없음).

## 관련 데이터/모델 경로

- 데이터셋: `/home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_200k_qwen3_train.jsonl` (7.5GB), `/home/doyoonkim/projects/onpolicyelsa_code/elsa/data/ot3_fineweb_200k_qwen3_opdprompts.jsonl` (835MB) — HF `cosmos1030/ot3-fineweb-200k-qwen3`에서 다운로드해둔 것.
- 학습 결과 체크포인트: `/home/doyoonkim/projects/onpolicyelsa_code/elsa/models/` 아래 (예: `gmp_s50pct_lr0.0001_20260808_024943` 등, 각 row의 `sub` 텍스트에 job 번호로 매핑돼 있음).
- 토큰화 캐시: `/home/doyoonkim/projects/onpolicyelsa_code/elsa/.cache/datasets/6a37c5438de9.pkl` (fast tokenizer 기준, 1.7B/4B 공용, 179207 샘플, 8.4GB) — `use_fast=True` 수정 이후 생성된 올바른 캐시. 이거 지우지 말 것, 다음 실험에서도 그대로 재사용됨.
- 로그: `/home/doyoonkim/projects/onpolicyelsa_code/elsa/logs/` 아래 `{job명}_{SLURM_JOB_ID}.out`.

## 새로 포팅한 optimizer: ActivationMetricProjectedSGD

`elsa/lib/activation_metric_projected_sgd.py` + `elsa/lib/activation_tracker.py` — `/home/doyoonkim/projects/opt_baseline_run/sparsegpt_lib/`에서 포팅. 매 스텝마다 gradient를 "안 잘린(active) 좌표"로만 프로젝션하는데, 그 프로젝션을 활성화 covariance(그룹 크기 4, 즉 2:4 블록 단위) 기준으로 함 — PCG의 post-hoc 보정을 온라인/매스텝 버전으로 만든 것에 가까움. `--gmp_base_optimizer=activation_metric_pgd`로 켬.

포팅하면서 실제로 고친 원본 버그 2개 (opt_baseline_run 쪽 원본에도 있었을 가능성 있음, 거긴 안 고침):
- global forward hook이 위치 인자 없이 호출되는 모듈(`self_attn` 등)에서 `inputs[0]` 접근 시 `IndexError`.
- `fisher.update()`가 `gmp_fixed_mask=true`든 아니든 매 스텝 무조건 호출되는데, 이 optimizer엔 Adam류 state가 없어서 `fisher=None`으로 했다가 `AttributeError`. 그냥 기존 `FisherAccumulator`를 그대로 쓰게 되돌림 (Adam state 없으면 `None` 반환하도록 이미 안전하게 짜여있었음).

## 실험 결과 (전부 wandb + HF Hub 대시보드 아티팩트에 반영)

**대시보드**: https://claude.ai/code/artifact/cc1474a7-5020-41c5-90b4-ea16e1d3f51f ("Qwen3 OT80/FW20 Rerun")
⚠️ 이 아티팩트는 다른 세션도 동시에 수정 중. **업데이트 전 항상 WebFetch로 최신본을 다시 받아서 파싱한 뒤 수정할 것** — 한 번 다른 세션이 복구 작업하면서 내가 넣은 4B row 3개를 통째로 날린 적 있음(이미 재발행해서 복구함).

### 핵심 비교 기준선
| | math500 |
|---|---|
| 1.7B ALPS 원샷 (파인튜닝 없음) | **74.6** |
| 4B ALPS 원샷 (파인튜닝 없음) | **88.4** |

### 1.7B s50, ALPS 체크포인트에서 시작해서 NTP-only로 추가 학습 (fixed mask)
| optimizer / lr | math500 | gpqa | ifeval | lcb | gsm8k | ALPS 원샷 대비 |
|---|---|---|---|---|---|---|
| AdamW | 64.6 | 32.3 | 42.1 | 7.5 | 63.1 | -10.0 |
| PGD lr=0.03 | 69.8 | 22.2 | 35.7 | 6.7 | 72.2 | -4.8 |
| PGD lr=0.01 | 69.8 | 27.3 | 35.7 | 6.7 | 67.3 | -4.8 |
| PGD lr=0.003 (batch=8) | 73.6 | - | - | - | - | -1.0 |
| **PGD lr=0.001 (batch=8)** | **75.6** | - | - | - | - | **+1.0 (유일하게 이김)** |

lr이 낮을수록 좋아지는 추세. lr=0.003/0.001은 zero-shot/gpqa/ifeval/lcb/gsm8k 아직 안 나왔을 수 있음 — wandb `reasoning_qwen3_1.7b` 프로젝트에서 run id `q2hbtz1k`(0.003 재시도), 마지막 재제출은 H200 batch=8로 성공(job 40432/40433, wandb run은 로그 확인 필요).

### 4B s50/s60/s70, dense에서 TR-GMP로 NTP+KD+OPD(0.33 each), cosine LR
| mask_interval | s50 | s60 | s70 |
|---|---|---|---|
| 32 (기본) | **77.2** | 64.0 | 39.8 |
| 8 | 71.2 (s50만 완료) | 진행중(job 40446) | 진행중(job 40447) |

mask_interval=32가 8보다 나음 (1.7B에서 봤던 "16이 32보다 낫다"는 것과 반대 결과 — 사이즈/조합마다 다른 듯).

### 결론 (사족 없이)
- **ALPS 원샷을 이긴 건 딱 하나, PGD lr=0.001**. 나머지는 전부 ALPS 원샷보다 못함 (1.7B든 4B든).
- OPD(on-policy 생성) 끼면 대체로 더 나빠짐 — 노이즈가 마스크를 흔드는 것으로 추정.
- PCG(one-shot이든 sequential이든)도 뚜렷한 개선 없음.
- 지금까지 나온 것 중 "추가 학습이 ALPS보다 나은 유일한 레시피"는 PGD optimizer + 낮은 lr. 더 낮은 lr(0.0003 등)도 시도해볼 가치 있음.

## 진행 중이던 job (세션 종료 시점)

- 40446 (H200, 4B s60, mask_interval=8): 학습 중 (step ~330/2048)
- 40447 (H200, 4B s70, mask_interval=8): 학습 중 (step ~305/2048)
- 40438 (H200, 4B s50, mask_interval=8): eval 거의 끝남 (math500=71.2 이미 나옴, 나머지 벤치마크 진행중이었음)

다음 세션은 `squeue -u doyoonkim`으로 상태 확인부터 할 것.

## 잡다한 참고

- H200 GPU 메모리: 143GB (H200 NVL). A100: 79GB. 3090: 24GB.
- 3090에서 PGD optimizer 쓸 땐 `gmp_gradient_checkpointing=true` 필요 (24GB로는 그냥 두면 OOM).
- 4B TR-GMP NTP+KD+OPD 조합은 A100(80GB)에서 vLLM OPD 엔진 + AdamW + 모델 다 얹으면 기본 메모리가 79GB 근처까지 차서 candidate_masks 버그를 고쳐도 여전히 빡빡함 — H200 아니면 힘듦.
- LiveCodeBench(`lighteval/code_generation_lite`, subset `v4_v5`) 데이터셋은 HuggingFace Hub의 "xet" 전송 방식이 이 서버 네트워크에서 무한 대기함. `HF_HUB_DISABLE_XET=1` 필수. 이미 `/home/shared/huggingface`에 미리 받아둠 (268 샘플).
