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
| optimizer / lr | math500 | gpqa | ifeval | lcb | gsm8k |
|---|---|---|---|---|---|
| **ALPS 원샷 (기준선, job 689591)** | 74.6 | 31.8 | 46.2 | 8.6 | 72.7 |
| AdamW | 64.6 | 32.3 | 42.1 | 7.5 | 63.1 |
| PGD lr=0.03 | 69.8 | 22.2 | 35.7 | 6.7 | 72.2 |
| PGD lr=0.01 | 69.8 | 27.3 | 35.7 | 6.7 | 67.3 |
| PGD lr=0.003 (batch=8) | 73.6 | 26.3 | 결측(아래) | 3.7 | 71.2 |
| PGD lr=0.001 (batch=8) | **75.6** | 24.2 | 44.0 | 6.3 | 68.8 |

**주의**: lr=0.001이 math500만 74.6→75.6(+1.0)로 ALPS 원샷을 넘지만, gpqa(-7.6)/ifeval(-2.2)/lcb(-2.3)/gsm8k(-3.9)는 전부 ALPS 원샷보다 낮음. math500 하나만 보고 "이겼다"고 하면 안 됨 — 5개 벤치마크 종합으로는 이 recipe도 여전히 퇴보. lr이 낮을수록 math500은 단조 증가(0.03/0.01→69.8, 0.003→73.6, 0.001→75.6)하지만 다른 벤치마크는 그 추세를 따라가지 않음.

lr=0.003(job 40432)은 ifeval 도중 `RuntimeError: CUDA error: an illegal memory access was encountered`로 죽어서 그 항목만 결측 — 재시도 안 하고 4/5 벤치마크로 기록함. lr=0.001(job 40433)은 5개 다 정상 완료. 둘 다 아티팩트에 반영 완료.

### 4B s50/s60/s70, dense에서 TR-GMP로 NTP+KD+OPD(0.33 each), cosine LR
| mask_interval | s50 | s60 | s70 |
|---|---|---|---|
| 32 (기본) | **77.2** | 64.0 | 39.8 |
| 8 | 71.2 (s50만 완료) | 진행중(job 40446) | 진행중(job 40447) |

mask_interval=32가 8보다 나음 (1.7B에서 봤던 "16이 32보다 낫다"는 것과 반대 결과 — 사이즈/조합마다 다른 듯).

### 결론 (사족 없이)
- **어떤 recipe도 ALPS 원샷을 종합적으로 이기지 못함.** PGD lr=0.001(job 40433)이 math500만 74.6→75.6(+1.0)으로 앞섰지만, 나머지 4개 벤치마크(gpqa -7.6, ifeval -2.2, lcb -2.3, gsm8k -3.9)는 전부 ALPS 원샷보다 떨어짐 — math500 단일 지표로만 보면 "이긴 것처럼" 보이지만 전체적으로는 여전히 퇴보임. 1.7B든 4B든 마찬가지.
- OPD(on-policy 생성) 끼면 대체로 더 나빠짐 — 노이즈가 마스크를 흔드는 것으로 추정.
- PCG(one-shot이든 sequential이든)도 뚜렷한 개선 없음.
- PGD optimizer + 낮은 lr이 5개 벤치마크 평균으로는 그나마 ALPS 원샷과 가장 가깝게 좁혀지는 방향이긴 하지만, "이겼다"고 말할 수 있는 recipe는 세션 종료 시점까지 없음.

## Job ID ↔ wandb run ↔ 결과, 전체 매핑 (최종/유효한 것만 — 버그로 취소/재제출된 중간 job ID는 뺐음)

| Job ID | 내용 | wandb project/run | 결과 |
|---|---|---|---|
| 40425 (재실행 전 40414) | 1.7B s50, ALPS→NTP-only, AdamW | `reasoning_qwen3_1.7b`/`bcdyv5co` | math500=64.6 (표 참고, 완료) |
| 40426 (재실행 전 40418) | 1.7B s50, PGD lr=0.03 | `reasoning_qwen3_1.7b`/`2t4z4uol` | math500=69.8 (완료) |
| 40427 (재실행 전 40419) | 1.7B s50, PGD lr=0.01 | `reasoning_qwen3_1.7b`/`8gpc0gih` | math500=69.8 (완료) |
| 40432 | 1.7B s50, PGD lr=0.003, batch=8(H200) | `reasoning_qwen3_1.7b`/`bdhybpiu` | math500=73.6, gpqa=26.3, lcb=3.7, gsm8k=71.2 (완료, ifeval만 CUDA 에러로 결측) |
| 40433 | 1.7B s50, PGD lr=0.001, batch=8(H200) | `reasoning_qwen3_1.7b`/`j6h7ikzz` | **math500=75.6 (ALPS 원샷 74.6 처음 이김)**, gpqa=24.2, ifeval=44.0, lcb=6.3, gsm8k=68.8 (완료) |
| 40428 (재실행 전 40415) | 4B s50, TR-GMP NTP+KD+OPD, mask_interval=32 | `reasoning_qwen3_4b`/`ggrptvah` | math500=77.2 (완료) |
| 40429 (재실행 전 40416) | 4B s60, TR-GMP NTP+KD+OPD, mask_interval=32 | `reasoning_qwen3_4b`/`jvjj0p84` | math500=64.0 (완료) |
| 40430 (재실행 전 40417) | 4B s70, TR-GMP NTP+KD+OPD, mask_interval=32 | `reasoning_qwen3_4b`/`m1zxzi0n` | math500=39.8 (완료) |
| **40438** | 4B s50, TR-GMP NTP+KD+OPD, mask_interval=8 (candidate_masks 메모리 fix 적용판) | `reasoning_qwen3_4b`/`smx7fbtw` | **완료**: math500=71.2, gpqa=32.8, ifeval=58.4, lcb=16.4, gsm8k=81.1, wt2=13.81 — mask_interval=32 형제 행(ggrptvah, 77.2)보다 전 지표에서 낮음 |
| **40446** | 4B s60, 위와 동일, mask_interval=8 | `reasoning_qwen3_4b`/`sj5yqq7j` | **완료**: math500=44.4, gpqa=29.8, ifeval=27.2, lcb=2.2, gsm8k=73.9, wt2=18.02 — mask_interval=32 형제 행(jvjj0p84, 64.0)보다 낮음, 특히 ifeval(43.1→27.2)·lcb(10.1→2.2) 급락 |
| **40447** | 4B s70, 위와 동일, mask_interval=8 | `reasoning_qwen3_4b`/`ytkownau` | **완료**: math500=39.2, gpqa=25.3, ifeval=26.3, lcb=0.0, gsm8k=68.8, wt2=22.60 — mask_interval=32 형제 행(m1zxzi0n, 39.8)과 math500은 비슷하나 lcb 0.4→0.0 등 나머지는 하락 |

40438/40446/40447 세 잡 모두 완료, 아티팩트(대시보드) 및 로그 기준으로 상태 갱신됨. **결론: mask_interval=8은 s50/s60/s70 전부에서 32보다 나쁨** — math500보다 ifeval/lcb에서 손해가 훨씬 크다 (특히 s60). candidate_masks() OOM을 피하려고 mask_interval을 8로 줄인 것이었는데, 정확도 목적이라면 32(또는 16, 703325 참고)가 낫다.

**주의**: 40415/40416/40417(mask_interval=32) 원본 로그는 lcb 도중 CANCELLED로 끝나 있는데, 이는 진짜 실패가 아니라 이후 별도 eval-only job(40428/40429/40430, `slurm_eval_lighteval_only.sh`)이 같은 checkpoint·같은 wandb run ID에 이어서 lcb/gsm8k/zero-shot을 채운 것 — wandb API로 `run.state == finished` 및 전체 메트릭 존재를 확인해야 진짜 완료 여부를 알 수 있음 (원본 슬럼 로그만 보고 "취소됐으니 무효"라고 판단하면 안 됨). 대시보드에서 이 3개 행이 한 번 누락됐다가(동시 편집 충돌) 이 방식으로 재검증 후 복원됨.

**대시보드 동시편집 참고**: 40446/40447 결과를 올리는 도중 다른 세션이 4B S50/S60/S70에 새 행 4개(wbid fnxyxnee/vsmcdjh9/4wsr5kgp/jhij6epp)를 먼저 게시해서 최초 publish가 409 conflict로 거부됨 — 최신본을 다시 fetch해서 그 위에 40446/40447만 병합 후 재게시함. 다음 세션도 이 패턴(publish 전 항상 최신 fetch) 유지할 것.

버그 수정 과정에서 취소/재제출된 중간 job ID들(40380대~40420대 다수, use_fast 버그·캐시 재빌드·OOM 재시도 등으로 여러 번 죽었다 다시 제출됨)은 결과가 없거나 무효라 위 표에서 뺐음 — wandb에 orphan run으로 남아있을 수 있으니 혼동하지 말 것.

### THINKSTRIP-200K 4B TR-GMP NTP+KD+OPD 스윕 (2026-08-09)

데이터셋: `cosmos1030/ot3-fineweb-200k-qwen3-thinkstrip` (`elsa/data/ot3_fineweb_200k_qwen3_thinkstrip.jsonl`, 5.9GB, 20만 샘플) — PLAIN-200K와 달리 seqlen=2048 초과시 `<think>` 블록을 잘라내서 truncation을 80.3%→48.5%로 낮춘 버전. 자세한 설명은 `elsa/data/DATASETS.md` 참고.

캐시: `elsa/scripts/log_cluster/slurm_prebuild_mixed_cot_cache.sh` (job 41178) — 이 클러스터엔 CPU 전용 파티션이 없어서 `--gres=gpu` 없이 3090 파티션에 제출, GPU 슬롯 안 잡고 CPU만 사용 (1시간 41분 소요, 198868 샘플 캐싱). 스크립트: `elsa/scripts/log_cluster/prebuild_mixed_cot_cache.py`.

학습 런처: `elsa/scripts/log_cluster/slurm_gmp_tr_ntpkd_opd_qwen3_4b_thinkstrip.sh <SPARSITY> <LR> <KL_THRESHOLD> [LR_SCHEDULER] [MASK_INTERVAL]` — 기존 4B TR-GMP NTP+KD+OPD 레시피(mask_interval=32 고정)에 `--gmp_post_target_steps=0`(목표 희소도 조기 도달해도 2048 끝까지 학습) 추가, 데이터만 THINKSTRIP으로 교체.

s50/s60/s70 × lr{1e-4, 5e-5} × kl_threshold{0.01, 0.02} = 12개 잡, 전부 캐시 잡(41178)에 `--dependency=afterok`로 걸어서 순차 실행:

| Job ID | sparsity | lr | kl | wandb run | math500 | gpqa | ifeval | lcb | gsm8k | wt2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 41179 | 0.5 | 1e-4 | 0.01 | `gy7hkspq` | 67.8 | 35.9 | 49.2 | 11.6 | 80.4 | 13.19 |
| 41182 | 0.5 | 1e-4 | 0.02 | `oqduc9io` | 68.8 | 35.9 | 57.3 | 16.0 | 81.6 | 13.11 |
| 41183 | 0.5 | 5e-5 | 0.01 | `cx7cpiua` | **76.8** | 43.9 | 71.2 | 22.0 | 86.1 | 12.73 |
| 41184 | 0.5 | 5e-5 | 0.02 | `s5k3xtx8` | **77.4** | 38.9 | 68.6 | 19.0 | 85.7 | 12.90 |
| 41180 | 0.6 | 1e-4 | 0.01 | `m42g2n88` | 63.2 | 36.9 | 45.5 | 12.3 | 76.7 | 15.66 |
| 41185 | 0.6 | 1e-4 | 0.02 | `jovxawah` | 63.8 | 29.8 | **결측(아래 참고)** | 11.2 | 76.2 | 15.70 |
| 41186 | 0.6 | 5e-5 | 0.01 | `zqaskjk7` | **74.0** | 31.3 | 59.3 | 15.7 | 80.7 | 16.12 |
| 41187 | 0.6 | 5e-5 | 0.02 | `q6tt2iw2` | 72.0 | 36.4 | 59.7 | 17.9 | 81.6 | 15.42 |
| 41181 | 0.7 | 1e-4 | 0.01 | `s6hnb9m2` | 51.6 | 28.8 | 27.0 | 4.9 | 67.6 | 23.16 |
| 41188 | 0.7 | 1e-4 | 0.02 | `ypoklh00` | **53.0** | 29.8 | 30.1 | 8.6 | 71.0 | 21.88 |
| 41189 | 0.7 | 5e-5 | 0.01 | `2fuheac7` | 38.0 | 22.7 | 26.8 | 1.5 | 62.9 | 31.84 |
| 41190 | 0.7 | 5e-5 | 0.02 | `54gbsr9f` | 46.8 | 23.7 | 34.8 | 6.3 | 69.4 | 22.82 |

**결론:**
- **s50/s60에서는 lr=5e-5가 lr=1e-4보다 math500 기준 8~10pt 우세** (s50: 76.8-77.4 vs 67.8-68.8, s60: 72.0-74.0 vs 63.2-63.8).
- **s70에서는 정반대로 뒤집힘** — lr=1e-4(51.6-53.0)가 lr=5e-5(38.0-46.8)보다 우세. 희소도가 높아질수록 낮은 lr이 mask 급변에 못 따라가는 것으로 추정.
- `kl_threshold`(0.01 vs 0.02) 영향은 lr보다 훨씬 작음, 모든 sparsity에서.
- **THINKSTRIP vs PLAIN-200K(같은 레시피, mask_interval=32) 최고 조합 비교**: s50 77.4 vs 77.2(거의 동일), s60 74.0 vs 64.0(**+10pt**), s70 53.0 vs 39.8(**+13pt**) — 희소도가 높을수록 THINKSTRIP 효과가 커짐 (PLAIN-200K의 truncation 문제가 고희소도에서 더 자주 발생하는 mask-update 스텝 수와 맞물려 악화되는 것으로 보임).
- **새 실패 유형**: job 41185(s60, lr1e-4, kl0.02)에서 ifeval이 `RecursionError`로 결측 — lighteval의 ifeval 채점기가 모델이 생성한 비정상적으로 긴/중첩된 문자열을 `json.loads`로 파싱하다 재귀 깊이 초과. subprocess라 recursion limit 조정 불가, 재시도 안 함 (이전 SparseGPT-s70 gsm8k RecursionError와 같은 계열의 문제).

**캐시 삭제 알림 (2026-08-09)**: 사용자 disk quota(2048G)가 거의 다 차서 `elsa/.cache/datasets/7fe7656ed131.pkl`(THINKSTRIP-200K, 198868 샘플, 7.1G)을 삭제함 — 위 스윕(41179-41190)은 이미 다 끝난 뒤라 데이터 손실은 없지만, THINKSTRIP 데이터셋으로 다시 학습을 돌리면 첫 잡에서 캐시가 다시 빌드되며 (`slurm_prebuild_mixed_cot_cache.sh` 기준) ~1시간 41분이 추가로 걸림. 같이 있던 `.tmp` 확장자 잔재 파일(788M, 중간에 끊긴 미완성 빌드)도 같이 지움. `6a37c5438de9.pkl`(PLAIN-200K)과 `42ab70a6b2d1.pkl`(ot3_100pct_100k, kl0.1/kl0.07 재실행 잡이 쓰는 중)은 그대로 둠.

## 잡다한 참고

- H200 GPU 메모리: 143GB (H200 NVL). A100: 79GB. 3090: 24GB.
- 3090에서 PGD optimizer 쓸 땐 `gmp_gradient_checkpointing=true` 필요 (24GB로는 그냥 두면 OOM).
- 4B TR-GMP NTP+KD+OPD 조합은 A100(80GB)에서 vLLM OPD 엔진 + AdamW + 모델 다 얹으면 기본 메모리가 79GB 근처까지 차서 candidate_masks 버그를 고쳐도 여전히 빡빡함 — H200 아니면 힘듦.
- LiveCodeBench(`lighteval/code_generation_lite`, subset `v4_v5`) 데이터셋은 HuggingFace Hub의 "xet" 전송 방식이 이 서버 네트워크에서 무한 대기함. `HF_HUB_DISABLE_XET=1` 필수. 이미 `/home/shared/huggingface`에 미리 받아둠 (268 샘플).
