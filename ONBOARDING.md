# 다른 서버로 옮길 때 알아야 할 것들

이 저장소(`elsa/` 중심의 LLM pruning 연구 코드)를 다른 서버/클러스터에서 이어서 작업하는 Claude를 위한 요약입니다. 환경 세팅 자체는 최상위 `README.md`/`elsa/README.md`를 보고, 여기는 **코드에 드러나지 않는 함정, 인프라 특이사항, 협업 방식**을 정리했습니다.

## 0. 가장 중요한 것: 작업 스타일

- **문제(잡 실패, OOM, 에러 로그)를 발견하면 물어보지 말고 바로 확인하고 고쳐서 재제출**한다. "확인할까요?"라고 묻는 것 자체가 지적받은 적 있음.
- 잡을 `sbatch`로 제출했으면 그걸로 끝이 아니라 **계속 모니터링**해야 한다. 사용자가 "잘 돌고 있나"라고 물을 때까지 기다리지 말 것. (이 세션에서는 15분 주기 `ScheduleWakeup`으로 자동 체크하는 걸로 보완함.)
- yaml/SLURM 스크립트/Python 코드 전부 **새로 작성하지 말고 기존에 동작하는 유사 파일을 Read로 읽은 후 필요한 부분만 수정**할 것. 새로 쓰면 boilerplate(env var, import 등) 빠뜨리는 사고가 반복됐음.
- run_name/job-name/output_dir은 **method+sparsity+스윕 축이 되는 모든 하이퍼파라미터**가 드러나게 짓는다. 타임스탬프만으로 유일성 확보하지 말 것 (초 단위 충돌로 체크포인트가 실제로 덮어써진 사고가 있었음 — 아래 "알려진 버그" 참고). `SLURM_JOB_ID`나 PID를 경로에 항상 같이 넣을 것.
- 결과를 wandb에서 조회한 것과 실제로 대시보드/DB에 반영한 것은 별개 — 반영 후 필드별로 다시 대조할 것.

## 1. 클러스터 종속적이라 재검증 필요한 것들 (다른 서버로 옮기면 의미 없음)

아래는 **이 특정 GIST 클러스터**에서만 유효한 정보입니다. 다른 서버로 옮기면 처음부터 다시 확인해야 합니다:

- SLURM `--exclude=` 노드 리스트 (`elsa/scripts/rerun_ot80fw20/*.sh`에 하드코딩된 `n3,n42,n46,...` 등) — 이 클러스터의 알려진 불량 노드 목록일 뿐, 다른 곳에선 의미 없음.
- 파티션 이름(`A100-80GB`, `RTX3090`, `RTX6000ADA`, `A5000`, `cpu-max16` 등)과 QOS(`hpgpu`, `normal`, `nogpu`) — 클러스터마다 다름. `sacctmgr show assoc user=$USER`로 계정이 가진 QOS 확인, `scontrol show partition <name> | grep AllowQos`로 파티션-QOS 매칭 확인.
- `/local-data/user-data/` 같은 노드-로컬 스토리지 경로 — 새 클러스터에 맞는 fast local scratch 경로로 교체 필요.
- `/home1/doyoonkim/miniconda3/envs/rac/bin/...` 같은 절대경로 하드코딩 — 새 서버의 conda env 경로로 교체.

**GPU 사이징 원칙(모델 크기 무관, 여기는 이식 가능한 교훈)**: "더 작은 모델이 N개 GPU에서 됐으니 더 큰 모델도 같은 개수면 될 것"이라는 추론을 검증 없이 믿지 말 것. 특히 ELSA ADMM 학습은 `admm_dual_dtype=fp32`/`admm_split_dtype=fp32` 오버헤드 때문에 zero-shot/one-shot 계열보다 훨씬 메모리를 많이 먹는다 (1.7B가 24GB×4에서 돌았다고 4B도 된다고 가정하면 실제로 OOM남). 새 하드웨어에서는 작은 스텝 수로 먼저 첫 스텝까지 실행해서 OOM 여부를 확인한 뒤 본 실험을 돌릴 것.

## 2. SLURM 스크립트 작성 관례 (이식 가능)

- `#SBATCH --error=...` 줄을 쓰지 말고, 스크립트 상단에 `exec 2>&1`을 넣어서 stderr를 stdout(.out)에 합칠 것.
- output/wandb 파일은 노드 로컬 스토리지에 쓰고, **잡 종료 시(성공/실패 무관) trap으로 NFS 등 영구 스토리지에 로그를 1회 복사**할 것. 로컬에만 남기면 잡 종료 후 그 노드 SSH 접근이 막혀 크래시 원인을 영영 못 보는 경우가 실제로 있었음.
  ```bash
  trap 'cp "$LOCAL_JOB_BASE/slurm_${SLURM_JOB_ID}.out" /path/to/persistent/logs/..._last.out 2>/dev/null || true' EXIT
  ```
- 디버그/스모크테스트 잡에는 반드시 dataset 크기 제한 flag를 넣을 것. 특히 `MixedTextDataset`류 클래스는 `nsamples=N` 인자를 넘겨도 **내부적으로 파일 전체를 먼저 다 읽은 뒤 자르기 때문에** 큰 jsonl(수십만 줄)에 대해선 무의미함 — 포맷만 빨리 확인하고 싶으면 `itertools.islice`로 직접 몇 줄만 읽을 것.
- 여러 잡이 wall-clock 기준 거의 동시에 죽는다고 공유 인프라 문제로 성급히 결론 내리지 말 것. **elapsed-time 기준으로도 동시(예: 매번 정확히 2h40m에 죽음)인지 확인** — elapsed 기준 일치는 인프라가 아니라 재현 가능한 코드 버그일 가능성이 훨씬 높다는 신호.

## 3. 알려진 코드 버그 / 함정 (이식 가능 — 새 서버에서도 그대로 존재하는 코드 버그)

- **`--do_kd_admm` vs `--do_offpolicy_kd_admm`을 혼동하지 말 것.** `do_kd_admm`은 on-policy(학생이 자기 rollout을 vLLM으로 생성해서 distill) 경로로, FSDP와 조합하면 `RuntimeError: 'weight' must be 2-D` 등으로 깨질 수 있음. 데이터셋의 기존 CoT로 KL만 계산하는 표준 KD를 원하면 `--do_offpolicy_kd_admm=true`를 쓸 것.
- **`--kd_topk`는 `0`(full vocab)으로 쓸 것.** `lib/gkd_admm_trainer.py`의 `_kl_loss` top-k 분기는 이미 full-vocab으로 계산된 `log_softmax`에서 top-K만 gather하고 재정규화를 안 해서, 부분합이 이론적으로 음수가 될 수 있는 값(진짜 KL divergence가 아님)을 loss로 씀. full vocab(`kd_topk=0`)이 수학적으로 올바르고, 모델 forward가 어차피 full-vocab logits를 만들기 때문에 메모리상 손해도 없음.
- **vLLM을 서브프로세스로 띄우는 eval 코드(`lib/lighteval_bench.py`, `lib/lighteval_math500.py`)는 `os.environ.copy()`로 부모(torchrun)의 `RANK`/`WORLD_SIZE`/`MASTER_ADDR`/`MASTER_PORT` 등을 그대로 상속받으면 안 됨.** vLLM이 내부적으로 자기만의 프로세스 그룹을 새로 만드는데, 이 값들이 새어 들어가면 TCPStore rendezvous가 600초×2 타임아웃으로 멎는다 (여러 다른 노드, TP=1/TP=4 무관하게 100% 재현됐던 버그, 수정 완료 — 커밋 참고). 비슷한 서브프로세스 launch 코드를 새로 짤 때는 이 env var들을 항상 지울 것.
- **체크포인트 저장 경로 충돌**: 초 단위 타임스탬프만으로 run_name/output_dir의 유일성을 확보하면, 같은 하이퍼파라미터로 다른 노드에서 거의 동시에 시작한 두 잡이 정확히 같은 디렉토리에 저장해서 한쪽이 다른 쪽을 silent overwrite할 수 있음 (실제로 발생, wandb엔 둘 다 finished로 떠서 한참 몰랐음). `SLURM_JOB_ID`나 PID를 경로에 추가해서 방지함 (`lib/gkd_admm.py`).
- **FSDP 멀티 GPU + zero-shot eval을 태스크 단위로 rank에 나눠서(`i % world_size == rank`) 돌리면 안 됨.** 태스크별 요청 수가 10~40배씩 차이나서(hellaswag/race vs boolq/rte), 빨리 끝난 rank가 `dist.all_gather_object`에서 NCCL 기본 watchdog 타임아웃(~2시간)보다 오래 기다리다 잡 전체가 죽음. zero-shot/reasoning eval은 **학습과 분리해서 rank0-only 단일 GPU 잡으로 따로 돌릴 것** (`scripts/eval_full.py`가 이 패턴).
- **`MixedTextDataset`(`lib/gkd_admm_trainer.py`)**: 매우 짧은 시퀀스(128 토큰 미만)가 배치에 섞이면 긴→짧 시퀀스 전환 지점에서 `CUBLAS_STATUS_INTERNAL_ERROR`가 날 수 있음 (데이터 손상이 아니라 shape-transition 버그) — `MIN_SAMPLE_LEN=128` 필터로 방지돼 있음. 캐시 키는 `tokenizer.backend_tokenizer.to_str()` 해시 기반이라 Qwen3 1.7B/4B/8B처럼 바이트 단위로 동일한 토크나이저를 쓰면 모델 크기 무관하게 캐시가 공유됨 — 모델 크기별로 따로 캐시를 다시 만들 필요 없음. `dist.broadcast_object_list`나 `dist.barrier()`로 대용량 데이터셋을 rank 간에 동기화하지 말 것 (세그폴트/watchdog 타임아웃 둘 다 실제로 겪음) — 파일 존재 폴링 방식이 안전함.
- **단일 GPU 잡인데도 `torch.distributed.is_initialized()`가 `True`가 될 수 있음.** OPKD용 in-process vLLM `LLM()` 엔진이 TP=1이어도 내부적으로 기본 process group을 초기화하는 부작용이 있음 — `gmp_trainer.py`에서 `is_distributed`/TR-GMP의 `_tr_dist`를 판정할 때 `is_initialized()`만 보면 이 부작용을 "진짜 멀티랭크 분산 학습"으로 착각해서, `GradualMaskManager.current_sparsity()` 등이 world_size=1인 그룹에 불필요한 `all_reduce`를 매 스텝 여러 번 호출하게 됨. vLLM sleep-mode와 겹치면 이 스푸리어스 `all_reduce`가 `NCCL error ... Cuda failure 'out of memory'`로 100% 재현 크래시함(growth 경계마다). 고친 지점: 세 곳 모두 `is_initialized() and get_world_size() > 1`로 바꿔야 함 — `GradualMaskManager.current_sparsity()`, `_tr_mask_update`의 `_tr_dist` 초기화, `globalprune_gmp`의 `is_distributed` 초기화.
- **vLLM 0.10.0의 sleep-mode(`CuMemAllocator`)를 학습 모델과 같은 GPU에 얹어 쓰면(OPKD처럼 `gpu_memory_utilization`을 작게 잡아 co-locate) 드물게(관측치 ~1/14 run) 랜덤한 시점에 세그폴트/illegal-memory-access가 남.** 원인 두 가지, 둘 다 `scripts/patch_vllm_cumem_sleep.py`로 패치 가능 (새 서버에서 `vllm==0.10.0` 깐 직후 그 스크립트 한 번 실행): (1) `CuMemAllocator`의 malloc/free 콜백이 bound method를 약한 참조로만 들고 있어서 GC가 수거해버림 (upstream 수정: vllm-project/vllm#23477, 2025-08-24 머지, vllm>=0.10.2부터 포함 — 근데 lighteval이 vllm>=0.10.2를 아직 지원 안 해서 0.10.0에 직접 백포트해야 함). (2) `CuMemAllocator.sleep()`이 끝에 무조건 `gc.collect()`+`torch.cuda.empty_cache()`를 부르는데, 이게 vLLM 자기 풀이 아니라 **프로세스 전체의 PyTorch 기본 allocator**에 작동해서, 학습 모델의 아직 안 끝난 비동기 CUDA 작업과 레이스가 남 — `sleep()` 안 그 직전에 `torch.cuda.synchronize()` 한 줄 추가로 완화.

## 4. 데이터셋 관련 주의사항

- **`ot3_fineweb_200k_qwen3.jsonl`**이 현재 표준 데이터셋(OpenThoughts3-CoT 80% + FineWeb-Edu 20%, Qwen3 토크나이저로 렌더링). 예전에 같은 이름 패턴의 `ot3_fineweb_20k.jsonl` 파일이 **DeepSeek-R1-Distill 토크나이저의 chat template**으로 렌더링된 걸 여러 Qwen3 실험(ELSA/GMP/ALPS 일부)이 그대로 잘못 가져다 쓴 사고가 있었음. **파일명에 토크나이저/모델명이 안 박혀있는 데이터 파일은 항상 의심하고, 실제로 어떤 토크나이저로 렌더링됐는지 먼저 확인**할 것.
- **`math_cot`(`math_220k_cot.jsonl`, 순수 수학 CoT) vs `mixed_cot`(`ot3_fineweb_200k_qwen3.jsonl`, 일반 웹텍스트+CoT 혼합)는 완전히 다른 도메인의 데이터셋**이다. 둘을 학습한 모델의 eval loss/PPL 절대값을 직접 비교하면 안 됨 (수학 텍스트가 훨씬 반복적/저엔트로피라 loss가 원래 낮게 나옴). "하이퍼파라미터 레시피를 복제했다"는 말이 데이터셋까지 복제했다는 뜻은 아니므로, 두 run을 비교할 때는 `dataset`/`data_path` config가 실제로 같은지 반드시 먼저 확인할 것 — 다른 걸 눈치채지 못하고 성능 차이를 다른 요인(스케줄, 버그) 탓으로 잘못 짚을 뻔한 사고가 있었음.

## 5. 결과 취합 시 함정

- **zero-shot 9개 태스크의 wandb 메트릭 키 이름이 파이프라인마다 다름**: SparseGPT/Wanda(`RAC/open-r1-main/src/open_r1/prune_and_eval.py`)는 `zero_shot/{task}`(언더스코어 있음), ALPS/SparseLLM/ELSA/GMP(`elsa/scripts/eval_full.py`)는 `zeroshot/{task}`(언더스코어 없음). 값만 보고 하드코딩하지 말고 `dict(run.summary)`로 실제 키를 먼저 확인할 것.
- **HF 업로드 시 `allow_patterns`로 필터링 필수** — 모델 저장 경로에 lighteval eval 캐시가 같이 쌓여서(모델 하나당 수 GB) 필터 없이 업로드하면 잡동사니가 그대로 repo에 올라감.

## 6. 참고: 이번 세션에서 정리한 문서

- `elsa/README.md` — 전체 flag 레퍼런스, 환경변수, KD/standalone-eval 사용법.
- `/README.md` — 저장소 구조, 환경 세팅, 현재 활성 워크플로우(OT80/FW20) 안내.
- `elsa/scripts/rerun_ot80fw20/` — 실제로 동작 확인된 SLURM 스크립트 (train/eval 분리, dependency chaining, KD, sweep 등) — 새 클러스터에 맞게 파티션/exclude/경로만 바꿔서 재사용 가능.
