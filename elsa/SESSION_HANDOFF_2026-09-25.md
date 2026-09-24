# 세션 인계 — 2026-09-25 (이 SLURM 클러스터)

두 곳에서 같은 저장소·같은 wandb 를 쓴다: **이 클러스터**(SLURM, 잡 id 99xxxx)와
**B200 컨테이너**(log-node04/07, 잡 id 5xxxx). 손을 바꿀 때 아래를 먼저 읽을 것.

---

## 1. 지금 돌고 있는 것 (8개, PENDING 없음)

| 잡 | arm | 채우는 칸 | 노드 |
|---|---|---|---|
| 989243 | `s3_1.7b_s70_cubicnopgd` | 1.7B s70 cubic(noPGD) 3시드 | n48 (A100-40GB) |
| 994191 | `s3_1.7b_s50_sgpt_selfgen` | 1.7B s50 SparseGPT selfgen, 시드 1·42 남음 | n55 |
| 994192 | `s3_1.7b_s60_alps_selfgen` | 1.7B s60 ALPS selfgen, 시드 1·42 남음 | n53 |
| 995335 | `s3_1.7b_s60_ours` | **1.7B s60 SCOUT 재평가 (ro=32)**, 3시드 | n55 |
| 995994 | `s3_4b_s80_alpsretrain_noopd` | 4B s80 ALPS+retrain w/o OPD, **시드 1만** | n76 |
| 996870 | `s3_1.7b_s60_sgpt_selfgen_seed42` | 위 994191 계열, 시드 42만 | n52 |
| 996871 | `s3_1.7b_s70_sgpt_selfgen_seed42` | 1.7B s70 SparseGPT selfgen, 시드 42만 | n56 |
| 996872 | `s3_1.7b_s70_sparsellm_seed42` | 1.7B s70 SparseLLM, 시드 42만 | n58 |

`_seed42` / `_s1` 접미사는 harvest 의 `auto_label` 이 `_seeds?\d+$` 를 떼므로 기존
블록에 합쳐진다. **이름을 바꾸지 말 것.**

## 2. 오늘 취소한 것과 이유

| 잡 | 왜 |
|---|---|
| 988627/988628/988632/988634 | 9/23 동결 사건으로 29시간 GPU 0% 좀비. §4 참조 |
| 994193 / 994194 | 1.7B ALPS+PGD (TR / no-TR). **불필요 판정, 돌리지 말 것** |
| 994241 | `s3_4b_s80_alpsretrain_noopd` 시드 0·1. 시드 0 은 `fkndzhi2` 에 이미 완전. 995994(시드 1만)로 대체 |
| 989246/989247/989248 | 느린 카드(A6000/Ada/A100-40GB)에 있었고 시드 0·1 은 wandb 에 기록 완료. 996870~996872(시드42만, A100-80GB)로 대체 |
| 995041/995042, 995471~995474, 995502~995505 | MACKO 측정 재제출 과정. §6 |

**중요**: `wandb.log` 는 **시드 단위**로 일어난다(`eval_full.py:271-276`). 한 시드가
네 태스크를 다 끝내야 기록된다. 그래서 잡을 끊을 때 "완료된 시드"는 살아 있고
"진행 중인 시드"만 날아간다. 남은 단위 수를 세어 **재제출이 실제로 이득인지**
계산할 것 — 994191 은 이미 A100-80GB 라 끊으면 오히려 1단위 손해였다.

## 3. 다른 서버와의 분담 (2026-09-24 합의)

- **저쪽**: 4B s80 ALPS+training baseline 3시드 (`51422~51424`,
  체크포인트 `cosmos1030/gmp-kd3e-1-4b-s80pct-lr1e-4_20260917_112952`, 허브 생존 확인)
- **이쪽**: `s3_4b_s80_alpsretrain_noopd` (체크포인트 `...kd5e-1-4b-...102335`)
- 저쪽이 "n76 의 `arrzfl9a` 가 중복"이라고 알려왔는데, 그건 **이 클러스터의 994241**
  이었다. 중복이 아니라 시드 1 미완 때문이었다.

제출 전 `long_tsv_results/check_before_submit.py <size> <ckpt>` 로 반드시 확인.
**한계**: 이 스크립트는 런 상태와 요청 시드만 본다. `failed` 여도 시드 결과가
남아 있을 수 있다(`fkndzhi2` 가 그랬다). 실제 `lighteval/*_seed*` 키를 직접 볼 것.

## 4. 2026-09-23 /home1 동결 사건

11:34~11:54 에 돌던 eval 12개가 한꺼번에 얼었고 14:45 까지 `/home1` 에 쓰기가
전혀 없었다. `slurm_eval_full.sh` 가 표준출력을 NFS 에 직접 쓰고 있어서, 잡들이
uninterruptible I/O 에 박혀 GPU 를 0% 로 29시간 붙들었다(SLURM 은 RUNNING 표시).

**수정 완료**: 출력과 wandb 디렉터리를 `/local-data/user-data/$USER/` 로 옮기고,
종료 시 trap 이 `elsa/logs/eval_full_<jobid>.out` 로 1회 복사한다. 그래서 996870
이후 잡들은 **끝나기 전에는 NFS 에 로그가 없다** — 진행을 보려면
`srun --overlap --jobid=<id> cat /local-data/user-data/$USER/eval_full_<id>.out`.

멈춤 감지: SLURM 이 RUNNING 이어도 믿지 말고 (1) 로그 mtime, (2)
`srun --overlap --jobid=N nvidia-smi` 의 GPU 사용률을 볼 것. 0% 인데 메모리만
잡고 있으면 좀비다.

## 5. HF 대량 삭제 사고 (2026-09-24) — 반드시 읽을 것

public storage 할당량을 풀려고 "미참조" 420개(4.26TB)를 지웠는데, 미참조 판정을
**저장소 파일 grep 으로만** 했다. 평가런이 쓰는 체크포인트는 wandb 의
`config.model_path` / `summary.hub_model_id` 에만 있다.

**복구한 것**
- `gmp-kd3e-1-s70pct-lr1e-4_20260908_102348` (8B s70 A3 jump) — HF 캐시에서 복구
- `gmp-kd3e-1-s60pct-lr5e-5_20260903_081011` (4B s60 w/o refresh) — 다른 서버가 재업로드
- `gmp-kd3e-1-s60pct-lr5e-5_20260902_173413` (**1.7B s60 SCOUT ro=32**) — 로컬
  `elsa/models/gmp_s60pct_lr5e-05_onpol_lmda0.33_20260902_150611` 에서 복구
- 캐시에 가중치가 남아 있던 15개

**영구 소실**: 나머지 405개. 전부 8~9월 구 네이밍 스윕 체크포인트
(`gmp-kd3e-1-s{50,60,70}pct-lr*_<ts>`)이고 **현재 논문 표가 쓰는 것은 없다**.
평가 수치는 wandb 에 남아 있으나 재평가는 불가. 학습런 기준 99개는 다른 서버
`/NHNHOME/log-postech/doyoonkim/models` 에 원본이 있을 수 있다.

**앞으로**: 삭제 전 wandb 전 프로젝트의 `model_path`/`hub_model_id` 를 훑을 것.
블록당 wandb id 가 여럿이면 전부 볼 것(첫 id 만 보면 로컬 경로라 놓친다).
최근 60일은 무조건 제외. RUNNING/PENDING 잡이 쓰는 것은 절대 금지.
규모가 크면 목록을 사용자에게 보이고 승인받을 것.

## 6. MACKO 시스템 측정 (오늘 갱신)

번들: `macko_scout_speedup_vram.zip` (옛 `macko_s70_effective_speedup.zip` 은 삭제).

처음엔 `qwen3_8b_selfgen_v3_alps_s*pct` 로 쟀는데 그 체크포인트는 층별 밀도가
균일해 144/144 압축된다. **SCOUT 은 global-scope PGD 라 density>0.5 층이 dense 로
폴백**한다: s50 60/144, s60 99/144, s70 121/144, s80 136/144. 그래서 per-token 이
s50 1.39→1.22x, s60 1.61→1.43x, s70 1.90→1.75x 로 내려간다. **SCOUT 수치만 인용할 것.**

- 8B dense 대비 실질 가속(토큰 증가 반영): s60·s70 이 math500/gsm8k 1.24~1.31x,
  LCB 1.07~1.10x, IFEval 0.31~0.43x(짐), s80 은 본전.
- **4B dense 는 못 이긴다**: 8B s60+MACKO 84.3s vs 4B dense 53.8s (math500),
  avg4 72.88 vs 80.64.
- VRAM: dense 사본을 유지해야 해서 실제 상주는 dense(15.26 GiB)보다 크다
  (s70 19.56). dense 사본을 뺀 값은 s70 7.65 (0.50x), s80 5.95 (0.39x).
- 함정: json 의 sparse 키가 `macko_s80` 으로 하드코딩. `.out` 의 `model='...'` 로
  확인할 것. `.out` 끝의 `SpMV kernel 2.58x` 등 참고 줄도 하드코딩이라 인용 금지.
- 스크립트가 `TRANSFORMERS_OFFLINE=1` 이라 **허브 id 가 아니라 로컬 스냅샷 경로**를
  줘야 한다.

## 7. harvest 변경

- `.gitattributes` + `merge=keepmine`: 생성 TSV 는 병합하지 않는다(두 서버가 각자
  재생성해 매번 충돌했다). 드라이버는 **로컬 git config** 라 서버마다 심어야 하고,
  `harvest_loop.sh` 시작부가 자동으로 심는다.
- `harvest_loop.sh` 가 주기마다 `git pull --rebase` → harvest → 변화 시 commit+push.
  커밋은 `git add long_tsv_results` 만(진행 중인 수정이 딸려가면 안 된다).
- 루프를 죽일 때 `pkill -f harvest_loop.sh` 금지 — 자기 셸까지 죽인다(exit 144).
  `pgrep -f "^bash long_tsv_results/harvest_loop\.sh$"` 로 pid 를 집어 kill 하고,
  남은 `sleep 1800` 이 flock 을 쥐고 있으니 `fuser long_tsv_results/.harvest_loop.lock`
  로 찾아 같이 죽일 것.
- `EXCLUDE_RUNS` 에 `padwnv7g` 추가: 1.7B s60 SCOUT 이 이 칸만 rollout interval **8**
  로 학습한 체크포인트로 평가돼 있었다(다른 칸은 전부 ro=32). ro=32 인 `94pwg46m`
  의 체크포인트로 995335 가 다시 돌고 있고, 같은 런 이름이라 옛 런을 빼두지 않으면
  두 설정의 시드가 섞인다. 옛 값 42.97±0.71 은 `tsv_excluded/` 에 남는다.
- `METHOD` 에 `alpsretrain_noopd` 추가: 4B s80 만 런 이름에 밑줄이 하나 더 있어
  (`s3_4b_s80_alpsretrain_noopd`) 라벨이 런 이름 그대로 잡혔고, 시드 42 블록과
  갈라져 `fkndzhi2` 의 시드 0 이 표에 안 들어갔다. 이제 `[2/3]`.

## 8. 남은 일

- 8B s70 NTP+KD(2-term) 학습 여부 결정 (Figure 3(a))
- 1시드뿐인 블록들: 4B s70 cubic pace-matched, 4B s50/s60 cubic, 4B s50/s70 w/o
  rollout refresh, 8B s60 jump — 제출 전 중복 확인 필수
- 블록 라벨 미매핑: `s3_8b_a3jump003_s70_seed0/1`, `s3_8b_s70_ours_klb002`
- `s3_4b_s70_norefresh_rep` 재라벨 (실제로는 frozen-pool + jump)
- `check_before_submit.py` 를 "실제 정확도 값이 있는 시드" 기준으로 고치기
- 8B s70 Ours (KL budget 0.02) 평가(985126) 재제출 — 동결 사건으로 유실, 아직 안 걸었음
