# B200 container (doyoon-test) — environment notes

This folder holds launchers specific to the single-GPU B200 docker container
(`doyoon-test`, accessed via VSCode remote / `IS_CTN=1`), as opposed to the
`log_cluster` SLURM cluster `elsa/scripts/log_cluster/` targets. Lives at the
repo root (not under `elsa/scripts/`) because these launchers span both
`ALPS/` and `elsa/`. **This container has no SLURM** (`sbatch`/`srun` don't
exist here) — every script in this folder is a plain bash launcher meant to
be run directly (`bash b200_scripts/foo.sh <args>`), not submitted via
sbatch, and each one blocks the GPU until it finishes (single GPU, so runs
are queued sequentially by hand rather than scheduled).

## Why a separate folder

The container-local filesystem (`/home/log_lab/...`, including this repo
clone) is ephemeral docker overlay storage — it is wiped if this container is
ever replaced with a new one. Only `/NHNHOME/log-postech/doyoonkim/` is
genuine persistent storage (a Lustre network mount, confirmed via `findmnt`/
`stat -c '%d'` — distinct device from `/`). Every path below that matters for
reproducibility lives there, not in the repo clone itself.

## Reproducing this environment from scratch

```bash
# 1. Miniconda -> persistent storage, NOT the repo clone or $HOME
curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash /tmp/miniconda.sh -b -p /NHNHOME/log-postech/doyoonkim/miniconda3
source /NHNHOME/log-postech/doyoonkim/miniconda3/etc/profile.d/conda.sh
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n rac python=3.10 -y
conda activate rac

# 2. requirements.txt, minus flash_attn (needs torch present first)
grep -v '^flash_attn\|^packaging @ file://' requirements.txt > /tmp/requirements_final.txt
uv pip install --python "$(which python)" -r /tmp/requirements_final.txt

# 3. B200 = Blackwell (sm_100). torch==2.7.1+cu126 from requirements.txt has
#    NO sm_100 kernels ("no kernel image is available for execution on the
#    device") even though torch.cuda.is_available() still returns True. Swap
#    to the cu128 build of the SAME torch version (keeps vllm/xformers/trl's
#    version pins happy -- only the CUDA variant changes):
uv pip install --reinstall torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# 4. flash-attn now that torch is importable (needs --no-build-isolation)
uv pip install flash-attn==2.8.3 --no-build-isolation

# 5. conda init + persistent env vars -- see the block this appends to
#    ~/.bashrc (~/.bashrc itself is ephemeral -- container-local -- so on a
#    fresh container you re-run conda init and re-paste the export block
#    below; the actual conda env / secrets / caches it points at persist)
/NHNHOME/log-postech/doyoonkim/miniconda3/bin/conda init bash
```

Append to `~/.bashrc` (after the `conda initialize` block conda init adds):

```bash
# Secrets -- literal `export VAR=value` lines on purpose: ~40+ SLURM scripts
# in this repo do `grep WANDB_API_KEY ~/.bashrc | cut -d'=' -f2`, which
# breaks on anything fancier (command substitution, quoting) than a literal
# assignment. chmod 600 ~/.bashrc once these are in it.
export WANDB_API_KEY=<from /NHNHOME/log-postech/doyoonkim/secrets/wandb_api_key>
export HF_TOKEN=<from /NHNHOME/log-postech/doyoonkim/secrets/hf_token>
export HF_HOME=/NHNHOME/log-postech/doyoonkim/.cache/huggingface

export OT3_DATA=/NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl

# Compile/build caches -- all default under ~/.cache or repo-relative paths,
# both ephemeral. Redirect to persistent storage so they survive container
# reallocation (and so 8B compiles aren't paid from scratch every run).
export VLLM_CACHE_ROOT=/NHNHOME/log-postech/doyoonkim/.cache/vllm
export TRITON_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/NHNHOME/log-postech/doyoonkim/.cache/torchinductor
```

Then symlink the ephemeral repo-relative paths scripts/code expect onto
persistent storage (re-run these after a fresh git clone on a new container):

```bash
ln -sf /NHNHOME/log-postech/doyoonkim/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl \
    elsa/data/ot3_fineweb_40k_qwen3_nostrip_8192.jsonl
ln -sf /NHNHOME/log-postech/doyoonkim/dataset_cache elsa/.cache   # MixedTextDataset/MixedPromptDataset pickle cache (lib/gkd_admm_trainer.py _DEFAULT_DATASET_CACHE_DIR)
ln -sf /NHNHOME/log-postech/doyoonkim/secrets/hf_token ~/.hf_token
```

## Persistent storage layout (`/NHNHOME/log-postech/doyoonkim/`)

| Path | Contents |
|---|---|
| `miniconda3/envs/rac/` | The conda env (python 3.10, torch 2.7.1+cu128, vllm 0.10.0, flash-attn 2.8.3) |
| `secrets/` | `hf_token`, `wandb_api_key` — `chmod 700` dir / `600` files, ACL-masked so the shared group (`26msit001_A`/`2000`, which has default rwx on everything under `doyoonkim/`) can't read them |
| `data/` | Downloaded datasets (currently `ot3_fineweb_40k_qwen3_nostrip_8192.jsonl`) |
| `models/` | ALPS/GMP checkpoint saves (`--save`/`--gmp_save_path` point here) |
| `logs/<job_tag>/` | Per-run wandb dir + eval_out, one subfolder per job (see launcher scripts) |
| `dataset_cache/` | `MixedTextDataset`/`MixedPromptDataset` pickle cache (symlinked from `elsa/.cache`) |
| `.cache/huggingface/` | `HF_HOME` — model/dataset downloads + `token` file |
| `.cache/vllm/`, `.cache/triton/`, `.cache/torchinductor/` | Compile caches |
| `runs_db_qwen3_8b_nostrip8192_b200.json` | Results DB for runs done in *this* container — kept separate from the repo's `runs_db.json` (that file's existing entries used different calibration/training data; mixing them would misrepresent the comparison) |

## Single-GPU vs the 2-GPU FSDP scripts this folder mirrors

The `log_cluster` 8B scripts (e.g. `slurm_gmp_tr_ntpkd_opd_qwen3_8b_fsdp2gpu.sh`)
use `torchrun --nproc_per_node=2` because single-GPU 8B + full on-policy KD
OOM'd at ~136-141GB peak on that cluster's hardware. This container's single
B200 has 183GB — more than that peak — so runs here go through
`main.py`'s plain single-GPU path (`--gmp_use_fsdp=false`, vLLM built
in-process for OPD rollouts) instead of FSDP. Confirmed by reading
`main.py`/`lib/gmp_trainer.py`: with `WORLD_SIZE=1`, `is_distributed=False`,
which silently skips FSDP wrapping *and* the FSDP-subprocess vLLM colocation
path even if `--gmp_use_fsdp=true` is passed — so don't pass it. Mirrors the
existing single-GPU `slurm_alps_sft_ntpkd_opkd_qwen3_4b.sh`/`_1.7b.sh`
recipe, just scaled to 8B. Also note (from that same 4B script): leave
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` **unset** for these
single-GPU OPKD runs — vLLM's `CuMemAllocator` (`enable_sleep_mode=True`)
hard-asserts against it at load time. The 2-GPU FSDP-sidecar vLLM path is
fine with it since it's a separate subprocess.

## Scripts in this folder

- `alps_prune_qwen3_8b.sh <SPARSITY>` — one-shot ALPS pruning + quick eval, adapted from `../ALPS/slurm_alps_prune_8b_rtx6000ada.sh` (Qwen hub id instead of a local snapshot path, saves under `/NHNHOME/.../models/`). Sets `EVAL_FULL_SCRIPT` env var so `ALPS/qwen3_alps.py` finds `elsa/scripts/eval_full.py` in *this* repo clone instead of the other server's hardcoded fallback path (`ALPS/qwen3_alps.py` reads `os.environ.get("EVAL_FULL_SCRIPT", "/home1/doyoonkim/...")` — that fallback is another server's path, always pass the env var here).

## KNOWN UNRESOLVED: this host is heavily multi-tenant, default-width thread pools are catastrophic here (2026-08-13)

**s50 ALPS pruning did not complete in 4 attempts (~4h22m total, all killed).** Not a B200/CUDA
issue -- torch/vllm/flash-attn all confirmed working on this GPU separately (see main README/
ONBOARDING). The failure is CPU-side, before any GPU work starts:

- `vmstat`/`uptime` on this box show 80+ processes on the run queue and load average spiking as
  high as 42 (baseline ~8-15) -- this physical host is shared with other tenants' containers, and
  unlike a SLURM allocation (`--cpus-per-task=N` reserves real, isolated cores), this docker
  container's `nproc`=72 is **not** an exclusive reservation -- those 72 logical CPUs are
  contended with whatever else is scheduled on the same physical cores.
- `qwen3_alps.py`'s `get_ot_fw()` originally tokenized 40k calibration docs one-at-a-time in a
  Python loop -- measured ~44s/300 docs (~100min for the full corpus) under contention. Patched to
  `datasets.Dataset.map(batched=True, num_proc=16)`: same per-doc `tokenizer()` call, just spread
  across worker processes -- cut it to 1m44s for the full 40k on the one attempt that got that far.
  **This part of the fix is validated and worth keeping regardless of the rest.**
- A *second* single massive tokenize call in the same function (joining 500 raw docs into one
  ~6.75M-token string for the eval/PPL slice) then hung for 2h19m at 0% GPU util. Patched to reuse
  the per-doc token ids already computed above instead of re-tokenizing anything -- also worth
  keeping.
- A third attempt then hung 14min *before* the tokenizing progress bar even appeared (no
  reproducible line to blame). Tried capping `OMP_NUM_THREADS`/`MKL_NUM_THREADS`/
  `OPENBLAS_NUM_THREADS`/`RAYON_NUM_THREADS`/`NUMEXPR_NUM_THREADS=4` (all currently in
  `alps_prune_qwen3_8b.sh`) on the theory that default nproc-wide (72) thread pools synchronizing
  under heavy external contention is the common thread (measured `torch.set_num_threads(1)` made
  *no* difference for the per-doc loop specifically, so it isn't simply "torch's own pool" -- capping
  everything broadly was a hedge, not a confirmed diagnosis).
- A **fourth** attempt with those caps applied then hung 7+ min between "Loading checkpoint shards:
  100%" and the (should-be-instant) `Calibration FLOPs` print -- i.e. stuck inside
  `model.eval()` / `sum(p.numel() for p in model.parameters())`, which do no I/O and shouldn't be
  contention-sensitive at all. Working theory (unverified): capping threads to 4 may have made the
  genuinely bulk, parallelism-benefiting part of `from_pretrained` (copying/casting ~16GB of
  weights into place) *slower*, trading one failure mode for another. **The thread caps are a
  hedge that made a different, still-unexplained stall worse or just as likely -- do not assume
  they're a real fix without re-verifying.**

**Net: every attempt stalled at a different point, each incompatible with the previous fix's
theory.** This smells like host-level contention severe/variable enough that no single-process
thread-count tuning reliably fixes it -- the real fix is probably requesting an environment with
actual CPU isolation (a batch/SLURM-style allocation that reserves cores, if this platform offers
one) rather than more trial-and-error in this interactive container. Session ended with the GPU
deallocated before s50 pruning ever completed. **Next person/session: don't re-derive this from
scratch** -- start from whatever `git log -- ALPS/qwen3_alps.py b200_scripts/` shows as the latest
state, check whether host load (`uptime`) is actually low before assuming a fix works, and prefer
verifying against a short/cheap repro (small `nsamples`, or time just `get_ot_fw()` in isolation)
before burning a multi-hour end-to-end run again.
