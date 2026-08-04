# On-Policy ELSA: Reasoning-Aware LLM Pruning with ADMM

Joint pruning and training of LLMs via ADMM, with on-policy knowledge distillation and CoT-aware NTP. Evaluated on MATH-500.

---

## Repository Structure

```
projects/
├── elsa/                         # ELSA: ADMM-based pruning + training
│   ├── main.py                   # Entry point (all flags defined here)
│   ├── lib/
│   │   ├── gkd_admm.py           # Hybrid KD+NTP ADMM trainer setup
│   │   ├── gkd_admm_trainer.py   # GKDADMMTrainer (NTP-CoT + on-policy KD)
│   │   ├── prune.py              # Standard ADMM pruning (NTP on c4/math_trace)
│   │   ├── trainer.py            # Base ADMMTrainer (ADMM z/u updates)
│   │   ├── data.py               # Dataset loaders (c4, math_cot, math_trace, ...)
│   │   ├── lighteval_math500.py  # MATH-500 eval via lighteval + vLLM subprocess
│   │   └── eval.py               # Zero-shot eval (lm-eval)
│   ├── config/                   # Accelerate configs + wandb sweep yamls
│   │   ├── sweep_qwen_ntp_cot_{30,40,50,60,70}pct.yaml
│   │   └── sweep_qwen_hybrid_cot_kd_{30,40,50,60,70}pct.yaml
│   ├── scripts/                  # SLURM job scripts
│   │   ├── slurm_sweep_agent_qwen_ntp_cot.sh
│   │   └── slurm_sweep_agent_hybrid_cot_kd.sh
│   └── data/
│       └── math_220k_cot.jsonl   # Pre-extracted CoT traces (see Data Prep)
│
└── RAC/open-r1-main/             # SparseGPT pruning + GRPO baseline
    ├── src/open_r1/grpo.py       # SparseGPT pruning + math500 eval pipeline
    └── sweep_configs/
        ├── qwen3_0.6_cot_sparsity.yaml    # CoT calibration sweep (30-70%)
        └── qwen3_0.6_prompt_sparsity.yaml # Prompt calibration sweep (30-70%)
```

---

## Environment Setup

```bash
conda create -n rac python=3.10
conda activate rac
pip install -r elsa/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

> `elsa/requirements.txt` is a full `pip freeze` of the actual working environment (torch 2.7.1, transformers 4.56.2, vllm 0.10.0, trl 0.21.0, lighteval 0.12.0 — vLLM/lighteval are already included, no separate install needed). The conda env name **must stay `rac`** if you want to run the SLURM scripts under `elsa/scripts/` unmodified — they hardcode `/home1/.../miniconda3/envs/rac/bin/...` paths; otherwise update those paths at the top of each script for your own machine.

### HuggingFace Token (for model upload / gated model downloads)

```bash
echo "hf_YOUR_TOKEN_HERE" > ~/.hf_token
chmod 600 ~/.hf_token
```

Scripts read `~/.hf_token` automatically — never hardcode tokens in scripts.

### wandb Login

```bash
wandb login
```
Alternatively, scripts read `WANDB_API_KEY` out of `~/.bashrc` (`grep WANDB_API_KEY ~/.bashrc`) — add `export WANDB_API_KEY=...` there if you'd rather not run `wandb login` interactively (e.g. on a headless SLURM node).

### Other environment variables scripts rely on

| Variable | Purpose |
|---|---|
| `HF_HOME` | HuggingFace cache dir (models/datasets) — point at fast local/scratch storage. |
| `HF_DATASETS_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` once everything needed is cached locally (avoids hub lookups mid-training); `0` when you need to download or push. |
| `VLLM_HOST_IP` | Set to `127.0.0.1` — vLLM's IP auto-detection fails on some cluster nodes. |
| `TOKENIZERS_PARALLELISM` | Set to `false` to silence tokenizer fork warnings under `torchrun`. |
| `PYTORCH_CUDA_ALLOC_CONF` | Set to `expandable_segments:True` — reduces fragmentation-driven OOMs on long ADMM runs. |

On a new SLURM cluster, also update each script's `--output=/local-data/...` scratch path and `--exclude=` node list — both are specific to the cluster they were written on.

---

## Data Preparation

### CoT Trace Dataset (`math_220k_cot.jsonl`)

Extracted from `open-r1/OpenR1-Math-220k`. Each line: `{"text": "<problem>\n\n<think>...</think>..."}`.

```bash
cd elsa
python scripts/make_math_cot_jsonl.py \
    --output data/math_220k_cot.jsonl
```

---

## Methods

### 1. NTP-CoT (Next-Token Prediction on CoT tokens only)

Prunes with ADMM while training NTP loss **only on CoT tokens** (everything after `<think>`).

```bash
cd elsa
python main.py \
    --model="Qwen/Qwen3-0.6B" \
    --dataset=math_cot \
    --data_path=data/math_220k_cot.jsonl \
    --sparsity_ratio=0.5 \
    --admm_steps=4096 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_lr=1e-5 \
    --admm_lmda=5e-3 \
    --admm_lmda_schedule_mode=cosine \
    --admm_interval=32 \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --save_model=True \
    --admm_save_path=models/ \
    --eval_math500=True \
    --math500_max_new_tokens=30000 \
    --wandb=True \
    --wandb_project=elsa_qwen3_0.6b
```

### 2. Hybrid NTP-CoT + On-Policy KD

NTP on CoT tokens every step + on-policy KD (reverse KL, top-k=50) every `kd_interval` steps.

```bash
cd elsa
python main.py \
    --model="Qwen/Qwen3-0.6B" \
    --dataset=math_cot \
    --data_path=data/math_220k_cot.jsonl \
    --sparsity_ratio=0.5 \
    --admm_steps=4096 \
    --admm_batch_size=1 \
    --admm_gradient_accumulation_steps=8 \
    --admm_lr=1e-5 \
    --admm_lmda=5e-3 \
    --admm_lmda_schedule_mode=cosine \
    --admm_interval=32 \
    --admm_base_optimizer=adamw \
    --admm_precision=bf16 \
    --do_kd_admm=True \
    --kd_use_cot_dataset=True \
    --kd_data_path=data/math_220k_cot.jsonl \
    --kd_interval=16 \
    --kd_lambda=0.5 \
    --kd_max_new_tokens=512 \
    --kd_max_prompt_len=512 \
    --kd_topk=50 \
    --save_model=True \
    --admm_save_path=models/ \
    --eval_math500=True \
    --math500_max_new_tokens=32768 \
    --wandb=True \
    --wandb_project=elsa_qwen3_0.6b
```

### 3. NTP-Trace (Full prompt+CoT NTP)

Standard ELSA NTP on full prompt+CoT traces.

```bash
cd elsa
python main.py \
    --model="Qwen/Qwen3-0.6B" \
    --dataset=math_trace \
    --data_path=data/math_220k_cot.jsonl \
    --sparsity_ratio=0.3 \
    --admm_steps=4096 \
    ...
```

### 4. SparseGPT (RAC baseline)

One-shot pruning with SparseGPT, calibrated on CoT traces or prompts.

```bash
cd RAC/open-r1-main
python src/open_r1/grpo.py \
    --config recipes/Qwen3-0.6B/grpo/config_pruning.yaml \
    --model_name_or_path=Qwen/Qwen3-0.6B \
    --do_train=False \
    --prune=True \
    --pruning_method=SparseGPT \
    --prune_calib_tokens=1_000_000 \
    --prune_sparsity=0.5 \
    --eval_math500=True \
    --math_eval_max_new_tokens=30000
```

---

## Sparsity Sweeps (SLURM + wandb)

### Create and run a sweep

```bash
cd elsa

# 1. Create sweep (prints SWEEP_ID)
wandb sweep config/sweep_qwen_ntp_cot_50pct.yaml

# 2. Submit agents (9 = 3 lr × 3 lmda)
for i in {1..9}; do
    sbatch scripts/slurm_sweep_agent_qwen_ntp_cot.sh <SWEEP_ID>
done
```

Same pattern for hybrid:
```bash
wandb sweep config/sweep_qwen_hybrid_cot_kd_50pct.yaml
for i in {1..9}; do
    sbatch scripts/slurm_sweep_agent_hybrid_cot_kd.sh <SWEEP_ID>
done
```

### Available sweep configs

| Method | Sparsity | File |
|--------|----------|------|
| NTP-CoT | 30% | `config/sweep_qwen_ntp_cot.yaml` |
| NTP-CoT | 40–70% | `config/sweep_qwen_ntp_cot_{40,50,60,70}pct.yaml` |
| Hybrid KD | 30% | `config/sweep_qwen_hybrid_cot_kd.yaml` |
| Hybrid KD | 40–70% | `config/sweep_qwen_hybrid_cot_kd_{40,50,60,70}pct.yaml` |
| SparseGPT CoT | 30–70% | `RAC/open-r1-main/sweep_configs/qwen3_0.6_cot_sparsity.yaml` |
| SparseGPT prompt | 30–70% | `RAC/open-r1-main/sweep_configs/qwen3_0.6_prompt_sparsity.yaml` |

Sweep grid: `admm_lr ∈ {5e-6, 1e-5, 5e-5}` × `admm_lmda ∈ {1e-3, 5e-3, 1e-2}` = 9 configs.

---

## Current Active Workflow (OT80/FW20 rerun)

The sections above describe the original paper setup (`math_220k_cot.jsonl`, single-GPU 0.6B sweeps). The actively-maintained scripts for larger models (Qwen3-1.7B/4B/8B) and the corrected mixed dataset live elsewhere:

- **`elsa/scripts/rerun_ot80fw20/`** — reference SLURM scripts (train, KD, dependency-chained train→eval) for SparseGPT/ALPS/SparseLLM/ELSA on Qwen3-1.7B/4B/8B, using `elsa/data/ot3_fineweb_200k_qwen3.jsonl` (`--dataset=mixed_cot`) instead of `math_220k_cot.jsonl`.
- **`elsa/scripts/build_ot3_fineweb_dataset.py`** — builds that mixed dataset (OpenThoughts3 CoT 80% + FineWeb-Edu 20%).
- **`elsa/scripts/eval_full.py`** — standalone single-GPU eval (PPL + zero-shot + 5-benchmark reasoning suite), meant to run as a separate job from training (see `elsa/README.md` for why).
- **KD flags**: use `--do_offpolicy_kd_admm=true` (dataset-CoT KD, no generation) for standard KD, **not** `--do_kd_admm` (that's a different, on-policy/generation-based path). Set `--kd_topk=0` (full vocab) — the top-k path has a known bug where it doesn't renormalize over the truncated support, producing an invalid (sometimes negative) loss.

See `elsa/README.md` for the full flag reference and env var setup used by these scripts.

---

## HuggingFace Model Upload

Pruned models are automatically uploaded to `cosmos1030/` after MATH-500 eval when `--push_to_hub=True`.

Repo name format: `cosmos1030/{base_model}-{method}-{sparsity}-{lr}-{lmda}`

Example: `cosmos1030/Qwen3-0.6B-elsa-ntp-cot-s50pct-lr1e-5-lmda5e-3`

To enable upload, add to your run:
```bash
--push_to_hub=True
# optionally override:
--hub_model_id=cosmos1030/my-custom-name
```

---

## Key Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `facebook/opt-125m` | HF model path or local snapshot |
| `--sparsity_ratio` | `0.6` | Target sparsity (0.3–0.9) |
| `--dataset` | `c4` | Calibration data: `c4`, `wikitext2`, `math_cot`, `math_trace`, `math_prompt` |
| `--data_path` | `None` | Path to local JSONL |
| `--admm_steps` | `10` | Training steps |
| `--admm_lr` | `2e-4` | Learning rate |
| `--admm_lmda` | `0.01` | ADMM penalty λ |
| `--admm_lmda_schedule_mode` | `constant` | λ schedule: `constant`, `linear`, `cosine` |
| `--admm_interval` | `2` | Steps between z/u ADMM updates |
| `--admm_base_optimizer` | `adam` | `adam`, `adamw`, `adam8bit` |
| `--admm_precision` | `bf16` | `fp32`, `fp16`, `bf16` |
| `--do_kd_admm` | `False` | Enable on-policy KD inside ADMM |
| `--kd_use_cot_dataset` | `False` | Use hybrid mode (NTP-CoT + KD) |
| `--kd_interval` | `1` | Run KD generation every N steps |
| `--kd_lambda` | `1.0` | Weight of KD loss vs NTP loss |
| `--kd_max_new_tokens` | `512` | Max tokens generated per KD step |
| `--kd_topk` | `50` | Top-k vocab for KD loss |
| `--eval_math500` | `False` | Run MATH-500 eval after pruning |
| `--math500_max_new_tokens` | `4096` | Max tokens for MATH-500 generation |
| `--save_model` | `False` | Save pruned model |
| `--admm_save_path` | `None` | Directory to save model |
| `--push_to_hub` | `False` | Upload pruned model to HuggingFace Hub |
| `--hub_model_id` | `None` | HF repo id (auto-generated if not set) |
| `--wandb` | `False` | Enable wandb logging |
| `--wandb_project` | `None` | wandb project name |
