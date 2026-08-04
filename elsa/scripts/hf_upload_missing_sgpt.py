"""One-off retroactive HF Hub upload for SparseGPT models that finished before
--push_to_hub was added to the slurm scripts. Also writes hub_model_id/url
into each run's wandb summary so the dashboard/wandb views pick it up.

CPU-only: this is a plain file upload (HfApi.upload_folder), no GPU needed.
"""
import os

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)

from huggingface_hub import HfApi
import wandb

ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"

# (wandb_project, wandb_run_id, model_dir, hub_repo)
JOBS = [
    ("reasoning_qwen3_1.7b", "e7bnbwl8", "qwen3_1.7b_sgpt_s50pct_n128", "cosmos1030/qwen3-1.7b-sgpt-s50pct-ot80fw20"),
    ("reasoning_qwen3_1.7b", "ruvmq3dz", "qwen3_1.7b_sgpt_s60pct_n128", "cosmos1030/qwen3-1.7b-sgpt-s60pct-ot80fw20"),
    ("reasoning_qwen3_1.7b", "y90smdu6", "qwen3_1.7b_sgpt_s70pct_n128", "cosmos1030/qwen3-1.7b-sgpt-s70pct-ot80fw20"),
    ("reasoning_qwen3_1.7b", "2h03xjeh", "qwen3_1.7b_sgpt_s24_n128",    "cosmos1030/qwen3-1.7b-sgpt-2to4-ot80fw20"),
    ("reasoning_qwen3_4b",   "oyk1bskx", "qwen3_4b_sgpt_s50pct_n128",   "cosmos1030/qwen3-4b-sgpt-s50pct-ot80fw20"),
    ("reasoning_qwen3_4b",   "o06kdz7a", "qwen3_4b_sgpt_s60pct_n128",   "cosmos1030/qwen3-4b-sgpt-s60pct-ot80fw20"),
    ("reasoning_qwen3_4b",   "neofreir", "qwen3_4b_sgpt_s70pct_n128",   "cosmos1030/qwen3-4b-sgpt-s70pct-ot80fw20"),
    ("reasoning_qwen3_4b",   "07tjr91i", "qwen3_4b_sgpt_s24_n128",      "cosmos1030/qwen3-4b-sgpt-2to4-ot80fw20"),
    ("reasoning_qwen3_8b",   "8bbx14g2", "qwen3_8b_sgpt_s50pct_n128",   "cosmos1030/qwen3-8b-sgpt-s50pct-ot80fw20"),
    ("reasoning_qwen3_8b",   "96wx909u", "qwen3_8b_sgpt_s60pct_n128",   "cosmos1030/qwen3-8b-sgpt-s60pct-ot80fw20"),
    ("reasoning_qwen3_8b",   "2wb6flzu", "qwen3_8b_sgpt_s70pct_n128",   "cosmos1030/qwen3-8b-sgpt-s70pct-ot80fw20"),
    ("reasoning_qwen3_8b",   "1923gyhb", "qwen3_8b_sgpt_s24_n128",      "cosmos1030/qwen3-8b-sgpt-2to4-ot80fw20"),
]

MODELS_BASE = "/home1/doyoonkim/projects/elsa/models"

# save_path doubles as the lighteval vLLM cache dir (prune_and_eval.py passes
# it straight through as model_path for eval), so it's full of eval-cache
# junk (lighteval/, hash-named result dirs) alongside the actual model files —
# only upload the real HF model/tokenizer files.
ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "chat_template.jinja",
    "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "vocab.json", "merges.txt", "tokenizer.json",
    "model.safetensors", "model.safetensors.index.json", "model-*.safetensors",
]

api = HfApi()
wandb_api = wandb.Api()

for project, run_id, model_dir, hub_repo in JOBS:
    local_path = os.path.join(MODELS_BASE, model_dir)
    print(f"=== {model_dir} -> {hub_repo} ===")
    if not os.path.isfile(os.path.join(local_path, "config.json")):
        print(f"  SKIP: no config.json at {local_path}")
        continue
    try:
        api.create_repo(repo_id=hub_repo, exist_ok=True)
        api.upload_folder(
            folder_path=local_path,
            repo_id=hub_repo,
            allow_patterns=ALLOW_PATTERNS,
            commit_message=f"SparseGPT pruned (OT80/FW20): {model_dir}",
        )
        hub_url = f"https://huggingface.co/{hub_repo}"
        print(f"  Uploaded: {hub_url}")
        run = wandb_api.run(f"{ENTITY}/{project}/{run_id}")
        run.summary["hub_model_id"] = hub_repo
        run.summary["hub_model_url"] = hub_url
        run.summary.update()
        print(f"  wandb summary updated: {project}/{run_id}")
    except Exception as e:
        print(f"  FAILED: {e}")

print("=== all done ===")
