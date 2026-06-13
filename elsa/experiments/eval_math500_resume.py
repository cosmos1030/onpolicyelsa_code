"""Standalone MATH-500 eval that resumes a specific wandb run.

Usage:
    python eval_math500_resume.py <model_path> <wandb_run_id> [max_new_tokens]
"""
import os
import sys
import torch

model_path        = sys.argv[1]
wandb_run_id      = sys.argv[2]
max_new_tokens    = int(sys.argv[3]) if len(sys.argv) > 3 else 32768
tensor_parallel_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1

os.environ["VLLM_USE_V1"] = "0"
for _k in ["MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "LOCAL_RANK", "RANK",
           "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_RUN_ID"]:
    os.environ.pop(_k, None)

import wandb
wandb.init(
    project="elsa_qwen3_0.6b",
    entity="dyk6208-gwangju-institute-of-science-and-technology",
    id=wandb_run_id,
    resume="allow",
)

sys.path.insert(0, os.path.dirname(__file__))
from lib.lighteval_math500 import run_lighteval_math500

_free_mem, _total_mem = torch.cuda.mem_get_info(0)
_vllm_gpu_util = (_free_mem / _total_mem) * 0.95
print(f"[eval] vllm gpu_memory_utilization: {_vllm_gpu_util:.3f}", flush=True)

pass_at_1 = run_lighteval_math500(
    model_path=model_path,
    output_dir=os.path.join(model_path, "lighteval_math500"),
    max_new_tokens=max_new_tokens,
    gpu_memory_utilization=_vllm_gpu_util,
    tensor_parallel_size=tensor_parallel_size,
    log_to_wandb=True,
    wandb_step=0,
)
wandb.log({"math500_pass@1": pass_at_1})
print(f"[eval] MATH-500 pass@1 = {pass_at_1:.4f}", flush=True)
wandb.finish()
