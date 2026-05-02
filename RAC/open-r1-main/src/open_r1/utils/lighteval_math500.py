"""Run MATH-500 eval via lighteval + vLLM subprocess.

Saves GPU memory by unloading the current model, spawning vLLM in a
subprocess, then returning pass@1. Logs to active wandb run if present.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def run_lighteval_math500(
    model_path: str,
    *,
    output_dir: Optional[str] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_length: int = 32768,
    max_new_tokens: int = 32768,
    max_samples: Optional[int] = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    lighteval_bin: Optional[str] = None,
    log_to_wandb: bool = True,
    wandb_step: Optional[int] = None,
    wandb_metric_name: str = "eval/math500_pass@1",
) -> float:
    """Evaluate model at `model_path` on MATH-500 using lighteval+vLLM.

    Returns pass@1 (float). Logs to active wandb run if log_to_wandb=True.
    """
    if lighteval_bin is None:
        lighteval_bin = _find_lighteval()

    if output_dir is None:
        output_dir = str(Path(model_path) / "lighteval_math500")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_args = (
        f"model_name={model_path},"
        f"dtype=bfloat16,"
        f"trust_remote_code=true,"
        f"tensor_parallel_size={tensor_parallel_size},"
        f"gpu_memory_utilization={gpu_memory_utilization},"
        f"max_model_length={max_model_length},"
        f"override_chat_template=true,"
        f"generation_parameters={{max_new_tokens:{max_new_tokens},temperature:{temperature},top_p:{top_p}}}"
    )

    cmd = [
        lighteval_bin, "vllm",
        model_args,
        "lighteval|math_500|0|0",
        "--output-dir", output_dir,
        "--save-details",
    ]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]

    print(f"[lighteval_math500] Running: {' '.join(cmd)}", flush=True)

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, capture_output=False)
    if result.returncode != 0:
        logger.warning(f"[lighteval_math500] lighteval exited with code {result.returncode}")

    # Parse results JSON
    pass_at_1 = _parse_results(output_dir, model_path)

    if log_to_wandb:
        _log_wandb(pass_at_1, wandb_step, wandb_metric_name)

    return pass_at_1


def _find_lighteval() -> str:
    # Same env as running script
    candidate = Path(sys.executable).parent / "lighteval"
    if candidate.exists():
        return str(candidate)
    # Fallback: PATH
    import shutil
    lb = shutil.which("lighteval")
    if lb:
        return lb
    raise RuntimeError("lighteval not found. Install it or pass lighteval_bin=")


def _parse_results(output_dir: str, model_path: str) -> float:
    # lighteval writes results under output_dir/<model_name_path>/results_*.json
    results_files = sorted(Path(output_dir).rglob("results_*.json"))
    if not results_files:
        logger.warning(f"[lighteval_math500] No results_*.json found in {output_dir}")
        return float("nan")
    result_json = results_files[-1]
    print(f"[lighteval_math500] Parsing {result_json}", flush=True)
    with open(result_json) as f:
        d = json.load(f)
    score = (
        d.get("results", {})
         .get("lighteval|math_500|0", {})
         .get("pass@k:k=1&n=1", float("nan"))
    )
    print(f"[lighteval_math500] MATH-500 pass@1 = {score}", flush=True)
    return score


def _log_wandb(pass_at_1: float, wandb_step: Optional[int], metric_name: str):
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        payload = {metric_name: pass_at_1}
        if wandb_step is not None:
            payload["eval/step"] = wandb_step
        wandb.log(payload)
    except ImportError:
        pass