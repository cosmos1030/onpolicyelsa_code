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
        f"max_num_batched_tokens={max_model_length},"
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
    pass_at_1, stderr = _parse_results(output_dir, model_path)

    if log_to_wandb:
        _log_wandb(pass_at_1, stderr, wandb_step, wandb_metric_name)

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


def _parse_results(output_dir: str, model_path: str):
    # lighteval writes results under output_dir/<model_name_path>/results_*.json
    results_files = sorted(Path(output_dir).rglob("results_*.json"))
    if not results_files:
        logger.warning(f"[lighteval_math500] No results_*.json found in {output_dir}")
        return float("nan"), float("nan")
    result_json = results_files[-1]
    print(f"[lighteval_math500] Parsing {result_json}", flush=True)
    with open(result_json) as f:
        d = json.load(f)
    task = d.get("results", {}).get("lighteval|math_500|0", {})
    score = task.get("pass@k:k=1&n=1", float("nan"))
    stderr = task.get("pass@k:k=1&n=1_stderr", float("nan"))
    print(f"[lighteval_math500] MATH-500 pass@1 = {score:.4f} ± {stderr:.4f}", flush=True)

    # Log first 10 samples as wandb Table
    _log_sample_table(output_dir)

    return score, stderr


def _log_sample_table(output_dir: str, n_samples: int = 10):
    try:
        import numpy as np
        import pandas as pd
        import wandb
        if wandb.run is None:
            return
        detail_files = sorted(Path(output_dir).rglob("details_lighteval|math_500|0_*.parquet"))
        if not detail_files:
            return
        df = pd.read_parquet(detail_files[-1])

        # Per-sample token stats
        out_lens = [len(r["output_tokens"][0]) for r in df["model_response"]]
        in_lens = [len(r["input_tokens"]) for r in df["model_response"]]
        correct = [bool(m.get("pass@k:k=1&n=1", 0)) for m in df["metric"]]

        # generation_size is the max_new_tokens cap used for this run
        gen_size = df["doc"].iloc[0].get("generation_size", max(out_lens))
        truncated = [l >= gen_size * 0.999 for l in out_lens]

        correct_lens = [l for l, c in zip(out_lens, correct) if c]
        wrong_lens   = [l for l, c in zip(out_lens, correct) if not c]
        correct_trunc = [t for t, c in zip(truncated, correct) if c]
        wrong_trunc   = [t for t, c in zip(truncated, correct) if not c]

        log_dict = {
            "math500_avg_output_tokens":          float(np.mean(out_lens)),
            "math500_avg_input_tokens":           float(np.mean(in_lens)),
            "math500_max_output_tokens":          float(np.max(out_lens)),
            "math500_truncation_rate":            float(np.mean(truncated)),
            "math500_correct_avg_output_tokens":  float(np.mean(correct_lens)) if correct_lens else float("nan"),
            "math500_correct_max_output_tokens":  float(np.max(correct_lens))  if correct_lens else float("nan"),
            "math500_wrong_avg_output_tokens":    float(np.mean(wrong_lens))   if wrong_lens   else float("nan"),
            "math500_correct_truncation_rate":    float(np.mean(correct_trunc)) if correct_trunc else float("nan"),
            "math500_wrong_truncation_rate":      float(np.mean(wrong_trunc))   if wrong_trunc   else float("nan"),
        }
        wandb.log(log_dict)

        # Full CSV upload as wandb artifact
        csv_rows = []
        for i, row in df.iterrows():
            csv_rows.append({
                "idx": i,
                "problem": str(row["doc"].get("query", "")),
                "model_answer": str(row["model_response"].get("text", [""])[0]),
                "gold_answer": str(row["doc"].get("choices", [""])[0]),
                "correct": bool(row["metric"].get("pass@k:k=1&n=1", 0)),
                "output_tokens": len(row["model_response"]["output_tokens"][0]),
                "input_tokens": len(row["model_response"]["input_tokens"]),
            })
        csv_df = pd.DataFrame(csv_rows)
        csv_path = str(detail_files[-1]).replace(".parquet", ".csv")
        csv_df.to_csv(csv_path, index=False)
        artifact = wandb.Artifact("math500_results", type="dataset")
        artifact.add_file(csv_path)
        wandb.log_artifact(artifact)

        # First N samples table (for quick preview)
        table = wandb.Table(columns=["idx", "problem", "model_answer", "gold_answer", "correct", "output_tokens"])
        for row in csv_rows[:n_samples]:
            table.add_data(row["idx"], row["problem"][:1000], row["model_answer"][:2000],
                           row["gold_answer"][:500], row["correct"], row["output_tokens"])
        wandb.log({"math500_samples": table})
    except Exception as e:
        logger.warning(f"[lighteval_math500] Could not log sample table: {e}")


def _log_wandb(pass_at_1: float, stderr: float, wandb_step: Optional[int], metric_name: str):
    try:
        import wandb
        if wandb.run is None:
            return
        wandb.define_metric("eval/step")
        wandb.define_metric("eval/*", step_metric="eval/step")
        payload = {
            metric_name: pass_at_1,
            metric_name + "_stderr": stderr,
            metric_name + "_upper": pass_at_1 + stderr,
            metric_name + "_lower": pass_at_1 - stderr,
        }
        if wandb_step is not None:
            payload["eval/step"] = wandb_step
        wandb.log(payload)
    except ImportError:
        pass