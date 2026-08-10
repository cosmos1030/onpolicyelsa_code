"""Run 7-benchmark eval suite via lighteval+vLLM subprocess.

Benchmarks: MATH-500, AIME24, AIME25, GPQA-Diamond, IFEval, LiveCodeBench, GSM8K.
(MMLU-Redux was dropped: Qwen3's default "thinking mode" makes the model emit
a `<think>` opening token first, which the task's generation_size=1 cuts off
before any answer letter appears, so every subset scored 0 — not worth the
complexity of forcing enable_thinking=False through lighteval's chat template
plumbing for a non-reasoning probe.)
Results are logged to the active wandb run if present.
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

LIGHTEVAL_BIN = str(Path(sys.executable).parent / "lighteval")
# Runs lighteval as `python lighteval_patched_runner.py ...` instead of the
# raw `lighteval` binary so a known upstream bug fix (AvgAtN.compute using
# self.k instead of self.n) ships with this repo via git, rather than
# depending on a manual site-packages edit per machine -- see that file for
# details.
_LIGHTEVAL_RUNNER = str(Path(__file__).resolve().parent.parent / "scripts" / "lighteval_patched_runner.py")


def _run_lighteval(model_path: str, task_str: str, out_dir: str,
                   max_new_tokens: int, gpu_util: float,
                   max_samples: Optional[int] = None,
                   max_model_length: int = 8192,
                   extra_args: Optional[list] = None,
                   tp_size: int = 1,
                   seed: int = 42) -> int:
    model_args = (
        f"model_name={model_path},dtype=bfloat16,trust_remote_code=true,"
        f"tensor_parallel_size={tp_size},gpu_memory_utilization={gpu_util:.4f},"
        f"max_model_length={max_model_length},max_num_batched_tokens={max_model_length},seed={seed},"
        f"override_chat_template=true,"
        f"generation_parameters={{max_new_tokens:{max_new_tokens},temperature:0.6,top_p:0.95,top_k:20}}"
    )
    cmd = [sys.executable, _LIGHTEVAL_RUNNER, "vllm", model_args, task_str,
           "--output-dir", out_dir, "--save-details"]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    if extra_args:
        cmd += extra_args
    print(f"[lighteval_bench] Running: {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # Strip the torchrun/torch-elastic distributed env vars inherited from the
    # parent training process. vLLM spins up its own internal process group
    # (even at tensor_parallel_size=1) via init_method="env://"; if stale
    # RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT from the parent torchrun leak
    # through, vLLM's rendezvous either expects peers that never show up or
    # collides with the still-open parent port, hanging until the 600s x2
    # TCPStore timeout. Confirmed reproducing identically across 4 different
    # nodes (n31/n10/n52/n59) and both TP=1 and TP=4 configs.
    for _var in (
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE",
        "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_USE_AGENT_STORE", "PET_NPROC_PER_NODE",
    ):
        env.pop(_var, None)
    env.setdefault("VLLM_HOST_IP", "127.0.0.1")
    env["HF_DATASETS_OFFLINE"] = "0"
    env["TRANSFORMERS_OFFLINE"] = "0"
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    elapsed = time.time() - t0
    return rc, elapsed


def _parse_results(out_dir: str) -> dict:
    # Use os.walk because glob(**) skips hidden dirs like .cache in Python <3.12
    import os as _os
    files = sorted(
        _os.path.join(root, f)
        for root, _, fs in _os.walk(out_dir)
        for f in fs
        if f.startswith("results_") and f.endswith(".json")
    )
    if not files:
        logger.warning(f"[lighteval_bench] no results JSON in {out_dir}")
        return {}
    with open(files[-1]) as f:
        return json.load(f).get("results", {})


def _compute_token_stats(out_dir: str, bench_name: str, max_new_tokens: int,
                         correct_metric_keys: list[str]) -> dict:
    """Parse lighteval details parquet and compute token/truncation stats."""
    try:
        import numpy as np
        import pandas as pd
        from pathlib import Path as _Path

        detail_files = sorted(_Path(out_dir).rglob("details_*.parquet"))
        if not detail_files:
            return {}

        dfs = [pd.read_parquet(f) for f in detail_files]
        df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        out_lens = [len(r["output_tokens"][0]) for r in df["model_response"]]
        in_lens  = [len(r["input_tokens"])     for r in df["model_response"]]

        def is_correct(m):
            for k in correct_metric_keys:
                if k in m:
                    return bool(m[k])
            return False

        correct   = [is_correct(m) for m in df["metric"]]
        truncated = [l >= max_new_tokens * 0.99 for l in out_lens]

        correct_lens  = [l for l, c in zip(out_lens, correct) if c]
        wrong_lens    = [l for l, c in zip(out_lens, correct) if not c]
        correct_trunc = [t for t, c in zip(truncated, correct) if c]
        wrong_trunc   = [t for t, c in zip(truncated, correct) if not c]

        p = bench_name
        stats = {
            f"{p}_avg_output_tokens":         float(np.mean(out_lens)),
            f"{p}_avg_input_tokens":          float(np.mean(in_lens)),
            f"{p}_max_output_tokens":         float(np.max(out_lens)),
            f"{p}_truncation_rate":           float(np.mean(truncated)),
            f"{p}_correct_avg_output_tokens": float(np.mean(correct_lens))  if correct_lens  else float("nan"),
            f"{p}_wrong_avg_output_tokens":   float(np.mean(wrong_lens))    if wrong_lens    else float("nan"),
            f"{p}_correct_truncation_rate":   float(np.mean(correct_trunc)) if correct_trunc else float("nan"),
            f"{p}_wrong_truncation_rate":     float(np.mean(wrong_trunc))   if wrong_trunc   else float("nan"),
        }
        return stats
    except Exception as e:
        logger.warning(f"[lighteval_bench] token stats failed for {bench_name}: {e}")
        return {}


def run_lighteval_bench(
    model_path: str,
    out_base: str,
    gpu_util: float = 0.9,
    max_samples: Optional[int] = None,
    log_to_wandb: bool = True,
    tp_size: int = 1,
    only_tasks: Optional[list] = None,
    seed: int = 42,
) -> dict:
    """Run the 7 benchmarks (or a subset) and return metrics dict.

    Args:
        model_path: path to saved model directory
        out_base: base directory for lighteval output
        gpu_util: vLLM gpu_memory_utilization
        max_samples: if set, limit samples per benchmark (smoketest mode)
        log_to_wandb: whether to log metrics to active wandb run
        only_tasks: if set, only run these benchmark names (subset of
            math500/aime24/aime25/gpqa/ifeval/lcb/gsm8k) instead of all 7 --
            e.g. to re-run just the one benchmark that crashed a prior eval.
    Returns:
        dict with keys like "lighteval/math500", "lighteval/gpqa_diamond", etc.
    """
    # (name, task_str, max_new_tokens, max_model_length, max_samples, correct_metric_keys)
    benchmarks = [
        ("math500", "lighteval|math_500|0",           32768, 32768, max_samples,
         ["pass@k:k=1&n=1"]),
        ("aime24",  "lighteval|aime24|0",             38912, 38912, max_samples,
         ["pass@k:k=1", "pass@k:k=1&n=1"]),
        ("aime25",  "lighteval|aime25|0",             38912, 38912, max_samples,
         ["pass@k:k=1&n=1"]),
        ("gpqa",    "lighteval|gpqa:diamond|0",       32768, 32768, max_samples,
         ["gpqa_pass@k:k=1", "pass@k:k=1&n=1", "acc_norm", "acc"]),
        ("ifeval",  "lighteval|ifeval|0",             8192, 8192, max_samples,
         ["prompt_level_strict_acc"]),
        ("lcb",     "lighteval|lcb:codegeneration|0", 32768, 32768, max_samples,
         ["codegen_pass@1:16", "pass@1"]),
        ("gsm8k",   "lighteval|gsm8k|0",              2048, 4096, max_samples,
         ["extractive_match", "acc"]),
    ]
    if only_tasks is not None:
        benchmarks = [b for b in benchmarks if b[0] in only_tasks]

    metrics = {}
    for name, task_str, max_tok, ctx_len, ms, correct_keys in benchmarks:
        out_dir = os.path.join(out_base, name)
        os.makedirs(out_dir, exist_ok=True)
        rc, elapsed = _run_lighteval(model_path, task_str, out_dir, max_tok, gpu_util, ms, ctx_len, tp_size=tp_size, seed=seed)
        metrics[f"eval_time_sec/{name}"] = elapsed
        if rc != 0:
            logger.warning(f"[lighteval_bench] {name} exited with code {rc}")
            continue
        r = _parse_results(out_dir)

        if name == "math500":
            v = r.get("lighteval|math_500|0", {}).get("pass@k:k=1&n=1")
            metrics["lighteval/math500"] = float(v) if v is not None else float("nan")
        elif name == "aime24":
            t = r.get("lighteval|aime24|0", {})
            v = t.get("pass@k:k=1", t.get("pass@k:k=1&n=1"))
            metrics["lighteval/aime24"] = float(v) if v is not None else float("nan")
        elif name == "aime25":
            t = r.get("lighteval|aime25|0", {})
            v = t.get("pass@k:k=1", t.get("pass@k:k=1&n=1"))
            metrics["lighteval/aime25"] = float(v) if v is not None else float("nan")
        elif name == "gpqa":
            t = r.get("lighteval|gpqa:diamond|0", {})
            v = t.get("gpqa_pass@k:k=1", t.get("acc_norm", t.get("acc", t.get("pass@k:k=1&n=1"))))
            metrics["lighteval/gpqa_diamond"] = float(v) if v is not None else float("nan")
        elif name == "ifeval":
            t = r.get("lighteval|ifeval|0", {})
            v = t.get("prompt_level_strict_acc", t.get("acc"))
            metrics["lighteval/ifeval_prompt"] = float(v) if v is not None else float("nan")
        elif name == "lcb":
            t = r.get("lighteval|lcb:codegeneration|0", {})
            v = t.get("codegen_pass@1:16", t.get("pass@1"))
            metrics["lighteval/lcb"] = float(v) if v is not None else float("nan")
        elif name == "gsm8k":
            t = r.get("lighteval|gsm8k|0", {})
            v = t.get("extractive_match", t.get("acc"))
            metrics["lighteval/gsm8k"] = float(v) if v is not None else float("nan")

        metrics.update(_compute_token_stats(out_dir, name, max_tok, correct_keys))

    for k, v in metrics.items():
        logger.info(f"[lighteval_bench] {k}: {v:.4f}")

    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except ImportError:
            pass

    return metrics
