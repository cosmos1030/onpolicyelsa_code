#!/usr/bin/env python
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import importlib.util  # ← new
import pathlib
from pathlib import Path

import torch            # ← added (used for device info / pruning path)
import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

# ---------------------------------------------------------------------------
#  Make sure the repo root is importable
# ---------------------------------------------------------------------------
repo_root = pathlib.Path(__file__).resolve().parents[1]   # adjust “1” if needed
sys.path.append(str(repo_root))

# ---------------------------------------------------------------------------
#  Local imports
# ---------------------------------------------------------------------------
from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.lighteval_math500 import run_lighteval_math500
from open_r1.utils.wandb_logging import init_wandb_training

from open_r1_trl import GRPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def to_conversation(
    example,
    *,
    prompt_column: str,
    system_prompt: str | None,
):
    # If the prompt column is already a list of messages, leave untouched
    if isinstance(example.get(prompt_column), list):
        return {"prompt": example[prompt_column]}

    # Else wrap in (system, user) two-turn chat
    prompt = []
    if system_prompt is not None:
        prompt.append({"role": "system", "content": system_prompt})
    prompt.append({"role": "user", "content": example[prompt_column]})
    return {"prompt": prompt}


def _resolve_current_run_pruned_model_path(model_args, training_args, script_args) -> str:
    """Resolve the pruned model directory for *this* run.

    Prefer the deterministic path pattern used by GRPOTrainer pruning save logic.
    Fall back to a filtered glob only when needed.
    """
    model_stem = Path(model_args.model_name_or_path).stem
    dataset_tag = (script_args.dataset_name or "data").split("/")[-1]

    thirds = getattr(training_args, "prune_thirds_to_prune", None)
    if isinstance(thirds, (list, tuple)):
        thirds_tag = "_".join(str(t) for t in thirds) if len(thirds) > 0 else "none"
    elif isinstance(thirds, str):
        thirds_tag = "1_2_3" if thirds.lower() == "all" else thirds.replace(",", "_")
    else:
        thirds_tag = "none"

    prune_n = getattr(training_args, "prune_N", None)
    prune_m = getattr(training_args, "prune_M", None)
    nm_tag = f"N{prune_n}_M{prune_m}" if prune_n and prune_m else ""

    expected = (
        Path(training_args.save_dir)
        / (
            f"{model_stem}"
            f"_pruned_{int(training_args.prune_sparsity * 100)}"
            f"_{training_args.prune_scope}"
            f"_tokens{training_args.prune_calib_tokens}"
            f"_prunemethod_{training_args.pruning_method}"
            f"_thirds_{thirds_tag}"
            f"_{nm_tag}"
            f"_{dataset_tag}"
        )
    )
    if (expected / "config.json").is_file():
        return str(expected)

    # Fallback: restrict candidates to this run's sparsity + dataset tag + valid config
    import glob as _glob

    pattern = f"{model_stem}_pruned_{int(training_args.prune_sparsity * 100)}_*_{dataset_tag}"
    candidates = [
        p
        for p in _glob.glob(str(Path(training_args.save_dir) / pattern))
        if (Path(p) / "config.json").is_file()
    ]
    if candidates:
        candidates.sort(key=os.path.getmtime)
        return candidates[-1]

    # Last-resort fallback to previous behavior output dir.
    return str(training_args.output_dir)


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main(script_args, training_args, model_args):
    # Seed
    set_seed(training_args.seed)

    # ── Logging ────────────────────────────────────────────────────────────
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16‑bit: {training_args.fp16}"
    )
    logger.info(f"Model parameters   : {model_args}")
    logger.info(f"Script parameters  : {script_args}")
    logger.info(f"Training arguments : {training_args}")

    # ── Resume checkpoint if any ───────────────────────────────────────────
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected – resuming from {last_checkpoint}")

    # ── Dataset & tokenizer ────────────────────────────────────────────────
    dataset   = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)

    # ── Model ───────────────────────────────────────────────────────────────
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # ── Reward functions ───────────────────────────────────────────────────
    reward_funcs = get_reward_funcs(script_args)

    # Pruning on JSONL trace datasets expects prompt text to stay as-is.
    if not training_args.prune:
        dataset = dataset.map(
            to_conversation,
            fn_kwargs={
                "prompt_column": script_args.dataset_prompt_column,
                "system_prompt": training_args.system_prompt,
            },
        )
        for split in dataset:
            if "messages" in dataset[split].column_names:
                dataset[split] = dataset[split].remove_columns("messages")

    # ── Trainer ────────────────────────────────────────────────────────────
    for fld in ("dataset_name", "dataset_config_name", "dataset_split"):
        setattr(training_args, fld, getattr(script_args, fld, None))

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    # ════════════════════════════════════════════════════════════════════════
    #  Train / Prune / Trace
    # ════════════════════════════════════════════════════════════════════════
    if training_args.do_train or training_args.trace_only:
        logger.info("*** Train / Trace / Prune ***")

        checkpoint = (
            training_args.resume_from_checkpoint
            or last_checkpoint
            or None
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # ── Save model artefacts ───────────────────────────────────────────
        logger.info("*** Save model ***")
        trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

        # ── Model card (only when real training *and* `trl` present) ───────
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["open-r1"],
        }
        if trainer.accelerator.is_main_process:
            if training_args.do_train:                       # guard #1
                if importlib.util.find_spec("trl") is not None:  # guard #2
                    trainer.create_model_card(**kwargs)
                else:
                    logger.info("`trl` package not found – skipping model‑card generation.")
            # Restore cache flags regardless
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

        # ── Evaluation ────────────────────────────────────────────────────
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        # ── Push to hub ───────────────────────────────────────────────────
        if training_args.push_to_hub:
            logger.info("Pushing to hub …")
            trainer.push_to_hub(**kwargs)

    else:
        logger.info("Nothing to train / trace – exiting.")

    # ── Post-pruning MATH-500 eval via lighteval+vLLM ─────────────────────
    # Pruning happens inside Trainer.__init__, so this runs regardless of
    # do_train / trace_only flags.
    if getattr(training_args, "prune", False) and trainer.accelerator.is_main_process:
        logger.info("*** Running MATH-500 eval on pruned model (lighteval+vLLM) ***")
        # Resolve pruned model path for the current run only.
        pruned_model_path = _resolve_current_run_pruned_model_path(
            model_args=model_args,
            training_args=training_args,
            script_args=script_args,
        )
        logger.info(f"Pruned model path: {pruned_model_path}")

        # Init wandb if not already active (do_train=False skips trainer wandb init)
        report_to = training_args.report_to
        if isinstance(report_to, str):
            report_to = [report_to]
        if report_to and "wandb" in report_to:
            try:
                import wandb
                if wandb.run is None:
                    sp = getattr(training_args, "prune_sparsity", None)
                    auto_name = f"rac_s{int(sp*100)}" if sp is not None else training_args.run_name
                    wandb.init(
                        project=os.environ.get("WANDB_PROJECT", "prune_eval"),
                        name=os.environ.get("WANDB_NAME") or auto_name,
                    )
            except ImportError:
                pass

        # Free GPU memory before launching vLLM (vLLM loads from disk independently)
        del trainer.model
        del trainer
        del model
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

        run_lighteval_math500(
            model_path=pruned_model_path,
            output_dir=str(Path(pruned_model_path) / "lighteval_math500"),
            max_new_tokens=getattr(training_args, "math_eval_max_new_tokens", 30000),
            max_samples=getattr(training_args, "math_eval_max_samples", None),
            wandb_step=0,
        )


# ════════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
