"""In-process MATH-500 evaluation as a TrainerCallback.

Logs `eval/math500_pass@1` through `trainer.log()` so it goes to the same
wandb run as training — no shell script, no wandb step conflicts.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_math500_eval(
    model,
    tokenizer,
    *,
    dataset_name: str = "HuggingFaceH4/MATH-500",
    split: str = "test",
    max_samples: Optional[int] = None,
    max_new_tokens: int = 4096,
    temperature: float = 0.6,
    top_p: float = 0.95,
    batch_size: int = 8,
    system_prompt: Optional[str] = None,
    log_to_wandb: bool = True,
    wandb_step: Optional[int] = None,
    wandb_metric_name: str = "eval/math500_pass@1",
) -> float:
    """Standalone MATH-500 eval. Returns pass@1.

    Can be called directly (e.g. after one-shot pruning) or internally by
    `MathEvalCallback`. If a wandb run is active and `log_to_wandb=True`,
    the score is logged to that run under `wandb_metric_name` on an
    independent `eval/step` axis (so it coexists with trainer logs).
    """
    ds = load_dataset(dataset_name, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    problems = [ex["problem"] for ex in ds]
    golds = [ex["answer"] for ex in ds]
    print(f"[run_math500_eval] loaded {len(problems)} problems", flush=True)

    model.eval()
    was_gc = getattr(model, "is_gradient_checkpointing", False)
    if was_gc:
        model.gradient_checkpointing_disable()
    prev_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    prompts = []
    for p in problems:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": p})
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ))

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        gld = golds[i : i + batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=2048, padding_side="left",
        ).to(device)
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=pad_id, use_cache=True,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)
        for text, gold in zip(texts, gld):
            s = _score(text, gold)
            scores.append(s if s is not None else 0.0)
        print(f"[run_math500_eval] {i + len(batch)}/{len(prompts)} done, "
              f"running pass@1={sum(scores)/len(scores):.3f}", flush=True)

    # restore state
    model.config.use_cache = prev_use_cache
    if was_gc:
        model.gradient_checkpointing_enable()

    pass_at_1 = sum(scores) / len(scores) if scores else 0.0
    print(f"[run_math500_eval] FINAL pass@1={pass_at_1:.4f}", flush=True)

    if log_to_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.define_metric("eval/step")
                wandb.define_metric("eval/*", step_metric="eval/step")
                wandb.log({
                    "eval/step": wandb_step if wandb_step is not None else 0,
                    wandb_metric_name: pass_at_1,
                })
        except ImportError:
            pass

    return pass_at_1


def _score(completion: str, gold_answer: str) -> Optional[float]:
    # Gold: wrap in \boxed{} so math_verify can extract it
    gold_parsed = parse(f"\\boxed{{{gold_answer}}}", extraction_mode="first_match")
    if len(gold_parsed) == 0:
        gold_parsed = parse(gold_answer, extraction_mode="first_match")
    if len(gold_parsed) == 0:
        return None
    answer_parsed = parse(
        completion,
        extraction_config=[
            LatexExtractionConfig(
                normalization_config=NormalizationConfig(
                    nits=False, malformed_operators=False, basic_latex=True,
                    equations=True, boxed="all", units=True,
                ),
                boxed_match_priority=0,
                try_extract_without_anchor=False,
            )
        ],
        extraction_mode="first_match",
    )
    try:
        return float(verify(gold_parsed, answer_parsed))
    except Exception:
        return 0.0


class MathEvalCallback(TrainerCallback):
    """Evaluate MATH-500 pass@1 on every save event.

    Uses the trainer's current (unwrapped) model for generation, then logs
    through trainer.log() so metrics land on the same wandb run/step axis
    as training.
    """

    def __init__(
        self,
        tokenizer,
        dataset_name: str = "HuggingFaceH4/MATH-500",
        split: str = "test",
        max_samples: Optional[int] = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.6,
        top_p: float = 0.95,
        batch_size: int = 8,
        system_prompt: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.system_prompt = system_prompt

        ds = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            ds = ds.select(range(min(max_samples, len(ds))))
        self.problems = [ex["problem"] for ex in ds]
        self.golds = [ex["answer"] for ex in ds]
        self._wandb_metrics_defined = False
        print(f"[MathEvalCallback] __init__: loaded {len(self.problems)} problems from {dataset_name}/{split}", flush=True)

    def _build_prompts(self) -> list[str]:
        prompts = []
        for p in self.problems:
            msgs = []
            if self.system_prompt:
                msgs.append({"role": "system", "content": self.system_prompt})
            msgs.append({"role": "user", "content": p})
            prompts.append(self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            ))
        return prompts

    @torch.no_grad()
    def _evaluate(self, model) -> float:
        model.eval()
        was_gc = getattr(model, "is_gradient_checkpointing", False)
        if was_gc:
            model.gradient_checkpointing_disable()
        prev_use_cache = getattr(model.config, "use_cache", False)
        model.config.use_cache = True

        prompts = self._build_prompts()
        device = next(model.parameters()).device
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        scores = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            golds = self.golds[i : i + self.batch_size]
            enc = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=2048, padding_side="left",
            ).to(device)
            out = model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=pad_id,
                use_cache=True,
            )
            # strip prompt
            gen = out[:, enc["input_ids"].shape[1]:]
            texts = self.tokenizer.batch_decode(gen, skip_special_tokens=True)
            for text, gold in zip(texts, golds):
                s = _score(text, gold)
                scores.append(s if s is not None else 0.0)
            print(f"[MathEvalCallback] {i + len(batch)}/{len(prompts)} done, running pass@1={sum(scores)/len(scores):.3f}", flush=True)

        # restore
        model.config.use_cache = prev_use_cache
        if was_gc:
            model.gradient_checkpointing_enable()
        model.train()

        return sum(scores) / len(scores) if scores else 0.0

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        print(f"[MathEvalCallback] on_save fired: step={state.global_step}, is_world_zero={state.is_world_process_zero}, model_is_none={model is None}", flush=True)
        if not state.is_world_process_zero or model is None:
            return
        print(f"[MathEvalCallback] Running MATH-500 eval at step {state.global_step}", flush=True)
        pass_at_1 = self._evaluate(model)
        print(f"[MathEvalCallback] step={state.global_step} MATH-500 pass@1={pass_at_1:.4f}", flush=True)
        # Log to active wandb run so eval metrics share the training run.
        # Use define_metric to give eval its own step axis (eval/step) to avoid
        # collisions with the trainer's monotonically increasing wandb _step.
        try:
            import wandb
            if wandb.run is not None:
                if not self._wandb_metrics_defined:
                    wandb.define_metric("eval/step")
                    wandb.define_metric("eval/*", step_metric="eval/step")
                    self._wandb_metrics_defined = True
                wandb.log({
                    "eval/step": state.global_step,
                    "eval/math500_pass@1": pass_at_1,
                })
        except ImportError:
            pass
