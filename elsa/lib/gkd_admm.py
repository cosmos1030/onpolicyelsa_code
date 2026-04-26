"""
Entry point for ADMM pruning with on-policy KD loss.
Mirrors globalprune_admm in prune.py but uses GKDADMMTrainer.
"""
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from transformers import TrainingArguments
from absl import logging
from functools import partial

from .trainer import ADMMTrainer

from .gkd_admm_trainer import (
    GKDADMMTrainer,
    MathPromptDataset, collate_prompts,
    MathCotKDDataset, collate_cot_kd,
)
from .data import get_dataset
from .prune import AdmmTrainingArguments  # reuse same args dataclass
from transformers import default_data_collator

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def globalprune_admm_kd(FLAGS, model, teacher_model, tokenizer, device, offpolicy_kd=False):
    """
    ADMM pruning with on-policy KD loss.
    Uses GKDADMMTrainer instead of ADMMTrainer.
    """
    model_name_part = FLAGS.model.split('/')[-1]
    kd_data_tag = Path(FLAGS.kd_data_path).stem if FLAGS.kd_data_path else "unknown"
    kd_lambda_tag = f"_kdlam{FLAGS.kd_lambda}" if getattr(FLAGS, 'kd_lambda', None) else ""
    run_name = (
        f"{model_name_part}_pruned{FLAGS.sparsity_ratio}"
        f"_kd_{kd_data_tag}_admm_lr{FLAGS.admm_lr}_lmda{FLAGS.admm_lmda}{kd_lambda_tag}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    if FLAGS.admm_save_path:
        output_dir = Path(FLAGS.admm_save_path) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)
    else:
        output_dir_str = f"./kd_admm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    training_args = AdmmTrainingArguments(
        wandb=FLAGS.wandb,
        run_name=run_name,
        output_dir=output_dir_str,
        num_train_epochs=FLAGS.admm_epochs,
        max_steps=FLAGS.admm_steps if FLAGS.admm_steps > 0 else -1,
        per_device_train_batch_size=FLAGS.admm_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=FLAGS.admm_gradient_accumulation_steps,
        learning_rate=FLAGS.admm_lr,
        lr_scheduler_type=FLAGS.admm_lr_scheduler,
        warmup_steps=FLAGS.admm_warmup_steps,
        weight_decay=FLAGS.admm_weight_decay,
        gradient_checkpointing=FLAGS.admm_gradient_checkpointing,
        fp16=(FLAGS.admm_precision == 'fp16'),
        bf16=(FLAGS.admm_precision == 'bf16' and torch.cuda.is_bf16_supported()),
        logging_steps=FLAGS.admm_logging_steps,
        eval_strategy="steps",
        logging_strategy="steps",
        eval_steps=FLAGS.admm_eval_steps,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if has_wandb and FLAGS.wandb else "none",
        remove_unused_columns=False,
        do_train=True,
        do_eval=True,
        # ADMM args
        admm_lmda=FLAGS.admm_lmda,
        admm_init_lmda=FLAGS.admm_init_lmda,
        admm_final_lmda=FLAGS.admm_final_lmda,
        admm_init_lambda_from_inv_resid=FLAGS.admm_init_lambda_from_inv_resid,
        admm_lmda_schedule_mode=FLAGS.admm_lmda_schedule_mode,
        sparsity_ratio=FLAGS.sparsity_ratio,
        admm_interval=FLAGS.admm_interval,
        base_optimizer_type=FLAGS.admm_base_optimizer,
        admm_projection_mode=FLAGS.admm_projection_mode,
        admm_projection_bias_correction=FLAGS.admm_projection_bias_correction,
        admm_dual_dtype=FLAGS.admm_dual_dtype,
        admm_split_dtype=FLAGS.admm_split_dtype,
        admm_beta1=FLAGS.admm_beta1,
        admm_beta2=FLAGS.admm_beta2,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    local_rank = training_args.local_rank

    # Fallback: use data_path as kd_data_path if not explicitly set (NTP-with-context mode)
    if not getattr(FLAGS, "kd_data_path", None):
        FLAGS.kd_data_path = FLAGS.data_path

    use_random_cot_ntp = getattr(FLAGS, "kd_use_random_cot_ntp", False)
    use_hybrid = teacher_model is None or getattr(FLAGS, "kd_interval", 1) > 1 or getattr(FLAGS, "kd_use_cot_dataset", False) or offpolicy_kd

    prompt_dataset = None

    if use_random_cot_ntp:
        # NTP: random 2048-token CoT windows (no prompt masking, all tokens contribute)
        # KD prompts: separate MathPromptDataset
        seqlen = getattr(FLAGS, "seqlen", 2048)
        nsamples = FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else 4096
        if local_rank == 0:
            logging.info(f"Loading random CoT NTP dataset ({nsamples} samples, seqlen={seqlen})")
        train_dataset = get_dataset(
            dataset_name="math_cot",
            tokenizer=tokenizer,
            nsamples=nsamples,
            seed=FLAGS.seed,
            seqlen=seqlen,
            data_type="train",
            data_path=FLAGS.kd_data_path,
        )
        valid_inputs = get_dataset(
            dataset_name="math_cot",
            tokenizer=tokenizer,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
            seqlen=seqlen,
            data_type="train",
            data_path=FLAGS.kd_data_path,
        )
        data_collator = default_data_collator
        prompt_dataset = MathPromptDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else None,
            seed=FLAGS.seed,
        )
        if local_rank == 0:
            logging.info(f"Prompt pool dataset: {len(prompt_dataset)} prompts")
    elif use_hybrid:
        if local_rank == 0:
            logging.info(f"Loading CoT KD dataset from {FLAGS.kd_data_path}")
        train_dataset = MathCotKDDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_len=getattr(FLAGS, "seqlen", 2048),
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else None,
            seed=FLAGS.seed,
        )
        data_collator = collate_cot_kd(tokenizer.pad_token_id)
        valid_inputs = MathCotKDDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_len=getattr(FLAGS, "seqlen", 2048),
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
        )
    else:
        if local_rank == 0:
            logging.info(f"Loading math prompts from {FLAGS.kd_data_path}")
        train_dataset = MathPromptDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else None,
            seed=FLAGS.seed,
        )
        data_collator = collate_prompts(tokenizer.pad_token_id)
        valid_inputs = MathPromptDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
        )
    if local_rank == 0:
        logging.info(f"KD-ADMM eval dataset: {len(valid_inputs)} samples")

    model.train()
    if teacher_model is not None:
        teacher_model.eval()
        teacher_model.to(device)

    trainer = GKDADMMTrainer(
        teacher_model=teacher_model,
        max_new_tokens=FLAGS.kd_max_new_tokens,
        gen_temperature=FLAGS.kd_temperature,
        kd_temperature=FLAGS.kd_temperature,
        ntp_lambda=FLAGS.kd_ntp_lambda,
        kd_topk=FLAGS.kd_topk,
        kd_interval=getattr(FLAGS, "kd_interval", 1),
        kd_lambda=getattr(FLAGS, "kd_lambda", 1.0),
        use_vllm=getattr(FLAGS, "kd_use_vllm", False),
        vllm_model_name=getattr(FLAGS, "model", None),
        vllm_gpu_memory_utilization=getattr(FLAGS, "kd_vllm_gpu_memory_utilization", 0.3),
        vllm_max_model_len=getattr(FLAGS, "kd_vllm_max_model_len", 0) or None,
        kd_buffer_size=getattr(FLAGS, "kd_buffer_size", 0),
        kd_buffer_refresh_interval=getattr(FLAGS, "kd_buffer_refresh_interval", 32),
        kd_step_interval=getattr(FLAGS, "kd_step_interval", 1),
        offpolicy_kd=offpolicy_kd,
        prompt_dataset=prompt_dataset,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()

    # Free vLLM engine from GPU before saving/eval (lighteval needs the memory)
    if getattr(trainer, "vllm_engine", None) is not None:
        del trainer.vllm_engine
        trainer.vllm_engine = None
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

    if FLAGS.save_model:
        trainer.save_model(output_dir_str)
        import json as _json
        from pathlib import Path as _Path
        cfg_path = _Path(output_dir_str) / "config.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            if cfg.get("architectures") and cfg["architectures"][0].startswith("FSDP"):
                cfg["architectures"] = [cfg["architectures"][0][len("FSDP"):]]
                cfg_path.write_text(_json.dumps(cfg, indent=2))
        logging.info(f"KD-ADMM pruned model saved to {output_dir_str}")
        return output_dir_str

    return None
