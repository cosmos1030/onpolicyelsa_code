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
from .data import get_dataset
from .gkd_admm_trainer import (
    GKDADMMTrainer,
    MathPromptDataset, collate_prompts,
    MathCotKDDataset, collate_cot_kd,
)
from .prune import AdmmTrainingArguments  # reuse same args dataclass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def globalprune_admm_kd(FLAGS, model, teacher_model, tokenizer, device):
    """
    ADMM pruning with on-policy KD loss.
    Uses GKDADMMTrainer instead of ADMMTrainer.
    """
    model_name_part = FLAGS.model.split('/')[-1]
    kd_data_tag = Path(FLAGS.kd_data_path).stem if FLAGS.kd_data_path else "unknown"
    run_name = (
        f"{model_name_part}_pruned{FLAGS.sparsity_ratio}"
        f"_kd_{kd_data_tag}_admm_lr{FLAGS.admm_lr}_lmda{FLAGS.admm_lmda}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M')}"
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

    # Dataset: use hybrid CoT+KD dataset when kd_interval > 1 or kd_data is cot-format
    use_hybrid = getattr(FLAGS, "kd_interval", 1) > 1 or getattr(FLAGS, "kd_use_cot_dataset", False)
    if use_hybrid:
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

    if local_rank == 0:
        logging.info(f"KD-ADMM dataset: {len(train_dataset)} samples")

    # Validation dataset (same source as NTP training data, for sparse eval)
    valid_inputs = get_dataset(
        dataset_name=FLAGS.dataset,
        tokenizer=tokenizer,
        nsamples=FLAGS.admm_num_eval_samples,
        seed=FLAGS.seed,
        seqlen=getattr(FLAGS, "seqlen", 2048),
        data_type="validation",
        save_to_cache=False,
        data_path=FLAGS.data_path,
    )
    if local_rank == 0:
        logging.info(f"KD-ADMM eval dataset: {len(valid_inputs)} samples")

    model.train()
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
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()

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
