"""
Entry point for ADMM pruning with on-policy KD loss.
Mirrors globalprune_admm in prune.py but uses GKDADMMTrainer.
"""
import os
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
    MixedPromptDataset, collate_prompts,
    MixedTextDataset, collate_cot_kd,
)
from .data import get_dataset
from .prune import AdmmTrainingArguments  # reuse same args dataclass
from transformers import default_data_collator

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


def globalprune_admm_kd(FLAGS, model, teacher_model, tokenizer, device,
                        offpolicy_kd=False, prebuilt_opd_vllm_engine=None,
                        prebuilt_opd_vllm_params=None):
    """
    ADMM pruning with on-policy KD loss.
    Uses GKDADMMTrainer instead of ADMMTrainer.
    """
    model_name_part = FLAGS.model.split('/')[-1]
    kd_data_tag = Path(FLAGS.kd_data_path).stem if FLAGS.kd_data_path else "unknown"
    kd_lambda_tag = f"_kdlam{FLAGS.kd_lambda}" if getattr(FLAGS, 'kd_lambda', None) else ""
    # second-resolution timestamp alone collided between two jobs that
    # happened to start in the same wall-clock second (different nodes,
    # near-simultaneous sbatch submission) — both wrote to the identical
    # output_dir and silently overwrote each other's checkpoint. SLURM_JOB_ID
    # is unique cluster-wide; PID is the fallback for non-SLURM runs.
    _unique_tag = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    run_name = (
        f"{model_name_part}_pruned{FLAGS.sparsity_ratio}"
        f"_kd_{kd_data_tag}_admm_lr{FLAGS.lr}_lmda{FLAGS.admm_lmda}{kd_lambda_tag}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_unique_tag}"
    )
    if FLAGS.admm_save_path:
        output_dir = Path(FLAGS.admm_save_path) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_str = str(output_dir)
    else:
        output_dir_str = f"./kd_admm_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{_unique_tag}"

    training_args = AdmmTrainingArguments(
        wandb=FLAGS.wandb,
        run_name=run_name,
        output_dir=output_dir_str,
        num_train_epochs=FLAGS.admm_epochs,
        max_steps=FLAGS.steps if FLAGS.steps > 0 else -1,
        per_device_train_batch_size=FLAGS.admm_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=FLAGS.admm_gradient_accumulation_steps,
        learning_rate=FLAGS.lr,
        lr_scheduler_type=FLAGS.lr_scheduler,
        warmup_steps=FLAGS.lr_warmup_steps,
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
        admm_lasso_lmda=getattr(FLAGS, 'admm_lasso_lmda', 0.0),
        admm_beta1=FLAGS.admm_beta1,
        admm_beta2=FLAGS.admm_beta2,
        fsdp="full_shard auto_wrap" if getattr(FLAGS, 'admm_use_fsdp', False) else "",
        fsdp_config={"fsdp_transformer_layer_cls_to_wrap": "Qwen3DecoderLayer"} if getattr(FLAGS, 'admm_use_fsdp', False) else {},
        admm_tr_z_proj=getattr(FLAGS, 'admm_tr_z_proj', False),
        admm_tr_kl_threshold=getattr(FLAGS, 'admm_tr_kl_threshold', 0.1),
        admm_tr_max_iters=getattr(FLAGS, 'admm_tr_max_iters', 8),
        admm_tr_init_delta=getattr(FLAGS, 'admm_tr_init_delta', 0.05),
        admm_tr_delta_min=getattr(FLAGS, 'admm_tr_delta_min', 1e-3),
        admm_z_schedule_mode=getattr(FLAGS, 'admm_z_schedule_mode', 'trust_region'),
        admm_z_layerwise=getattr(FLAGS, 'admm_z_layerwise', False),
        admm_cubic_steps=getattr(FLAGS, 'admm_cubic_steps', 15),
        admm_tr_gate_at_target=getattr(FLAGS, 'admm_tr_gate_at_target', True),
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

    opd_enabled = getattr(FLAGS, 'opd_enabled', False)
    opd_lambda = getattr(FLAGS, 'opd_lambda', 0.0)
    opd_vllm_max_tokens = getattr(FLAGS, 'opd_vllm_max_tokens', 256)
    opd_vllm_gpu_mem = getattr(FLAGS, 'opd_vllm_gpu_mem', 0.25)
    opd_max_prompt_len = getattr(FLAGS, 'kd_max_prompt_len', 512)

    prompt_dataset = None
    opd_prompt_dataset = None

    if use_random_cot_ntp:
        # NTP: random 2048-token windows (no prompt masking, all tokens contribute)
        # KD prompts: separate MixedPromptDataset
        seqlen = getattr(FLAGS, "seqlen", 2048)
        nsamples = FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else 4096
        ntp_dataset = getattr(FLAGS, "kd_ntp_dataset", "mixed_cot")
        ntp_data_path = FLAGS.data_path if ntp_dataset == "c4" else FLAGS.kd_data_path
        if local_rank == 0:
            logging.info(f"Loading random {ntp_dataset} NTP dataset ({nsamples} samples, seqlen={seqlen})")
        train_dataset = get_dataset(
            dataset_name=ntp_dataset,
            tokenizer=tokenizer,
            nsamples=nsamples,
            seed=FLAGS.seed,
            seqlen=seqlen,
            data_type="train",
            data_path=ntp_data_path,
        )
        valid_inputs = get_dataset(
            dataset_name=ntp_dataset,
            tokenizer=tokenizer,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
            seqlen=seqlen,
            data_type="train",
            data_path=ntp_data_path,
        )
        data_collator = default_data_collator
        prompt_dataset = MixedPromptDataset(
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
        _append_eos = getattr(FLAGS, "cot_append_eos", True)
        train_dataset = MixedTextDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_len=getattr(FLAGS, "seqlen", 2048),
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else None,
            seed=FLAGS.seed,
            append_eos=_append_eos,
        )
        data_collator = collate_cot_kd(tokenizer.pad_token_id)
        valid_inputs = MixedTextDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_len=getattr(FLAGS, "seqlen", 2048),
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
            append_eos=_append_eos,
        )
    else:
        if local_rank == 0:
            logging.info(f"Loading math prompts from {FLAGS.kd_data_path}")
        train_dataset = MixedPromptDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.kd_nsamples if FLAGS.kd_nsamples > 0 else None,
            seed=FLAGS.seed,
        )
        data_collator = collate_prompts(tokenizer.pad_token_id)
        valid_inputs = MixedPromptDataset(
            jsonl_path=FLAGS.kd_data_path,
            tokenizer=tokenizer,
            max_prompt_len=FLAGS.kd_max_prompt_len,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed + 1,
        )
    if local_rank == 0:
        logging.info(f"KD-ADMM eval dataset: {len(valid_inputs)} samples")

    # OPD prompt dataset and vLLM engine setup
    _opd_vllm_engine = prebuilt_opd_vllm_engine
    _opd_vllm_params = prebuilt_opd_vllm_params
    if opd_enabled:
        _opd_prompt_path = getattr(FLAGS, 'opd_prompt_path', '') or FLAGS.kd_data_path
        opd_prompt_dataset = MixedPromptDataset(
            jsonl_path=_opd_prompt_path,
            tokenizer=tokenizer,
            max_prompt_len=opd_max_prompt_len,
            nsamples=None,
            seed=FLAGS.seed,
        )
        if local_rank == 0:
            logging.info(f"OPD: prompt dataset {len(opd_prompt_dataset)} samples from {_opd_prompt_path}, "
                         f"lambda={opd_lambda}, max_tokens={opd_vllm_max_tokens}")

        # Single-GPU (no FSDP): init vLLM here if no pre-built engine provided
        if _opd_vllm_engine is None and not getattr(FLAGS, 'admm_use_fsdp', False):
            import os as _os
            _os.environ['VLLM_USE_V1'] = '0'
            from vllm import LLM, SamplingParams as _SP
            _opd_max_len = opd_vllm_max_tokens + opd_max_prompt_len
            logging.info(f"OPD: initializing vLLM engine (single-GPU, gpu_mem={opd_vllm_gpu_mem}, "
                         f"max_len={_opd_max_len})")
            _opd_vllm_engine = LLM(
                FLAGS.model,
                dtype="bfloat16",
                gpu_memory_utilization=opd_vllm_gpu_mem,
                trust_remote_code=True,
                max_model_len=_opd_max_len,
                enforce_eager=True,
            )
            _opd_vllm_params = _SP(
                max_tokens=opd_vllm_max_tokens,
                temperature=0.6,
                top_p=0.95,
            )
            logging.info("OPD: vLLM engine ready")

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
        kd_offpolicy_ntp=getattr(FLAGS, "kd_offpolicy_ntp", False),
        kd_triple_loss=getattr(FLAGS, "kd_triple_loss", False),
        kd_opkd_lambda=getattr(FLAGS, "kd_opkd_lambda", 0.0),
        admm_tr_use_opkd_rollout=getattr(FLAGS, "admm_tr_use_opkd_rollout", False),
        generate_with_teacher=getattr(FLAGS, "kd_generate_with_teacher", False),
        forward_kl=getattr(FLAGS, "kd_forward_kl", False),
        prompt_dataset=prompt_dataset,
        opd_enabled=opd_enabled,
        opd_lambda=opd_lambda,
        opd_vllm_max_tokens=opd_vllm_max_tokens,
        opd_vllm_engine=_opd_vllm_engine,
        opd_vllm_params=_opd_vllm_params,
        opd_prompt_dataset=opd_prompt_dataset,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
    )

    trainer.train()

    if training_args.local_rank == 0:
        # Gradient fine-tuning: ~6*N*tokens (forward+backward+update), vs ~2*N*tokens
        # for forward-only one-shot calibration (ALPS/SparseGPT/Wanda/SparseLLM).
        n_params = sum(p.numel() for p in model.parameters())
        global_batch = (FLAGS.admm_batch_size * FLAGS.admm_gradient_accumulation_steps
                        * max(training_args.world_size, 1))
        n_tokens = trainer.state.global_step * global_batch * FLAGS.seqlen
        flops = 6 * n_params * n_tokens
        logging.info(f"Training FLOPs: {flops:.3e} ({n_params} params x {n_tokens} tokens)")
        if FLAGS.wandb and has_wandb:
            wandb.log({"flops": flops})

    # Free vLLM engine from GPU before saving/eval (lighteval needs the memory)
    import gc as _gc
    if getattr(trainer, "vllm_engine", None) is not None:
        del trainer.vllm_engine
        trainer.vllm_engine = None
        _gc.collect()
        torch.cuda.empty_cache()
    if getattr(trainer, "_opd_vllm_engine", None) is not None:
        try:
            trainer._opd_vllm_engine.shutdown()
        except Exception:
            pass
        trainer._opd_vllm_engine = None
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
