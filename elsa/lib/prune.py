import time
import torch
import math
import torch.nn as nn
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from .data import get_loaders, get_dataset, TensorData, TensorDataLoader
from .utils import *
from .optimizers import SAFE
from .trainer import ADMMTrainer, Trainer
from absl import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from transformers import TrainingArguments, Trainer
try:
    from transformers.models.opt.modeling_opt import OPTDecoderLayer
except ImportError:
    OPTDecoderLayer = None
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

### Global pruning

# --- Define AdmmTrainingArguments ---
@dataclass
class AdmmTrainingArguments(TrainingArguments):
    """
    Training arguments specific for ADMM training, inheriting from standard TrainingArguments.
    """
    wandb: bool = field(default=False, metadata={"help": "Use wandb for logging."})
    admm_lmda: float = field(default=0.001, metadata={"help": "Lambda (rho) penalty parameter for ADMM (constant schedule)."})
    admm_init_lmda: float = field(default=0.0, metadata={"help": "Initial lambda for ADMM scheduling."})
    admm_final_lmda: float = field(default=0.01, metadata={"help": "Final lambda for ADMM scheduling."})
    admm_init_lambda_from_inv_resid: bool = field(default=False, metadata={"help": "Initialize lambda from inverse of initial residual."})
    admm_lmda_schedule_mode: str = field(default='constant', metadata={"help": "Mode for lambda schedule (linear/cosine/exponential/constant)."})
    admm_interval: int = field(default=32, metadata={"help": "Interval for ADMM projection and dual updates."})
    admm_projection_mode: str = field(default='identity', metadata={"help": "Projection mode for ADMM (identity/momentum)."})
    admm_projection_bias_correction: bool = field(default=False, metadata={"help": "Use bias correction in ADMM projection (for momentum)."})
    prune_n: int = field(default=0, metadata={"help": "N for N:M sparsity."})
    prune_m: int = field(default=0, metadata={"help": "M for N:M sparsity."})
    sparsity_ratio: float = field(default=0.0, metadata={"help": "Target sparsity ratio (for reference)."})
    admm_nonuniform_sparsity: bool = field(default=False, metadata={"help": "Use non-uniform sparsity(based on sensitivity score) in ADMM."})
    nonuniform_sparsity_config_file: str = field(default='', metadata={"help": "Config file for non-uniform sparsity in ADMM (JSON format)."})
    base_optimizer_type: str = field(default='adam', metadata={"help": "Base optimizer for ADMM primal update."})
    admm_dual_dtype: str = field(default='fp32', metadata={"help": "Dtype for ADMM dual variable (fp32 or bf16)."})
    admm_split_dtype: str = field(default='fp32', metadata={"help": "Dtype for ADMM split variable (fp32 or bf16)."})
    admm_beta1: float = field(default=0.9, metadata={"help": "Beta1 for ADMM Adam optimizer."})
    admm_beta2: float = field(default=0.95, metadata={"help": "Beta2 for ADMM Adam optimizer."})

# --- globalprune_admm function ---
def globalprune_admm(FLAGS, model, tokenizer, device, prune_n=0, prune_m=0):
    """
    Performs ADMM training globally.
    """
    model_name_part = FLAGS.model.split('/')[-1]
    dataset_tag = FLAGS.dataset if FLAGS.dataset else "unknown"
    admm_run_name = f"{model_name_part}_pruned{FLAGS.sparsity_ratio}_{dataset_tag}_admm_lr{FLAGS.admm_lr}_lmda{FLAGS.admm_lmda}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    if FLAGS.admm_save_path:
        admm_output_dir = Path(FLAGS.admm_save_path) / admm_run_name
        admm_output_dir.mkdir(parents=True, exist_ok=True)
        admm_output_dir_str = str(admm_output_dir)
    else:
        admm_output_dir_str = f"./admm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    admm_training_args = AdmmTrainingArguments(
        wandb=FLAGS.wandb,
        run_name=admm_run_name,
        output_dir=admm_output_dir_str,
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
        eval_strategy= "steps",
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
        # admm arguments
        admm_nonuniform_sparsity=FLAGS.admm_nonuniform_sparsity,
        nonuniform_sparsity_config_file=FLAGS.admm_nonuniform_sparsity_config_file,
        admm_lmda=FLAGS.admm_lmda,
        admm_init_lmda=FLAGS.admm_init_lmda,
        admm_final_lmda=FLAGS.admm_final_lmda,
        admm_init_lambda_from_inv_resid=FLAGS.admm_init_lambda_from_inv_resid,
        admm_lmda_schedule_mode=FLAGS.admm_lmda_schedule_mode,
        sparsity_ratio=FLAGS.sparsity_ratio,
        admm_interval=FLAGS.admm_interval,
        base_optimizer_type=FLAGS.admm_base_optimizer,
        ## admm projection
        admm_projection_mode=FLAGS.admm_projection_mode,
        admm_projection_bias_correction=FLAGS.admm_projection_bias_correction,
        prune_n=prune_n,
        prune_m=prune_m,
        admm_dual_dtype=FLAGS.admm_dual_dtype,
        admm_split_dtype=FLAGS.admm_split_dtype,
        admm_beta1=FLAGS.admm_beta1,
        admm_beta2=FLAGS.admm_beta2,
    )

    # --- Log on main process only ---
    if admm_training_args.local_rank == 0:
        logging.info("--- Starting Global Pruning via ADMM Training  ---")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
        if admm_training_args.local_rank == 0:
            logging.info(f"Tokenizer pad_token not set. Using eos_token as pad_token.")

    # 1. Prepare Datasets for ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Preparing dataset for ADMM training...")
    # Use ADMM specific flags for dataset parameters
    if FLAGS.admm_steps > 0:
        num_train_samples = FLAGS.admm_steps * FLAGS.admm_batch_size * FLAGS.admm_gradient_accumulation_steps * admm_training_args.world_size
    else:
        num_train_samples = FLAGS.admm_num_train_samples
    if FLAGS.data_ablation:
        num_train_samples = FLAGS.admm_num_train_samples
    # Ensure model's seqlen matches the one used for dataset processing
    model.seqlen = FLAGS.seqlen

    # Rank 0 tokenizes and caches; other ranks wait then load from cache.
    import torch.distributed as dist
    if admm_training_args.local_rank == 0:
        train_inputs = get_dataset(
            dataset_name=FLAGS.dataset,
            tokenizer=tokenizer,
            nsamples=num_train_samples,
            seed=FLAGS.seed,
            seqlen=model.seqlen,
            data_type="train",
            save_to_cache=True,
            data_path=FLAGS.data_path
        )
        valid_inputs = get_dataset(
            dataset_name=FLAGS.dataset,
            tokenizer=tokenizer,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed,
            seqlen=model.seqlen,
            data_type="validation",
            save_to_cache=True,
            data_path=FLAGS.data_path
        )
    if admm_training_args.world_size > 1:
        dist.barrier()
    if admm_training_args.local_rank != 0:
        train_inputs = get_dataset(
            dataset_name=FLAGS.dataset,
            tokenizer=tokenizer,
            nsamples=num_train_samples,
            seed=FLAGS.seed,
            seqlen=model.seqlen,
            data_type="train",
            save_to_cache=False,
            data_path=FLAGS.data_path
        )
        valid_inputs = get_dataset(
            dataset_name=FLAGS.dataset,
            tokenizer=tokenizer,
            nsamples=FLAGS.admm_num_eval_samples,
            seed=FLAGS.seed,
            seqlen=model.seqlen,
            data_type="validation",
            save_to_cache=False,
            data_path=FLAGS.data_path
        )

    if admm_training_args.world_size > 1:
        dist.barrier()  # ensure all ranks finish data loading before Trainer init

    if admm_training_args.local_rank == 0:
        logging.info(f"ADMM Datasets prepared: Train size {len(train_inputs)}, Valid size {len(valid_inputs)}")

    model.train()
    # 4. Initialize ADMMTrainer
    if admm_training_args.local_rank == 0:
        logging.info("Initializing ADMMTrainer...")
    trainer = ADMMTrainer(
        model=model, # Pass the model while it's on the CPU
        args=admm_training_args,
        train_dataset=train_inputs,
        eval_dataset=valid_inputs,
        tokenizer=tokenizer,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )

    # 5. Start ADMM Training
    if admm_training_args.local_rank == 0:
        logging.info("Starting ADMM training on all processes...")
    trainer.train()

    if FLAGS.save_model:
        trainer.save_model(admm_output_dir_str)
        # FSDP Trainer.save_model writes FSDPQwen2ForCausalLM into config.json.
        # Overwrite the saved config.json with the clean architecture name.
        import json as _json
        from pathlib import Path as _Path
        cfg_path = _Path(admm_output_dir_str) / "config.json"
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            if cfg.get("architectures") and cfg["architectures"][0].startswith("FSDP"):
                cfg["architectures"] = [cfg["architectures"][0][len("FSDP"):]]
                cfg_path.write_text(_json.dumps(cfg, indent=2))
        logging.info(f"ADMM pruned model saved to {admm_output_dir_str}")
        return admm_output_dir_str

    return None


def prune_safe(args, model, tokenizer, device, prune_n=0, prune_m=0):
    """Layer-by-layer SAFE pruning (ADMM + SAM). Minimizes MSE(sparse_out, dense_out)."""
    logging.info('Starting SAFE pruning...')
    dataloader, _ = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer, data_path=getattr(args, 'data_path', None)
    )
    use_cache = getattr(model.config, 'use_cache', None)
    if use_cache is not None:
        model.config.use_cache = False

    with torch.no_grad():
        inps, _, attention_mask, position_ids, position_embeddings = prepare_calibration_input(
            model, dataloader, device, nsamples=args.nsamples
        )

    layers = get_model_layers(model)
    current_layer_outputs = torch.zeros_like(inps)
    logging.info(f'SAFE calibration prepared: {inps.shape}')

    for i in range(len(layers)):
        layer = layers[i].float().to(device)
        current_inps = inps.float().to(device)

        dense_layer_targets = torch.zeros_like(current_inps, device='cpu')
        with torch.no_grad():
            for j in range(current_inps.shape[0]):
                inp_j = current_inps[j].unsqueeze(0)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    dense_layer_targets[j] = layer(inp_j, attention_mask=attention_mask)[0].detach().cpu()
                else:
                    dense_layer_targets[j] = layer(inp_j, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0].detach().cpu()

        tensordata = TensorData(current_inps.cpu(), dense_layer_targets, device)
        loader = TensorDataLoader(tensordata, args.safe_batch_size, shuffle=True, num_workers=0).get_loader()

        num_steps_per_epoch = max(len(loader), 1)
        max_steps = math.ceil(args.safe_epochs * num_steps_per_epoch)
        warmup_steps = math.ceil(args.safe_warmup_epochs * num_steps_per_epoch)

        prunable = find_layers(layer)
        safe_params, safe_ids = [], set()
        for name in prunable:
            w = prunable[name].weight
            if w.requires_grad:
                safe_params.append(w)
                safe_ids.add(id(w))

        other_params = [p for p in layer.parameters() if p.requires_grad and id(p) not in safe_ids]

        param_groups = []
        if safe_params:
            param_groups.append({'params': safe_params, 'admm': True})
        if other_params:
            param_groups.append({'params': other_params, 'admm': False})

        opt = SAFE(
            param_groups,
            projection_fn=projection,
            lmda=args.safe_lmda,
            sparsity=args.sparsity_ratio,
            interval=args.safe_interval,
            lr=args.safe_lr,
            rho=args.safe_rho,
            prune_n=prune_n,
            prune_m=prune_m,
            betas=(args.admm_beta1, args.admm_beta2),
        )

        loss_fn = torch.nn.MSELoss().to(device)
        lr_scheduler = get_linear_schedule_with_warmup(opt.base_optimizer.base_optimizer, warmup_steps, max_steps)

        global_step = 0
        sam_cache = []

        for epoch in range(args.safe_epochs):
            t0 = time.time()
            total_loss = 0.0
            for batch_in, batch_tgt in loader:
                with torch.enable_grad():
                    with autocast(device_type='cuda', dtype=torch.bfloat16):
                        if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                            out = layer(batch_in, attention_mask=attention_mask)[0]
                        else:
                            out = layer(batch_in, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0]
                        loss = loss_fn(out, batch_tgt)
                        if args.safe_accumulation_steps > 1:
                            loss = loss / args.safe_accumulation_steps
                    loss.backward()
                    global_step += 1
                    if args.safe_accumulation_steps > 1:
                        sam_cache.append(batch_in.detach().clone())

                if global_step % args.safe_accumulation_steps == 0:
                    with torch.enable_grad():
                        opt.first_step(zero_grad=True)
                        if sam_cache:
                            for cached in sam_cache:
                                with autocast(device_type='cuda', dtype=torch.bfloat16):
                                    if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                                        l2 = loss_fn(layer(cached, attention_mask=attention_mask)[0], batch_tgt)
                                    else:
                                        l2 = loss_fn(layer(cached, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0], batch_tgt)
                                    if args.safe_accumulation_steps > 1:
                                        l2 = l2 / args.safe_accumulation_steps
                                l2.backward()
                            sam_cache = []
                        else:
                            with autocast(device_type='cuda', dtype=torch.bfloat16):
                                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                                    lp = loss_fn(layer(batch_in, attention_mask=attention_mask)[0], batch_tgt)
                                else:
                                    lp = loss_fn(layer(batch_in, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0], batch_tgt)
                            lp.backward()
                        opt.second_step(zero_grad=True)
                    opt.zero_grad(set_to_none=True)
                    lr_scheduler.step()
                total_loss += loss.item() * len(batch_in)

            avg_loss = total_loss / max(len(tensordata), 1)
            logging.info(f'Layer {i} epoch {epoch+1}/{args.safe_epochs} loss={avg_loss:.6f} t={time.time()-t0:.1f}s')

        opt.final_projection()

        with torch.no_grad():
            for j in range(current_inps.shape[0]):
                inp_j = current_inps[j].unsqueeze(0)
                if OPTDecoderLayer and isinstance(layer, OPTDecoderLayer):
                    current_layer_outputs[j] = layer(inp_j, attention_mask=attention_mask)[0].cpu()
                else:
                    current_layer_outputs[j] = layer(inp_j, attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)[0].cpu()

        layers[i] = layer.to('cpu')
        del opt, lr_scheduler, tensordata, loader, prunable
        torch.cuda.empty_cache()
        inps, current_layer_outputs = current_layer_outputs, inps

    if use_cache is not None:
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    logging.info('SAFE pruning finished.')
