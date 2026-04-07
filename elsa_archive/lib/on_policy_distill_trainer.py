from transformers import Trainer
import torch
import torch.nn.functional as F
import os
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from tqdm import tqdm
import logging

# Pruning 웨이트 보존을 위한 전용 옵티마이저 임포트
from lib.optimizers import MaskedAdam 

import os
import torch
import torch.nn.functional as F
from typing import Any, Optional, Dict, Tuple
from dataclasses import dataclass, field
from transformers import Trainer, TrainingArguments
from lib.optimizers import MaskedAdam

from .trainer import OnPolicyDistillTrainer
from .data import get_dataset


@dataclass
class DistillTrainingArguments(TrainingArguments):
    """
    On-policy Distillation을 위한 전용 인자 추가
    """
    distill_temp: float = field(default=1.0, metadata={"help": "Temperature for generation and KL"})
    distill_lr: float = field(default=2e-5, metadata={"help": "Learning rate for distillation"})


def run_on_policy_distillation(args, model, tokenizer, device):
    """
    Perform on policy distillation to the pruned model using the Hugging Face Trainer and MaskedAdam optimizer.
    """
    world_size = 1
    if dist.is_initialized():
        world_size = dist.get_world_size()
    if model.dtype == torch.float16: ## upcast to fp32 for stability
        model = model.to(torch.float32)
    distill_args = DistillTrainingArguments(
        output_dir="./distill_output",
        learning_rate=args.distill_lr,
        per_device_train_batch_size=args.retrain_batch_size,
        max_steps=args.retrain_steps,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_accumulation_steps=args.retrain_gradient_accumulation_steps,
        bf16=True
    )

    num_train_samples = retrain_args.max_steps * retrain_args.per_device_train_batch_size * retrain_args.gradient_accumulation_steps * world_size

    train_dataset = get_dataset(
        args.dataset,
        tokenizer,
        nsamples=num_train_samples,
        seed=args.seed,
        seqlen=model.seqlen,
        data_path=args.data_path
    )

    trainer = Retrainer(
        model=model,
        args=retrain_args,
        train_dataset=train_dataset,
    )

    trainer.train()
