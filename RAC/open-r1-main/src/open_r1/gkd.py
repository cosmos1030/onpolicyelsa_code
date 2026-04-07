#!/usr/bin/env python
# On-policy distillation (GKD) training script.
# Student = pruned model, Teacher = dense model.
# lmbda=1.0 (always on-policy), beta=1.0 (reverse KL).

import logging
import os
import sys
import pathlib

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

repo_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(repo_root))

from open_r1.configs import GKDConfig, GKDScriptArguments
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.math_eval_callback import MathEvalCallback
from open_r1.utils.wandb_logging import init_wandb_training

from open_r1_trl import ModelConfig, TrlParser, get_peft_config
from open_r1_trl.trl import GKDTrainer

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

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

    # Resume checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected – resuming from {last_checkpoint}")

    # Dataset
    from open_r1.utils import get_dataset
    dataset = get_dataset(script_args)

    # Tokenizer
    tokenizer = get_tokenizer(model_args, training_args)

    # Convert dataset to messages format (user only – student generates responses on-policy)
    prompt_column = getattr(script_args, "dataset_prompt_column", "problem")

    def to_messages(example):
        content = example.get(prompt_column, "")
        if isinstance(content, list):
            # already messages format; ensure it ends with an assistant turn
            if content and content[-1]["role"] != "assistant":
                return {"messages": content + [{"role": "assistant", "content": ""}]}
            return {"messages": content}
        messages = []
        if getattr(training_args, "system_prompt", None):
            messages.append({"role": "system", "content": training_args.system_prompt})
        messages.append({"role": "user", "content": content})
        # DataCollatorForChatML slices messages[:-1] to get the prompt, so we
        # need at least one assistant turn as a placeholder (GKD replaces it
        # on-policy at training time).
        messages.append({"role": "assistant", "content": ""})
        return {"messages": messages}

    dataset = dataset.map(to_messages)

    # apply_chat_template only accepts {"messages"} OR {"prompt"} but not both.
    # Drop plain-string columns that conflict with our "messages" column.
    sample_split = dataset[script_args.dataset_train_split]
    conflicting = ["prompt", "completion", "chosen", "rejected"]
    cols_to_drop = [c for c in conflicting if c in sample_split.column_names]
    if cols_to_drop:
        print(f"[GKD] Dropping conflicting columns: {cols_to_drop}", flush=True)
        dataset = dataset.remove_columns(cols_to_drop)

    # Trainer
    trainer = GKDTrainer(
        model=model_args.model_name_or_path,
        teacher_model=training_args.teacher_model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    # In-process MATH-500 eval at every save step (logs to same wandb run)
    if training_args.do_train:
        print("[gkd.py] Registering MathEvalCallback", flush=True)
        trainer.add_callback(
            MathEvalCallback(
                tokenizer=tokenizer,
                max_samples=getattr(training_args, "math_eval_samples", None),
                max_new_tokens=getattr(training_args, "math_eval_max_new_tokens", 4096),
                batch_size=getattr(training_args, "math_eval_batch_size", 8),
                system_prompt=getattr(training_args, "system_prompt", None),
            )
        )
        print(f"[gkd.py] Trainer now has {len(trainer.callback_handler.callbacks)} callbacks", flush=True)

    # Train
    if training_args.do_train:
        logger.info("*** Train ***")
        checkpoint = training_args.resume_from_checkpoint or last_checkpoint or None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("*** Save model ***")
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    parser = TrlParser((GKDScriptArguments, GKDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
