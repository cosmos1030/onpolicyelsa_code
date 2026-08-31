# Copyright 2025 The HuggingFace Team
# SPDX-License-Identifier: Apache-2.0
#
# DPO/IPO fine-tuning on a preference dataset (e.g. UltraFeedback-binarized).
# loss_type="ipo" in the recipe switches DPOTrainer to IPO -- no separate
# trainer class needed, trl's DPOTrainer implements both.

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

from open_r1.configs import ScriptArguments, DPOConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from open_r1_trl import ModelConfig, DPOTrainer, TrlParser, get_peft_config, setup_chat_format


def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())

    last_ckpt = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if last_ckpt and not training_args.resume_from_checkpoint:
        logger.info(f"Resuming from {last_ckpt}")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    dataset = get_dataset(script_args)
    train_ds = dataset[script_args.dataset_train_split]
    eval_ds = (
        dataset[script_args.dataset_test_split]
        if training_args.do_eval and script_args.dataset_test_split in dataset
        else None
    )

    tok = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    if tok.chat_template is None:
        model, tok = setup_chat_format(model, tok, format="chatml")

    peft_config = get_peft_config(model_args)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT path: reference logprobs computed with adapters disabled, no separate copy needed
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=get_callbacks(training_args, model_args),
    )

    if training_args.do_train:
        ckpt = training_args.resume_from_checkpoint or last_ckpt
        train_out = trainer.train(resume_from_checkpoint=ckpt)
        trainer.log_metrics("train", train_out.metrics)
        trainer.save_metrics("train", train_out.metrics)
        trainer.save_state()

        trainer.save_model(training_args.output_dir)
        if trainer.accelerator.is_main_process:
            trainer.create_model_card(dataset_name=script_args.dataset_name, tags=["open-r1"])
            trainer.model.config.use_cache = True
            trainer.model.config.save_pretrained(training_args.output_dir)

            # PEFT saves only the adapter to output_dir -- merge it into the
            # base model and save full weights separately so downstream eval
            # (lighteval, etc.) can load it as a plain checkpoint.
            if peft_config is not None:
                logger.info("*** Merging LoRA adapter into base model ***")
                merged_dir = training_args.output_dir.rstrip("/") + "_merged"
                merged_model = trainer.model.merge_and_unload()
                merged_model.save_pretrained(merged_dir, safe_serialization=True)
                tok.save_pretrained(merged_dir)
                logger.info(f"Merged model saved to {merged_dir}")

        if training_args.do_eval and eval_ds is not None:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            metrics["eval_samples"] = len(eval_ds)
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name, tags=["open-r1"])


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
