import time
import numpy as np
import torch
from transformers import AutoTokenizer
from lib.prune import globalprune_admm
from lib.eval import eval_ppl, eval_zero_shot
from lib.lighteval_math500 import run_lighteval_math500
from lib.utils import check_sparsity, get_llm
from lib.on_policy_distill import run_on_policy_distillation
from lib.gkd_admm import globalprune_admm_kd
from lib.gmp_trainer import globalprune_gmp
from absl import logging, app, flags
from importlib.metadata import version
import os
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
import torch.distributed as dist
import wandb

logging.info(f"{version('torch')=}")
logging.info(f"{version('transformers')=}")
logging.info(f"{version('accelerate')=}")
logging.info(f'# of gpus: {torch.cuda.device_count()}')

FLAGS = flags.FLAGS


def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend='nccl')

    if FLAGS.wandb and local_rank == 0:
        if getattr(FLAGS, 'do_gmp', False):
            group = "gmp"
            run_name = (
                f"gmp_s{FLAGS.sparsity_ratio}"
                f"_lr{FLAGS.gmp_lr}"
                f"_steps{FLAGS.gmp_steps}"
            )
        elif FLAGS.do_kd_admm:
            group = "onpolicy_kd_admm"
            run_name = (
                f"onpolicy_kd_admm"
                f"_s{FLAGS.sparsity_ratio}"
                f"_lr{FLAGS.admm_lr}"
                f"_lmda{FLAGS.admm_lmda}"
                f"_kdlam{FLAGS.kd_lambda}"
                f"_topk{FLAGS.kd_topk}"
                f"_steps{FLAGS.admm_steps}"
            )
        elif getattr(FLAGS, 'do_offpolicy_kd_admm', False):
            group = "offpolicy_kd_admm"
            run_name = (
                f"offpolicy_kd_admm"
                f"_s{FLAGS.sparsity_ratio}"
                f"_lr{FLAGS.admm_lr}"
                f"_lmda{FLAGS.admm_lmda}"
                f"_topk{FLAGS.kd_topk}"
                f"_steps{FLAGS.admm_steps}"
            )
        else:
            group = "ntp_admm"
            run_name = (
                f"ntp_admm"
                f"_s{FLAGS.sparsity_ratio}"
                f"_lr{FLAGS.admm_lr}"
                f"_lmda{FLAGS.admm_lmda}"
                f"_steps{FLAGS.admm_steps}"
            )

        if FLAGS.do_distill:
            kl_type = "ForwardKL" if FLAGS.distill_alpha == 0.0 else ("ReverseKL" if FLAGS.distill_alpha == 1.0 else f"Alpha{FLAGS.distill_alpha}")
            run_name += f"_distill_{kl_type}_steps{FLAGS.distill_steps}"

        wandb.init(
            project=FLAGS.wandb_project,
            group=group,
            name=run_name,
            save_code=True,
        )

        if not dict(wandb.config):
            wandb.config.update(arguments)
        else:
            updated_args = {
                k: wandb.config.get(k, v) for k, v in arguments.items()
            }
            FLAGS = type('FLAGS', (), updated_args)()
            logging.info(f"Updated args with wandb.config: {FLAGS}")
    else:
        if local_rank == 0:
            logging.info('\n' + '\n'.join([f'{k} = {v}' for k, v in arguments.items()]))


    # admm_final_lmda should always follow admm_lmda unless explicitly overridden via command line.
    # The absl default is 0.01 and the dataclass default is also 0.01, so we can't distinguish
    # "user set it" from "it's just the default". Always override with admm_lmda to be sweep-safe.
    FLAGS.admm_final_lmda = FLAGS.admm_lmda

    # Setting seeds for reproducibility
    np.random.seed(FLAGS.seed)
    torch.random.manual_seed(FLAGS.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if FLAGS.sparsity_type != "unstructured":
        assert FLAGS.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, FLAGS.sparsity_type.split(":"))

    if local_rank == 0:
        logging.info(f"loading llm model {FLAGS.model}")

    model = get_llm(FLAGS.model, FLAGS.seqlen)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=False)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = model.to('cpu')
    model.config.use_cache = False

    logging.info(f"Process {local_rank} uses device {device}")

    saved_pruned_model_path = None
    _train_time_sec = 0.0
    if FLAGS.sparsity_ratio != 0:
        logging.info("pruning starts")
        _t_train_start = time.time()
        if getattr(FLAGS, 'do_gmp', False):
            model.to(torch.bfloat16).to(device)
            from lib.gkd_admm_trainer import MathCotKDDataset
            train_dataset = MathCotKDDataset(
                jsonl_path=FLAGS.data_path,
                tokenizer=tokenizer,
                max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                max_len=getattr(FLAGS, 'gmp_max_seq_len', 2048),
            )
            saved_pruned_model_path = globalprune_gmp(model, tokenizer, train_dataset, FLAGS)
        elif FLAGS.do_kd_admm:
            teacher_model = get_llm(FLAGS.model, FLAGS.seqlen)
            teacher_model.to(torch.bfloat16).to(device)
            saved_pruned_model_path = globalprune_admm_kd(FLAGS, model, teacher_model, tokenizer, device)
            del teacher_model
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'do_offpolicy_kd_admm', False):
            teacher_model = get_llm(FLAGS.model, FLAGS.seqlen)
            teacher_model.to(torch.bfloat16).to(device)
            saved_pruned_model_path = globalprune_admm_kd(FLAGS, model, teacher_model, tokenizer, device, offpolicy_kd=True)
            del teacher_model
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'dataset', '') == 'math_cot':
            # NTP with full problem context: no teacher, no KD, uses MathCotKDDataset
            saved_pruned_model_path = globalprune_admm_kd(FLAGS, model, None, tokenizer, device)
        else:
            saved_pruned_model_path = globalprune_admm(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        _train_time_sec = time.time() - _t_train_start
        logging.info(f"Training time: {_train_time_sec/3600:.2f}h")

    if local_rank == 0:
        logging.info("Pruning finished")

    if is_distributed:
        dist.barrier()
        state_dict_options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        full_state = get_model_state_dict(model, options=state_dict_options)
        if local_rank == 0:
            model = get_llm(FLAGS.model, FLAGS.seqlen)
            model.load_state_dict(full_state)
        dist.destroy_process_group()
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()


    if FLAGS.do_distill:
        if local_rank == 0:
            logging.info("--- Starting On-Policy Distillation Phase ---")
        # 1. 여기서 티처 모델을 직접 로드합니다. (원본 Dense 모델)
        # 메모리 효율을 위해 bfloat16을 권장하며, GPU 장치(device)로 이동시킵니다.
        teacher_model = get_llm(FLAGS.model, FLAGS.seqlen)
        teacher_model.to(torch.bfloat16).to(device)
        teacher_model.eval()

        # 2. 로드한 티처 모델을 인자로 명시적으로 넘겨줍니다.
        run_on_policy_distillation(FLAGS, model, teacher_model, tokenizer, device)

        # 3. 학습이 끝나면 티처를 메모리에서 해제하여 다음 단계(Eval 등)를 대비합니다.
        del teacher_model
        torch.cuda.empty_cache()

    if local_rank == 0:

        if "gemma-2-27b" in FLAGS.model:
            logging.info("gemma-2-27b model detected. Casting to torch.bfloat16 for stability.")
            model = model.to(torch.bfloat16)
        else:
            logging.info(f"Casting model ({FLAGS.model}) to torch.float16.")
            model = model.to(torch.float16)
        model.seqlen = FLAGS.seqlen
        model = model.to(device)
        model.eval()
        # sparsity sanity check
        logging.info("*"*30)
        sparsity_ratio = check_sparsity(model,log_by_block=True)
        logging.info(f"sparsity sanity check {sparsity_ratio:.4f}")
        logging.info("*"*30)

        # perplexity evaluation
        ppl_test = eval_ppl(FLAGS, model, tokenizer, device,data_path=FLAGS.data_path)
        logging.info([(key,ppl) for key,ppl in ppl_test.items()])
        if FLAGS.wandb:
            wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})
        ## zero-shot evaluation (runs before math500 so model is still in GPU)
        if FLAGS.eval_zero_shot:
            logging.info(f"--- Evaluating After Pruning (global_admm, Zero-Shot) ---")
            accelerate = "70b" in FLAGS.model
            task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa", "piqa","race"]
            num_shot = 0
            results_after = eval_zero_shot(FLAGS, FLAGS.model, model, tokenizer, task_list, num_shot, accelerate)
            logging.info(f"Zero-shot results after pruning (global_admm):")
            logging.info(results_after)
            if FLAGS.wandb:
                for task_name, metrics in results_after.items():
                    try:
                        acc = metrics.get('acc,none', metrics.get('acc', None))
                        stderr = metrics.get('acc_stderr,none', metrics.get('acc_stderr', None))
                        if acc is not None:
                                wandb.log({f"global_admm/{task_name}_acc": acc})
                        if stderr is not None:
                                wandb.log({f"global_admm/{task_name}_stderr": stderr})
                    except Exception as log_e:
                        logging.warning(f"Could not log zero-shot metric for {task_name}: {log_e}")

        ## MATH-500 evaluation via lighteval+vLLM (runs last — deletes model to free VRAM for vLLM)
        if FLAGS.eval_math500:
            logging.info(f"--- Evaluating After Pruning (MATH-500, lighteval+vLLM) ---")
            # Resolve saved model path
            if FLAGS.math500_model_path:
                _math500_model_path = FLAGS.math500_model_path
            elif saved_pruned_model_path and os.path.isfile(os.path.join(saved_pruned_model_path, "config.json")):
                _math500_model_path = saved_pruned_model_path
                logging.info(f"Using current run's saved pruned model dir: {_math500_model_path}")
            elif FLAGS.save_model and FLAGS.admm_save_path:
                import glob as _glob
                _subdirs = [
                    p for p in _glob.glob(os.path.join(FLAGS.admm_save_path, "*pruned*"))
                    if os.path.isfile(os.path.join(p, "config.json"))
                ]
                _subdirs = sorted(_subdirs, key=os.path.getmtime)
                if _subdirs:
                    _math500_model_path = _subdirs[-1]
                    logging.info(f"Found pruned model dir: {_math500_model_path}")
                else:
                    import tempfile as _tmpfile
                    _math500_model_path = _tmpfile.mkdtemp(prefix="elsa_eval_")
                    logging.info(f"No pruned subdir found; saving to temp: {_math500_model_path}")
                    model.save_pretrained(_math500_model_path)
                    tokenizer.save_pretrained(_math500_model_path)
            else:
                import tempfile as _tmpfile
                _math500_model_path = _tmpfile.mkdtemp(prefix="elsa_eval_")
                logging.info(f"Saving model to temp path for eval: {_math500_model_path}")
                model.save_pretrained(_math500_model_path)
                tokenizer.save_pretrained(_math500_model_path)

            # Delete model to free all VRAM for vLLM subprocess
            model.to("cpu")
            import gc as _gc
            del model
            _gc.collect()
            torch.cuda.empty_cache()

            for _k in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'LOCAL_RANK', 'RANK',
                       'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS', 'TORCHELASTIC_RUN_ID']:
                os.environ.pop(_k, None)
            os.environ['VLLM_USE_V1'] = '0'

            _free_mem, _total_mem = torch.cuda.mem_get_info(0)
            _vllm_gpu_util = (_free_mem / _total_mem) * 0.95
            logging.info(f"vLLM gpu_memory_utilization (dynamic): {_vllm_gpu_util:.3f} ({_free_mem/1e9:.1f}/{_total_mem/1e9:.1f} GB free)")

            _t_eval_start = time.time()
            pass_at_1 = run_lighteval_math500(
                model_path=_math500_model_path,
                output_dir=os.path.join(_math500_model_path, "lighteval_math500"),
                max_new_tokens=FLAGS.math500_max_new_tokens,
                max_samples=getattr(FLAGS, 'math500_max_samples', None) or None,
                tensor_parallel_size=world_size,
                gpu_memory_utilization=_vllm_gpu_util,
                log_to_wandb=FLAGS.wandb,
                wandb_step=0,
            )
            _eval_time_sec = time.time() - _t_eval_start
            logging.info(f"MATH-500 pass@1 = {pass_at_1:.4f}")
            logging.info(f"Eval time: {_eval_time_sec/3600:.2f}h")
            if FLAGS.wandb:
                wandb.log({
                    "train_time_sec": _train_time_sec,
                    "eval_time_sec": _eval_time_sec,
                    "total_time_sec": _train_time_sec + _eval_time_sec,
                })
            if FLAGS.wandb:
                wandb.log({"math500_pass@1": pass_at_1})

        if getattr(FLAGS, 'push_to_hub', False):
            _hub_model_path = locals().get('_math500_model_path', None) or saved_pruned_model_path
            if _hub_model_path and os.path.isfile(os.path.join(_hub_model_path, "config.json")):
                from huggingface_hub import HfApi
                # Auto-generate repo id if not specified
                _hub_repo = FLAGS.hub_model_id if FLAGS.hub_model_id else None
                if not _hub_repo:
                    _base_model = FLAGS.model.rstrip('/').split('/')[-1]
                    _sparsity_tag = f"s{int(FLAGS.sparsity_ratio * 100)}pct"
                    _method_tag = "elsa-hybrid-kd" if getattr(FLAGS, 'do_kd_admm', False) and getattr(FLAGS, 'kd_use_cot_dataset', False) \
                        else "elsa-kd" if getattr(FLAGS, 'do_kd_admm', False) \
                        else "elsa-offpolicy-kd" if getattr(FLAGS, 'do_offpolicy_kd_admm', False) \
                        else "elsa-ntp-cot" if getattr(FLAGS, 'dataset', '') == 'math_cot' \
                        else "elsa-ntp"
                    def _fmt_float(v):
                        s = f"{v:.0e}"  # e.g. "1e-05" → normalize
                        return s.replace("e-0", "e-").replace("e+0", "e")
                    _lr_tag = f"lr{_fmt_float(FLAGS.admm_lr)}"
                    _lmda_tag = f"lmda{_fmt_float(FLAGS.admm_lmda)}"
                    _hub_repo = f"cosmos1030/{_base_model}-{_method_tag}-{_sparsity_tag}-{_lr_tag}-{_lmda_tag}"
                logging.info(f"Uploading model to HuggingFace Hub: {_hub_repo}")
                api = HfApi()
                api.create_repo(repo_id=_hub_repo, exist_ok=True)
                api.upload_folder(
                    folder_path=_hub_model_path,
                    repo_id=_hub_repo,
                    commit_message=f"ELSA pruned: sparsity={FLAGS.sparsity_ratio}, lr={FLAGS.admm_lr}, lmda={FLAGS.admm_lmda}",
                )
                logging.info(f"Uploaded to https://huggingface.co/{_hub_repo}")
                if FLAGS.wandb:
                    wandb.log({"hub_model_id": _hub_repo})
            else:
                logging.warning("push_to_hub=True but no saved model path found. Skipping upload.")


if __name__ == '__main__':
    flags.DEFINE_string('model', 'facebook/opt-125m', 'model to prune. model name (hf repo) or local path to model snapshot')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model.')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.6, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2", "math_trace", "code_trace", "math_prompt", "math_cot"], 'Calibration dataset.')
    flags.DEFINE_string('data_path', None , 'Path to local snapshot (e.g., huggingface/hub/allenai-c4/snapshot/hash..)')

    # Global ADMM hyperparams
    flags.DEFINE_float('admm_beta1', 0.9, 'Beta1 for ADMM Adam optimizer.')
    flags.DEFINE_float('admm_beta2', 0.95, 'Beta2 for ADMM Adam optimizer.')
    flags.DEFINE_integer('admm_num_train_samples', 4, 'Number of training samples for ADMM.')
    flags.DEFINE_integer('admm_num_eval_samples', 4, 'Number of evaluation samples for ADMM.')
    flags.DEFINE_bool('admm_save_inputs', False , 'whether to save tokenized inputs as a cache')
    flags.DEFINE_string('admm_save_path', None, 'Path to save ADMM training results and checkpoints.')
    flags.DEFINE_bool('save_model',False, 'Whether to save the pruned model after ADMM training.')

    # Training Loop Config
    flags.DEFINE_integer('admm_epochs', 1, 'Number of epochs for ADMM training.')
    flags.DEFINE_integer('admm_steps', 10, 'Max steps for ADMM training. Overrides admm_epochs if > 0.')
    flags.DEFINE_integer('admm_batch_size', 2, 'Batch size for ADMM training, per device.')
    flags.DEFINE_integer('admm_gradient_accumulation_steps', 1, 'Gradient accumulation steps for ADMM.')
    flags.DEFINE_bool('admm_gradient_checkpointing', False, 'Use gradient checkpointing for ADMM training. Set False when using FSDP')
    flags.DEFINE_float('admm_lr', 2e-4, 'Learning rate for ADMM base optimizer.')
    flags.DEFINE_string('admm_lr_scheduler', 'linear', 'Learning rate scheduler type for ADMM.')
    flags.DEFINE_integer('admm_warmup_steps', 0, 'Warmup steps for ADMM learning rate scheduler.')
    flags.DEFINE_float('admm_weight_decay', 0.0, 'Weight decay for ADMM base optimizer.')
    flags.DEFINE_enum('admm_precision', 'bf16', ['fp32', 'fp16', 'bf16'], 'Precision for ADMM training (fp16/bf16 enables Trainer autocast).')
    flags.DEFINE_enum('admm_projection_mode', 'identity', ['identity', 'momentum'], 'objective-aware projection for ADMM.')
    flags.DEFINE_bool('admm_projection_bias_correction', False, 'Whether to use bias correction in obejctive-aware ADMM projection.')

    # ADMM Specific Config
    flags.DEFINE_float('admm_lmda', 0.01, 'Lambda penalty parameter for ADMM (for constant schedule).')
    flags.DEFINE_float('admm_init_lmda', 0.0, 'Initial lambda value for ADMM scheduling.')
    flags.DEFINE_float('admm_final_lmda', 0.01, 'Final lambda value for ADMM scheduling.')
    flags.DEFINE_bool('admm_init_lambda_from_inv_resid', False, 'Initialize lambda from inverse of initial residual.')
    flags.DEFINE_enum('admm_lmda_schedule_mode', 'constant', ['constant', 'linear', 'exponential', 'cosine'], 'Mode for lambda schedule (e.g., linear, cosine).')
    flags.DEFINE_integer('admm_interval', 2, 'Interval for ADMM projection and dual updates.')
    flags.DEFINE_enum('admm_base_optimizer', 'adam', ['adam','adamw','adam8bit','adam4bit','sgd'], 'Base optimizer for ADMM primal update.')
    flags.DEFINE_enum('admm_dual_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM dual variable (fp32 or bf16).')
    flags.DEFINE_enum('admm_split_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM split variable (fp32 or bf16).')
    flags.DEFINE_bool('admm_nonuniform_sparsity', False, 'Whether to use non-uniform sparsity based on sensitivity scores in ADMM.')
    flags.DEFINE_string('admm_nonuniform_sparsity_config_file', None, 'Path to non-uniform sparsity configuration file (JSON format).')
    # GMP (BEST-style)
    flags.DEFINE_bool('do_gmp', False, 'Use BEST-style gradual magnitude pruning with Fisher importance.')
    flags.DEFINE_integer('gmp_steps', 4096, 'Total training steps for GMP.')
    flags.DEFINE_integer('gmp_batch_size', 1, 'Per-device batch size for GMP.')
    flags.DEFINE_integer('gmp_grad_accum', 8, 'Gradient accumulation steps for GMP.')
    flags.DEFINE_float('gmp_lr', 1e-5, 'Peak learning rate for GMP.')
    flags.DEFINE_float('gmp_warmup_ratio', 0.05, 'Fraction of steps for LR warmup in GMP.')
    flags.DEFINE_integer('gmp_mask_interval', 32, 'Steps between mask updates in GMP.')
    flags.DEFINE_float('gmp_fisher_beta', 0.999, 'EMA beta for Fisher diagonal accumulation.')
    flags.DEFINE_string('gmp_save_path', '/home1/doyoonkim/projects/elsa/models', 'Directory to save GMP pruned model.')
    flags.DEFINE_integer('gmp_max_prompt_len', 512, 'Max prompt length for GMP NTP dataset.')
    flags.DEFINE_integer('gmp_max_seq_len', 512, 'Max CoT sequence length for GMP NTP dataset.')

    # KD-ADMM: on-policy distillation inside ADMM loop
    flags.DEFINE_bool('do_kd_admm', False, 'Use on-policy KD loss inside ADMM instead of NTP.')
    flags.DEFINE_bool('do_offpolicy_kd_admm', False, 'Use off-policy KD (dataset CoT as teacher targets) inside ADMM instead of NTP.')
    flags.DEFINE_string('kd_data_path', None, 'Path to math prompts JSONL for KD-ADMM.')
    flags.DEFINE_integer('kd_max_prompt_len', 512, 'Max prompt length for KD-ADMM.')
    flags.DEFINE_integer('kd_max_new_tokens', 512, 'Max new tokens for on-policy generation.')
    flags.DEFINE_float('kd_temperature', 1.0, 'Temperature for generation and KD loss.')
    flags.DEFINE_integer('kd_nsamples', 0, 'Number of prompts to sample (0 = use all).')
    flags.DEFINE_float('kd_ntp_lambda', 0.0, 'Weight of NTP loss on prompt tokens added to KD loss (0 = KD only).')
    flags.DEFINE_integer('kd_topk', 50, 'Top-k vocab filtering for KD loss (0 = full vocab).')
    flags.DEFINE_integer('kd_interval', 1, 'Run on-policy KD generation every N steps (1 = every step).')
    flags.DEFINE_float('kd_lambda', 1.0, 'Weight of KD loss when combined with NTP loss in hybrid mode.')
    flags.DEFINE_bool('kd_use_vllm', False, 'Use vLLM for on-policy student rollout generation (faster for large models).')
    flags.DEFINE_bool('kd_generate_with_teacher', False, 'Generate rollouts with teacher (dense) instead of student. Ablation for on-policy vs teacher-generated rollouts.')
    flags.DEFINE_bool('kd_forward_kl', False, 'Use forward KL D(teacher||student) instead of reverse KL D(student||teacher).')
    flags.DEFINE_float('kd_vllm_gpu_memory_utilization', 0.3, 'vLLM gpu_memory_utilization for rollout engine.')
    flags.DEFINE_integer('kd_vllm_max_model_len', 0, 'vLLM max_model_len (0 = auto: kd_max_new_tokens + 1024).')
    flags.DEFINE_bool('kd_use_cot_dataset', False, 'Use MathCotKDDataset (provides CoT NTP labels + prompt for KD).')
    flags.DEFINE_bool('kd_use_random_cot_ntp', False, 'Use random 2048-token CoT windows for NTP; use separate MathPromptDataset for KD prompts.')
    flags.DEFINE_integer('kd_step_interval', 1, 'Apply KD loss every N optimizer steps (1=every step). Reduces teacher forward cost.')
    flags.DEFINE_string('kd_ntp_dataset', 'math_cot', 'Dataset for NTP in random CoT mode: math_cot or c4.')
    flags.DEFINE_integer('kd_buffer_size', 0, 'Rollout buffer size: generate this many prompts in one vLLM batch (0 = disabled).')
    flags.DEFINE_integer('kd_buffer_refresh_interval', 32, 'Refresh rollout buffer every N steps (default: align with admm_interval).')

    # On-policy distillation (legacy post-ADMM phase)
    flags.DEFINE_bool('do_distill', False, 'Whether to perform on-policy distillation after retraining.')
    flags.DEFINE_float('distill_lr', 1e-5, 'Learning rate for distillation.')
    flags.DEFINE_integer('distill_steps', 50, 'Number of distillation steps.')
    flags.DEFINE_integer('distill_batch_size', 2, 'Batch size per device for distillation.')
    flags.DEFINE_float('distill_temp', 1.0, 'Temperature for distillation.')
    flags.DEFINE_float('distill_alpha', 1.0, 'Alpha for KL (1.0 for reverse KL).')
    flags.DEFINE_integer('distill_topk', None, 'Top-k for distillation. If None, use full logits.')
    flags.DEFINE_bool('distill_add_tail', True, 'Whether to add tail for top-k distillation.')

    # Logging & Evaluation
    flags.DEFINE_integer('admm_logging_steps', 1, 'Logging step interval for ADMM training.')
    flags.DEFINE_integer('admm_eval_steps', 1, 'Evaluation step interval for ADMM training.')

    flags.DEFINE_bool('data_ablation', False, 'Whether to use data ablation, for section 5.5. If True, we fix the step size and control the number of train samples with --admm_num_train_samples.')
    flags.DEFINE_bool('eval_zero_shot', True, 'Whether to evaluate zero-shot performance.')
    flags.DEFINE_bool('eval_math500', False, 'Whether to run MATH-500 pass@1 eval after pruning (via lighteval+vLLM).')
    flags.DEFINE_string('math500_model_path', None, 'Path to saved pruned model for lighteval eval. If None, saves model to temp dir.')
    flags.DEFINE_integer('math500_max_new_tokens', 4096, 'max_new_tokens for MATH-500 generation.')
    flags.DEFINE_integer('math500_max_samples', 0, 'Max samples for MATH-500 eval (0 = all 500).')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', None, 'wandb project name.')
    flags.DEFINE_bool('push_to_hub', False, 'Whether to push the pruned model to HuggingFace Hub after eval.')
    flags.DEFINE_string('hub_model_id', None, 'HuggingFace Hub repo id (e.g. username/model-name) to push pruned model.')
    app.run(main)
