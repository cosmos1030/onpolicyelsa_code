import time
import numpy as np
import torch
from transformers import AutoTokenizer
from lib.prune import globalprune_admm
from lib.eval import eval_ppl, eval_zero_shot
from lib.lighteval_math500 import run_lighteval_math500
from lib.lighteval_bench import run_lighteval_bench
from lib.utils import check_sparsity, get_llm
from lib.on_policy_distill import run_on_policy_distillation
from lib.gkd_admm import globalprune_admm_kd
from lib.gmp_trainer import globalprune_gmp
from lib.grpo_opkd import run_grpo_opkd
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


def _build_run_name(FLAGS):
    """Build a descriptive run name from FLAGS (call after sweep config is applied)."""
    F = FLAGS
    if getattr(F, 'do_grpo_opkd', False):
        sp = int(getattr(F, 'sparsity_ratio', 0) * 100)
        lr = getattr(F, 'lr', 0)
        steps = getattr(F, 'steps', 0)
        G = getattr(F, 'gmp_grpo_num_rollouts', 4)
        lam = getattr(F, 'gmp_grpo_lambda', 1.0)
        eps = getattr(F, 'gmp_grpo_eps_clip', 0.2)
        return f"grpo_opkd_s{sp}pct_G{G}_lam{lam}_eps{eps}_lr{lr}_{steps}steps"
    elif getattr(F, 'do_gmp', False):
        sp = int(getattr(F, 'sparsity_ratio', 0) * 100)
        lr = getattr(F, 'lr', 0)
        steps = getattr(F, 'steps', 0)
        prune_end = getattr(F, 'gmp_pruning_end_ratio', 1.0)

        kd_only   = getattr(F, 'gmp_kd_only', False)
        kd_lam    = getattr(F, 'gmp_kd_lambda', 0.0) or 0.0
        onpol_lam = getattr(F, 'gmp_onpolicy_kd_lambda', 0.0) or 0.0
        anc_lam   = getattr(F, 'gmp_anchor_kd_lambda', 0.0) or 0.0
        anc_pfx   = getattr(F, 'gmp_anchor_prefix_len', 0)
        onpol_tok = getattr(F, 'gmp_onpolicy_max_new_tokens', 0)

        if getattr(F, 'gmp_teacher_seqkd', False):
            method = "teacher_seqkd"
        elif anc_lam > 0:
            method = f"anchor_lam{anc_lam}_pfx{anc_pfx}"
        elif onpol_lam > 0:
            method = f"onpol_lam{onpol_lam}_tok{onpol_tok}"
        elif kd_lam > 0:
            method = f"{'kdonly' if kd_only else 'kd'}{kd_lam}"
        else:
            method = "ntp"

        name = f"gmp_s{sp}pct_{method}_lr{lr}_{steps}steps"
        if prune_end < 1.0:
            name += f"_prune{int(prune_end*100)}pct"
        mi = getattr(F, 'gmp_mask_interval', 32)
        if mi != 32:
            name += f"_mi{mi}"

        sparsity_type = getattr(F, 'sparsity_type', 'unstructured')
        if sparsity_type != 'unstructured':
            name += f"_{sparsity_type.replace(':', 'to')}"

        if getattr(F, 'gmp_tr_enabled', False):
            name += f"_kl{getattr(F, 'gmp_tr_kl_threshold', 0.01)}"
            dmin = getattr(F, 'gmp_tr_delta_min', 0.005)
            if dmin != 0.005:
                name += f"_dmin{dmin}"

        l1_lam = getattr(F, 'gmp_l1_lambda', 0.0) or 0.0
        if l1_lam > 0:
            l1_structured = getattr(F, 'gmp_l1_structured', True)
            if l1_structured:
                name += f"_l1struct{l1_lam}"
            else:
                name += f"_l1{getattr(F, 'gmp_l1_mode', 'plain')}{l1_lam}"
                if getattr(F, 'gmp_l1_open_groups_only', False):
                    name += "_openonly"

        lr_sched = getattr(F, 'lr_scheduler', 'cosine')
        if lr_sched in ('constant', 'constant_with_warmup'):
            name += "_constlr"

        if getattr(F, 'gmp_pcg_correct', False):
            name += f"_pcg{getattr(F, 'gmp_pcg_maxiter', 5)}"
            if getattr(F, 'gmp_pcg_sequential', False):
                name += "seq"

        return name

    elif getattr(F, 'do_kd_admm', False):
        sp   = int(getattr(F, 'sparsity_ratio', 0) * 100)
        lr   = F.lr
        lmda = F.admm_lmda
        steps = F.steps
        trz   = getattr(F, 'admm_tr_z_proj', False)
        kl_th = getattr(F, 'admm_tr_kl_threshold', 0.5) if trz else None
        lasso = getattr(F, 'admm_lasso_lmda', 0.0) or 0.0

        if getattr(F, 'kd_triple_loss', False):
            ntp_w  = getattr(F, 'kd_ntp_lambda', 0.33)
            kd_w   = getattr(F, 'kd_lambda', 0.33)
            opkd_w = getattr(F, 'kd_opkd_lambda', 0.33)
            method = f"triple_ntp{ntp_w}_dskd{kd_w}_opkd{opkd_w}"
        elif getattr(F, 'kd_offpolicy_ntp', False):
            ntp_w = getattr(F, 'kd_ntp_lambda', 0.5)
            kd_w  = getattr(F, 'kd_lambda', 0.5)
            method = f"ntp{ntp_w}_dskd{kd_w}"
        elif getattr(F, 'kd_use_cot_dataset', False):
            method = f"ntp_opkd_kdlam{F.kd_lambda}"
        else:
            method = f"opkd_kdlam{F.kd_lambda}"

        name = f"admm_{method}_s{sp}pct_lr{lr}_lmda{lmda}"
        if trz:
            name += f"_trz{kl_th}"
        if lasso > 0:
            name += f"_lasso{lasso}"
        name += f"_{steps}steps"
        return name
    elif getattr(F, 'do_offpolicy_kd_admm', False):
        return (f"offpolicy_kd_admm_s{F.sparsity_ratio}_lr{F.lr}"
                f"_lmda{F.admm_lmda}_steps{F.steps}")
    else:
        name = (f"ntp_admm_s{F.sparsity_ratio}_lr{F.lr}"
                f"_lmda{F.admm_lmda}")
        if getattr(F, 'admm_tr_z_proj', False):
            mode = getattr(F, 'admm_z_schedule_mode', 'trust_region')
            if mode == 'cubic':
                name += f"_cubic{getattr(F, 'admm_cubic_steps', 2048)}"
            elif mode == 'cosine':
                name += f"_cosinez{getattr(F, 'admm_cubic_steps', 2048)}"
            else:
                name += f"_trz{getattr(F, 'admm_tr_kl_threshold', 0.5)}"
        name += f"_steps{F.steps}"
        return name


def main(argv):
    global FLAGS
    arguments = FLAGS.flag_values_dict()
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    _teacher_model_path = getattr(FLAGS, 'gmp_teacher_model', None) or FLAGS.model

    _prebuilt_vllm_engine = None
    _prebuilt_vllm_params = None
    _prebuilt_opd_vllm_engine = None
    _prebuilt_opd_vllm_params = None
    _use_fsdp_opkd = (
        is_distributed
        and getattr(FLAGS, 'do_gmp', False)
        and getattr(FLAGS, 'gmp_use_fsdp', False)
        and getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0) > 0
    )
    # OPD's vLLM always runs isolated on a dedicated GPU (index = world_size,
    # i.e. one GPU beyond the training world — world_size=1 for single-GPU
    # training). This doesn't require FSDP/distributed training: a single
    # training GPU + one dedicated vLLM GPU works the same way, just with
    # world_size=1 and no dist.barrier() needed (see below).
    _use_admm_opd_dedicated = (
        (getattr(FLAGS, 'do_kd_admm', False) or getattr(FLAGS, 'do_offpolicy_kd_admm', False))
        and getattr(FLAGS, 'opd_enabled', False)
    )

    # For FSDP + OPKD: TRL-style isolation — vLLM runs in a separate subprocess
    # (multiprocessing spawn) so it has its own CUDA context and torch.distributed
    # state. No NCCL groups are shared with the training process group.
    # init_process_group first, then rank 0 launches vLLM and signals via dist.barrier().
    _vllm_eager = getattr(FLAGS, 'gmp_opkd_vllm_enforce_eager', False)

    # Pin this process to its own GPU BEFORE any collective (including init_process_group).
    # Without this, NCCL has to guess the intended device (see PyTorch's own warning:
    # "No device id is provided via init_process_group or barrier... using the current
    # device set by the user"), which is fragile and caused a real crash on n42
    # (5-GPU FSDP+dedicated-vLLM job: ranks 1-3 hit "device=4, num_gpus=4" CUDA assert
    # at the first real CUDA touch after the vLLM subprocess launch + barrier).
    if is_distributed:
        torch.cuda.set_device(local_rank)
    if is_distributed:
        # NCCL's default collective timeout is 10 minutes. The sharded post-training
        # zero-shot eval (each rank evaluates a different, unevenly-sized subset of
        # tasks) can easily exceed that on the all_gather_object that collects
        # results — a slow rank (e.g. one running hellaswag) blows past 10 minutes
        # while a fast rank sits idle at the same collective, and the watchdog
        # aborts the whole process group. Use a generous timeout to cover training
        # collectives (which are fast and frequent) as well as these occasional
        # long, imbalanced eval-time collectives.
        import datetime as _datetime
        dist.init_process_group(
            backend='nccl',
            device_id=torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else None,
            timeout=_datetime.timedelta(hours=2),
        )

    if _use_fsdp_opkd:
        _vllm_gpu_mem = getattr(FLAGS, 'gmp_opkd_vllm_gpu_mem', 0.25)
        _vllm_max_prompt = getattr(FLAGS, 'gmp_max_prompt_len', 512)
        _vllm_max_new = getattr(FLAGS, 'gmp_onpolicy_max_new_tokens', 256)
        _vllm_temp = getattr(FLAGS, 'gmp_onpolicy_temp', 0.6)

        if local_rank == 0:
            from lib.vllm_proc import launch_vllm_server as _launch_vllm
            # vLLM runs as a fully independent OS process (subprocess.Popen,
            # not a multiprocessing.Process child) with CUDA_VISIBLE_DEVICES
            # set to its own GPU(s) -- see lib/vllm_proc.py's module docstring
            # for why: nesting vLLM's distributed runtime inside torchrun's
            # process tree (the previous approach) hit a daemon/fork/rendezvous
            # bug chain once tensor_parallel_size>1 was introduced.
            # enforce_eager=True is mandatory: without it vLLM pre-allocates CUDA
            # graphs using (gpu_memory_utilization × GPU_MEM) of address space on the
            # vLLM GPU.  That allocation gets P2P-mapped into the training GPU's
            # virtual address space via NCCL peer access, consuming ~18 GB and
            # causing OOM in the training process even though both are on separate
            # physical GPUs.  Eager mode allocates only model weights + actual KV
            # cache, keeping the vLLM GPU footprint small.
            _vllm_gpu_index = getattr(FLAGS, 'gmp_opkd_vllm_gpu_index', -1)
            _vllm_tp_size = getattr(FLAGS, 'gmp_opkd_vllm_tp_size', 1)
            if _vllm_tp_size > 1:
                _vllm_cuda_dev = ','.join(str(i) for i in range(_vllm_tp_size))
            else:
                _vllm_cuda_dev = str(world_size) if _vllm_gpu_index < 0 else str(_vllm_gpu_index)
            logging.info(
                f"[rank 0] Launching standalone vLLM server on GPU(s) {_vllm_cuda_dev} "
                f"(tp_size={_vllm_tp_size}, training ranks on GPUs 0-{world_size-1}), gpu_mem={_vllm_gpu_mem}")
            _prebuilt_vllm_engine = _launch_vllm(
                FLAGS.model,
                cuda_device_str=_vllm_cuda_dev,
                gpu_mem=_vllm_gpu_mem,
                max_len=_vllm_max_new + _vllm_max_prompt,
                enforce_eager=True,
                default_max_new=_vllm_max_new,
                default_temp=_vllm_temp,
                startup_timeout=480,
                tensor_parallel_size=_vllm_tp_size,
            )
            _prebuilt_vllm_params = None
            logging.info(f"[rank 0] vLLM server ready — signaling via dist.barrier")
        else:
            logging.info(f"[rank {local_rank}] waiting for rank 0 vLLM via dist.barrier")
        # barrier: rank 1 waits here until rank 0 finishes vLLM launch
        dist.barrier()
        if local_rank != 0:
            logging.info(f"[rank {local_rank}] vLLM barrier passed — proceeding")

    # ADMM + OPD: vLLM subprocess on dedicated GPU (index = world_size, e.g.
    # GPU 1 for single-GPU training). Requires requesting world_size+1 GPUs
    # in the SLURM script.
    if _use_admm_opd_dedicated:
        _opd_gpu_mem = getattr(FLAGS, 'opd_vllm_gpu_mem', 0.25)
        _opd_max_prompt = getattr(FLAGS, 'kd_max_prompt_len', 512)
        _opd_max_new = getattr(FLAGS, 'opd_vllm_max_tokens', 256)
        _vllm_cuda_dev = str(world_size)
        if local_rank == 0:
            from lib.vllm_proc import launch_vllm_server as _launch_vllm_opd
            logging.info(
                f"[rank 0] OPD: launching vLLM server on GPU {_vllm_cuda_dev} "
                f"(training ranks 0-{world_size-1}), gpu_mem={_opd_gpu_mem}")
            _prebuilt_opd_vllm_engine = _launch_vllm_opd(
                FLAGS.model,
                cuda_device_str=_vllm_cuda_dev,
                gpu_mem=_opd_gpu_mem,
                max_len=_opd_max_new + _opd_max_prompt,
                enforce_eager=True,
                default_max_new=_opd_max_new,
                default_temp=0.6,
                startup_timeout=300,
            )
            logging.info("[rank 0] OPD vLLM subprocess ready"
                         + (" — signaling via dist.barrier" if is_distributed else ""))
        else:
            logging.info(f"[rank {local_rank}] OPD: waiting for rank 0 vLLM via dist.barrier")
        if is_distributed:
            dist.barrier()
            if local_rank != 0:
                logging.info(f"[rank {local_rank}] OPD vLLM barrier passed — proceeding")

    if FLAGS.wandb and local_rank == 0:
        if getattr(FLAGS, 'do_grpo_opkd', False):
            group = "grpo_opkd"
        elif getattr(FLAGS, 'do_gmp', False):
            group = "gmp"
        elif FLAGS.do_kd_admm:
            group = "onpolicy_kd_admm"
        elif getattr(FLAGS, 'do_offpolicy_kd_admm', False):
            group = "offpolicy_kd_admm"
        else:
            group = "ntp_admm"

        try:
            wandb.init(
                project=FLAGS.wandb_project,
                group=group,
                name="pending",
                save_code=True,
            )
        except Exception as _wandb_e:
            logging.warning(f"wandb online init failed ({_wandb_e}), retrying in offline mode")
            import os as _os
            _os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=FLAGS.wandb_project,
                group=group,
                name="pending",
                save_code=False,
            )

        if not dict(wandb.config):
            wandb.config.update(arguments)
        else:
            updated_args = {
                k: wandb.config.get(k, v) for k, v in arguments.items()
            }
            FLAGS = type('FLAGS', (), updated_args)()
            logging.info(f"Updated args with wandb.config: {FLAGS}")

        # Build run name after FLAGS is updated with sweep config
        run_name = _build_run_name(FLAGS)
        if getattr(FLAGS, 'run_name_suffix', ''):
            run_name += f"_{FLAGS.run_name_suffix}"
        wandb.run.name = run_name
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
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model, use_fast=True)

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    model = model.to('cpu')
    model.config.use_cache = False

    logging.info(f"Process {local_rank} uses device {device}")

    saved_pruned_model_path = None
    _train_time_sec = 0.0
    if FLAGS.sparsity_ratio != 0 or getattr(FLAGS, 'do_gmp', False) or getattr(FLAGS, 'do_grpo_opkd', False) or getattr(FLAGS, 'do_chunk_grpo_opkd', False):
        logging.info("pruning starts")
        _t_train_start = time.time()
        if getattr(FLAGS, 'do_chunk_grpo_opkd', False):
            model.to(device)
            from lib.gkd_admm_trainer import MixedTextDataset, MixedPromptDataset, collate_prompts
            from lib.grpo_opkd import run_chunk_grpo_opkd
            train_dataset = MixedTextDataset(
                jsonl_path=FLAGS.data_path,
                tokenizer=tokenizer,
                max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                max_len=getattr(FLAGS, 'seqlen', 2048),
                append_eos=getattr(FLAGS, 'cot_append_eos', False),
            )
            prompt_path = getattr(FLAGS, 'gmp_prompt_path', None) or FLAGS.data_path
            prompt_dataset = MixedPromptDataset(
                jsonl_path=prompt_path,
                tokenizer=tokenizer,
                max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
            )
            grpo_teacher = get_llm(_teacher_model_path, FLAGS.seqlen)
            grpo_teacher.to(device)
            grpo_teacher.eval()
            for p in grpo_teacher.parameters():
                p.requires_grad_(False)
            saved_pruned_model_path = run_chunk_grpo_opkd(
                model, grpo_teacher, tokenizer, train_dataset, prompt_dataset, FLAGS)
            del grpo_teacher
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'do_grpo_opkd', False):
            model.to(device)
            from lib.gkd_admm_trainer import (
                MixedTextDataset, MixedPromptDataset, MathPromptWithAnswerDataset, collate_prompts,
            )
            train_dataset = MixedTextDataset(
                jsonl_path=FLAGS.data_path,
                tokenizer=tokenizer,
                max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                max_len=getattr(FLAGS, 'seqlen', 2048),
                append_eos=getattr(FLAGS, 'cot_append_eos', False),
            )
            prompt_path = getattr(FLAGS, 'gmp_prompt_path', None) or FLAGS.data_path
            if getattr(FLAGS, 'gmp_correctness_reward', False):
                # Correctness reward: need gold answers → use CoT data
                prompt_dataset = MathPromptWithAnswerDataset(
                    jsonl_path=prompt_path,
                    tokenizer=tokenizer,
                    max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                )
                grpo_teacher = None  # teacher not needed for correctness reward
            else:
                prompt_dataset = MixedPromptDataset(
                    jsonl_path=prompt_path,
                    tokenizer=tokenizer,
                    max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                )
                grpo_teacher = get_llm(_teacher_model_path, FLAGS.seqlen)
                grpo_teacher.to(device)
                grpo_teacher.eval()
                for p in grpo_teacher.parameters():
                    p.requires_grad_(False)
            saved_pruned_model_path = run_grpo_opkd(
                model, grpo_teacher, tokenizer, train_dataset, prompt_dataset, FLAGS)
            if grpo_teacher is not None:
                del grpo_teacher
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'do_gmp', False):
            model.to(device)

            # ── Optional FSDP wrapping for multi-GPU GMP ─────────────────────
            gmp_use_fsdp = getattr(FLAGS, 'gmp_use_fsdp', False)
            if gmp_use_fsdp and is_distributed:
                from torch.distributed.fsdp import (
                    FullyShardedDataParallel as FSDP,
                    MixedPrecision,
                    ShardingStrategy,
                )
                from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
                import functools
                _block_cls = type(model.model.layers[0])
                _wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    transformer_layer_cls={_block_cls},
                )
                _mp = MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
                model = FSDP(
                    model,
                    auto_wrap_policy=_wrap_policy,
                    mixed_precision=_mp,
                    sharding_strategy=ShardingStrategy.FULL_SHARD,
                    use_orig_params=True,   # preserve param identity for optimizer state lookup
                    device_id=torch.cuda.current_device(),
                )
                logging.info(f"[rank {local_rank}] Model wrapped with FSDP (use_orig_params=True)")
            print(f"[DBG main] rank={local_rank} post-FSDP, loading dataset", flush=True)

            if getattr(FLAGS, 'gmp_random_cot_ntp', False):
                from lib.data import get_dataset
                train_dataset = get_dataset(
                    dataset_name='mixed_cot',
                    tokenizer=tokenizer,
                    nsamples=4096,
                    seed=FLAGS.seed,
                    seqlen=getattr(FLAGS, 'seqlen', 2048),
                    data_type='train',
                    data_path=FLAGS.data_path,
                )
            else:
                from lib.gkd_admm_trainer import MixedTextDataset
                train_dataset = MixedTextDataset(
                    jsonl_path=FLAGS.data_path,
                    tokenizer=tokenizer,
                    max_prompt_len=getattr(FLAGS, 'gmp_max_prompt_len', 512),
                    max_len=getattr(FLAGS, 'seqlen', 2048),
                    append_eos=getattr(FLAGS, 'cot_append_eos', False),
                    nsamples=getattr(FLAGS, 'kd_nsamples', 0) or None,
                )
            print(f"[DBG main] rank={local_rank} dataset loaded, loading teacher", flush=True)
            gmp_teacher = None
            if getattr(FLAGS, 'gmp_kd_lambda', 0.0) > 0 or getattr(FLAGS, 'gmp_hidden_lambda', 0.0) > 0 or getattr(FLAGS, 'gmp_onpolicy_kd_lambda', 0.0) > 0 or getattr(FLAGS, 'gmp_anchor_kd_lambda', 0.0) > 0 or getattr(FLAGS, 'gmp_teacher_seqkd', False) or getattr(FLAGS, 'gmp_blockwise_squarehead', False):
                print(f"[DBG main] rank={local_rank} get_llm teacher start", flush=True)
                gmp_teacher = get_llm(_teacher_model_path, FLAGS.seqlen)
                print(f"[DBG main] rank={local_rank} get_llm teacher done, .to(device)", flush=True)
                gmp_teacher.to(device)
                print(f"[DBG main] rank={local_rank} teacher to(device) done", flush=True)
                gmp_teacher.eval()
                for p in gmp_teacher.parameters():
                    p.requires_grad_(False)
                if gmp_use_fsdp and is_distributed:
                    gmp_teacher = gmp_teacher.to(torch.bfloat16)
                    logging.info(f"[rank {local_rank}] Teacher kept as plain model (no FSDP) — no NCCL needed for eval-only teacher")
            # DPO dense model: reuse teacher if loaded, else load separately
            print(f"[DBG main] rank={local_rank} before gmp_train call", flush=True)
            gmp_dpo_dense = None
            if getattr(FLAGS, 'gmp_dpo_lambda', 0.0) > 0:
                if gmp_teacher is not None:
                    gmp_dpo_dense = gmp_teacher  # reuse
                else:
                    gmp_dpo_dense = get_llm(_teacher_model_path, FLAGS.seqlen)
                    gmp_dpo_dense.to(device)
                    gmp_dpo_dense.eval()
                    for p in gmp_dpo_dense.parameters():
                        p.requires_grad_(False)
            saved_pruned_model_path = globalprune_gmp(
                model, tokenizer, train_dataset, FLAGS,
                teacher_model=gmp_teacher, dpo_dense_model=gmp_dpo_dense,
                prebuilt_vllm_engine=_prebuilt_vllm_engine,
                prebuilt_vllm_params=_prebuilt_vllm_params)
            gmp_teacher = None
            gmp_dpo_dense = None
            if _prebuilt_vllm_engine is not None and hasattr(_prebuilt_vllm_engine, 'shutdown'):
                _prebuilt_vllm_engine.shutdown()
                _prebuilt_vllm_engine = None
            torch.cuda.empty_cache()
        elif FLAGS.do_kd_admm:
            teacher_model = get_llm(_teacher_model_path, FLAGS.seqlen)
            teacher_model.to(device)
            saved_pruned_model_path = globalprune_admm_kd(
                FLAGS, model, teacher_model, tokenizer, device,
                prebuilt_opd_vllm_engine=_prebuilt_opd_vllm_engine,
                prebuilt_opd_vllm_params=_prebuilt_opd_vllm_params,
                prune_n=prune_n, prune_m=prune_m,
            )
            del teacher_model
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'do_offpolicy_kd_admm', False):
            teacher_model = get_llm(_teacher_model_path, FLAGS.seqlen)
            teacher_model.to(device)
            saved_pruned_model_path = globalprune_admm_kd(
                FLAGS, model, teacher_model, tokenizer, device, offpolicy_kd=True,
                prebuilt_opd_vllm_engine=_prebuilt_opd_vllm_engine,
                prebuilt_opd_vllm_params=_prebuilt_opd_vllm_params,
                prune_n=prune_n, prune_m=prune_m,
            )
            del teacher_model
            torch.cuda.empty_cache()
        elif getattr(FLAGS, 'do_safe', False):
            from lib.prune import prune_safe
            model.seqlen = FLAGS.seqlen
            prune_safe(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
            if getattr(FLAGS, 'save_model', False) and getattr(FLAGS, 'admm_save_path', None):
                from pathlib import Path as _Path
                from datetime import datetime as _dt
                _model_name = FLAGS.model.split('/')[-1]
                _sdir = _Path(FLAGS.admm_save_path) / f"{_model_name}_safe_s{int(FLAGS.sparsity_ratio*100)}pct_{_dt.now().strftime('%Y%m%d_%H%M')}"
                _sdir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(_sdir))
                tokenizer.save_pretrained(str(_sdir))
                logging.info(f"Saved SAFE pruned model to {_sdir}")
                saved_pruned_model_path = str(_sdir)
        elif getattr(FLAGS, 'dataset', '') == 'mixed_cot':
            # NTP with full problem context: no teacher, no KD, uses MixedTextDataset
            saved_pruned_model_path = globalprune_admm_kd(FLAGS, model, None, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        else:
            saved_pruned_model_path = globalprune_admm(FLAGS, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        _train_time_sec = time.time() - _t_train_start
        logging.info(f"Training time: {_train_time_sec/3600:.2f}h")

    if local_rank == 0:
        logging.info("Pruning finished")

    if is_distributed:
        dist.barrier()
        # FLAGS are identical on all ranks; saved_pruned_model_path is only set on rank 0.
        # Use FLAGS-based condition so both ranks take the same branch.
        _uses_fsdp = getattr(FLAGS, 'do_gmp', False) or getattr(FLAGS, 'admm_use_fsdp', False) or getattr(FLAGS, 'gmp_use_fsdp', False)
        _fsdp_trainer_saved = _uses_fsdp and getattr(FLAGS, 'save_model', False)
        if _fsdp_trainer_saved:
            # Zero-shot eval used to be sharded across ranks (each rank running a
            # subset of the 9 tasks, then dist.all_gather_object to combine) — but
            # task sizes are wildly uneven (hellaswag/race have tens of thousands of
            # loglikelihood requests vs. a few hundred for boolq/rte), so one rank
            # routinely took far longer than the others. The idle ranks then sat in
            # all_gather_object past NCCL's watchdog timeout and the whole job died
            # (observed repeatedly, on 6+ different nodes across both 4B and 8B —
            # not a bad-node issue, a structural one). Only rank 0 now runs ALL 9
            # tasks sequentially; other ranks poll a sentinel file instead of
            # blocking in a collective, so no NCCL timeout is possible regardless of
            # how long the eval takes.
            _sentinel = None
            if getattr(FLAGS, 'eval_zero_shot', False) and saved_pruned_model_path:
                _sentinel = os.path.join(saved_pruned_model_path, ".zeroshot_done")
                if local_rank == 0:
                    del model
                    import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
                    _eval_model = get_llm(saved_pruned_model_path, FLAGS.seqlen)
                    _eval_model.to(device)
                    _eval_model.eval()
                    _all_tasks = ["boolq", "rte", "hellaswag", "winogrande", "arc_easy",
                                  "arc_challenge", "openbookqa", "piqa", "race"]
                    logging.info(f"[rank 0] Running full zero-shot suite (no sharding): {_all_tasks}")
                    _zs_results = eval_zero_shot(FLAGS, FLAGS.model, _eval_model, tokenizer, _all_tasks, 0, False)
                    logging.info(f"[FSDP eval] zero-shot results: {_zs_results}")
                    if FLAGS.wandb:
                        for task_name, metrics in _zs_results.items():
                            try:
                                acc = metrics.get('acc_norm,none', metrics.get('acc,none', metrics.get('acc', None)))
                                stderr = metrics.get('acc_norm_stderr,none', metrics.get('acc_stderr,none', metrics.get('acc_stderr', None)))
                                if acc is not None:
                                    wandb.log({f"global_admm/{task_name}_acc": acc})
                                if stderr is not None:
                                    wandb.log({f"global_admm/{task_name}_stderr": stderr})
                            except Exception as log_e:
                                logging.warning(f"Could not log zero-shot metric for {task_name}: {log_e}")
                    del _eval_model
                    _gc.collect(); torch.cuda.empty_cache()
                    with open(_sentinel, "w") as _f:
                        _f.write("done")
                else:
                    logging.info(f"[rank {local_rank}] waiting for rank 0 zero-shot eval to finish...")
                    while not os.path.exists(_sentinel):
                        time.sleep(5)
                model = None
                FLAGS.eval_zero_shot = False  # already handled above; skip the single-GPU path below

            # Trainer already saved via FSDP-aware save_model; both ranks destroy PG together.
            dist.destroy_process_group()
            import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
            if local_rank == 0 and saved_pruned_model_path:
                model = get_llm(saved_pruned_model_path, FLAGS.seqlen)
        elif _uses_fsdp:
            # FSDP used but model not saved — just tear down process group.
            # model must be set to None: the FSDP object is unusable after PG teardown
            # (all-gather would crash). Downstream eval block is guarded by `model is not None`.
            dist.destroy_process_group()
            import gc as _gc; _gc.collect(); torch.cuda.empty_cache()
            model = None
        else:
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
        teacher_model = get_llm(_teacher_model_path, FLAGS.seqlen)
        teacher_model.to(device)
        teacher_model.eval()

        # 2. 로드한 티처 모델을 인자로 명시적으로 넘겨줍니다.
        run_on_policy_distillation(FLAGS, model, teacher_model, tokenizer, device)

        # 3. 학습이 끝나면 티처를 메모리에서 해제하여 다음 단계(Eval 등)를 대비합니다.
        del teacher_model
        torch.cuda.empty_cache()

    if local_rank == 0 and model is not None:

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
        try:
            ppl_test = eval_ppl(FLAGS, model, tokenizer, device, data_path="/home1/doyoonkim/projects/elsa/data/c4")
            logging.info([(key,ppl) for key,ppl in ppl_test.items()])
            if FLAGS.wandb:
                wandb.log({"sparsity_ratio": sparsity_ratio, **{f"ppl_test({key})": value for key, value in ppl_test.items()}})
        except Exception as _ppl_e:
            logging.warning(f"PPL eval failed (network/cache issue?): {_ppl_e}")
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
                        acc = metrics.get('acc_norm,none', metrics.get('acc,none', metrics.get('acc', None)))
                        stderr = metrics.get('acc_norm_stderr,none', metrics.get('acc_stderr,none', metrics.get('acc_stderr', None)))
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
                seed=getattr(FLAGS, 'seed', None),
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

        ## Full benchmark suite evaluation via lighteval+vLLM
        if getattr(FLAGS, 'eval_full_bench', False):
            logging.info("--- Evaluating After Training (5-benchmark suite, lighteval+vLLM) ---")
            # Resolve saved model path (same logic as eval_math500)
            if FLAGS.math500_model_path:
                _bench_model_path = FLAGS.math500_model_path
            elif saved_pruned_model_path and os.path.isfile(os.path.join(saved_pruned_model_path, "config.json")):
                _bench_model_path = saved_pruned_model_path
            elif FLAGS.save_model and FLAGS.admm_save_path:
                import glob as _glob
                _subdirs = [
                    p for p in _glob.glob(os.path.join(FLAGS.admm_save_path, "*"))
                    if os.path.isfile(os.path.join(p, "config.json"))
                ]
                _subdirs = sorted(_subdirs, key=os.path.getmtime)
                if _subdirs:
                    _bench_model_path = _subdirs[-1]
                else:
                    import tempfile as _tmpfile
                    _bench_model_path = _tmpfile.mkdtemp(prefix="elsa_eval_")
                    model.save_pretrained(_bench_model_path)
                    tokenizer.save_pretrained(_bench_model_path)
            else:
                import tempfile as _tmpfile
                _bench_model_path = _tmpfile.mkdtemp(prefix="elsa_eval_")
                model.save_pretrained(_bench_model_path)
                tokenizer.save_pretrained(_bench_model_path)

            # Free all VRAM for vLLM subprocess (skip if already freed by eval_math500)
            if not FLAGS.eval_math500:
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
            logging.info(f"vLLM gpu_memory_utilization: {_vllm_gpu_util:.3f}")

            _out_base = os.path.join(_bench_model_path, "lighteval_bench")
            bench_metrics = run_lighteval_bench(
                model_path=_bench_model_path,
                out_base=_out_base,
                gpu_util=_vllm_gpu_util,
                tp_size=world_size,
                log_to_wandb=FLAGS.wandb,
                seed=FLAGS.seed,
                profile=FLAGS.eval_profile,
            )
            # Also log math500_pass@1 as top-level for sweep metric
            if "lighteval/math500" in bench_metrics and FLAGS.wandb:
                wandb.log({"math500_pass@1": bench_metrics["lighteval/math500"]})

        # Write eval context for SLURM-level lighteval (when eval_math500=false and FSDP)
        if not FLAGS.eval_math500 and saved_pruned_model_path and is_distributed and local_rank == 0:
            import json as _json
            _ctx = {
                "saved_model_path": saved_pruned_model_path,
                "wandb_run_id": wandb.run.id if FLAGS.wandb and wandb.run else None,
                "wandb_project": getattr(FLAGS, 'wandb_project', None),
                "train_time_sec": _train_time_sec,
            }
            _ctx_file = os.path.join(saved_pruned_model_path, ".eval_ctx.json")
            with open(_ctx_file, 'w') as _f:
                _json.dump(_ctx, _f)
            logging.info(f"Wrote eval context to {_ctx_file}")

        if getattr(FLAGS, 'push_to_hub', False):
            _hub_model_path = locals().get('_math500_model_path', None) or saved_pruned_model_path
            if _hub_model_path and os.path.isfile(os.path.join(_hub_model_path, "config.json")):
                from huggingface_hub import HfApi
                # Auto-generate repo id if not specified
                _hub_repo = FLAGS.hub_model_id if FLAGS.hub_model_id else None
                if not _hub_repo:
                    from datetime import datetime as _dt
                    _now = _dt.now().strftime("%Y%m%d_%H%M%S")
                    _sparsity_tag = f"s{int(FLAGS.sparsity_ratio * 100)}pct"
                    def _fmt_float(v):
                        s = f"{v:.0e}"
                        return s.replace("e-0", "e-").replace("e+0", "e")
                    if getattr(FLAGS, 'do_gmp', False):
                        _kd_tag = f"-kd{_fmt_float(getattr(FLAGS, 'gmp_kd_lambda', 0))}" if getattr(FLAGS, 'gmp_kd_lambda', 0) > 0 else ""
                        _method_tag = f"gmp{_kd_tag}"
                        _lr_tag = f"lr{_fmt_float(FLAGS.lr)}"
                        _hub_repo = f"cosmos1030/{_method_tag}-{_sparsity_tag}-{_lr_tag}_{_now}"
                    else:
                        _method_tag = "elsa-hybrid-kd" if getattr(FLAGS, 'do_kd_admm', False) and getattr(FLAGS, 'kd_use_cot_dataset', False) \
                            else "elsa-kd" if getattr(FLAGS, 'do_kd_admm', False) \
                            else "elsa-offpolicy-kd" if getattr(FLAGS, 'do_offpolicy_kd_admm', False) \
                            else "elsa-ntp-cot" if getattr(FLAGS, 'dataset', '') == 'mixed_cot' \
                            else "elsa-ntp"
                        _lr_tag = f"lr{_fmt_float(FLAGS.lr)}"
                        _lmda_tag = f"lmda{_fmt_float(FLAGS.admm_lmda)}"
                        _hub_repo = f"cosmos1030/{_method_tag}-{_sparsity_tag}-{_lr_tag}-{_lmda_tag}_{_now}"
                logging.info(f"Uploading model to HuggingFace Hub: {_hub_repo}")
                try:
                    # TRANSFORMERS_OFFLINE blocks push_to_hub — unset for upload only
                    import os as _os
                    for _env in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
                        _os.environ.pop(_env, None)
                    try:
                        import huggingface_hub.constants as _hf_const
                        _hf_const.HF_HUB_OFFLINE = False
                    except Exception:
                        pass
                    api = HfApi()
                    api.create_repo(repo_id=_hub_repo, exist_ok=True)
                    api.upload_folder(
                        folder_path=_hub_model_path,
                        repo_id=_hub_repo,
                        commit_message=f"ELSA pruned: sparsity={FLAGS.sparsity_ratio}, lr={FLAGS.lr}, lmda={FLAGS.admm_lmda}",
                    )
                    _hub_url = f"https://huggingface.co/{_hub_repo}"
                    logging.info(f"Uploaded to {_hub_url}")
                    if FLAGS.wandb:
                        wandb.run.summary["hub_model_id"] = _hub_repo
                        wandb.run.summary["hub_model_url"] = _hub_url
                except Exception as _e:
                    logging.warning(f"push_to_hub upload failed ({_e}); continuing without upload.")
            else:
                logging.warning("push_to_hub=True but no saved model path found. Skipping upload.")

    # Write wandb run ID to file for SLURM post-job rundb hook
    _run_id_file = os.environ.get("WANDB_RUN_ID_OUTPUT")
    if _run_id_file and FLAGS.wandb and wandb.run:
        with open(_run_id_file, "w") as _f:
            _f.write(wandb.run.id + "\n")
        logging.info(f"Wrote wandb run ID '{wandb.run.id}' to {_run_id_file}")

    # Write the actual saved-model path (timestamp assigned at runtime, not
    # knowable at job-submission time) so a SLURM-dependency-chained eval job
    # can pick it up without guessing/globbing for the right checkpoint dir.
    _model_path_file = os.environ.get("MODEL_PATH_OUTPUT")
    if _model_path_file and saved_pruned_model_path:
        with open(_model_path_file, "w") as _f:
            _f.write(saved_pruned_model_path + "\n")
        logging.info(f"Wrote saved model path '{saved_pruned_model_path}' to {_model_path_file}")


if __name__ == '__main__':
    flags.DEFINE_string('model', 'facebook/opt-125m', 'model to prune. model name (hf repo) or local path to model snapshot')
    flags.DEFINE_string('gmp_teacher_model', None, 'GMP KD/OPKD teacher model path (hf repo or local snapshot). Defaults to --model when unset -- MUST be overridden to the original dense model when --model points at an already-pruned checkpoint (e.g. ALPS sparse-SFT: fixed-mask fine-tuning from an ALPS checkpoint), otherwise the "teacher" is just a frozen copy of the pruned starting point, not a real dense reference.')
    flags.DEFINE_integer('seqlen', 2048, 'Sequence length for the model (shared by ADMM/ELSA and GMP NTP dataset construction).')
    flags.DEFINE_integer('seed', 0, 'Seed for sampling the calibration data.')
    flags.DEFINE_integer('nsamples', 128, 'Number of calibration samples.')
    flags.DEFINE_float('sparsity_ratio', 0.6, 'Sparsity level')
    flags.DEFINE_enum('sparsity_type', "unstructured", ["unstructured", "4:8", "2:4"], 'Type of sparsity.')
    flags.DEFINE_enum('dataset', 'c4', ["c4", "wikitext2", "math_trace", "code_trace", "math_prompt", "mixed_cot"], 'Calibration dataset.')
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
    # lr / steps / lr_scheduler / lr_warmup_steps / seqlen are shared between
    # ADMM (ELSA) and GMP — the two never run in the same job (do_gmp vs the
    # ADMM sparsity_ratio!=0 branch are mutually exclusive), so one flag each
    # is enough instead of admm_*/gmp_* duplicates.
    flags.DEFINE_integer('steps', 4096, 'Max training steps (ADMM: overrides admm_epochs if > 0; GMP: total training steps).')
    flags.DEFINE_integer('admm_batch_size', 2, 'Batch size for ADMM training, per device.')
    flags.DEFINE_integer('admm_gradient_accumulation_steps', 1, 'Gradient accumulation steps for ADMM.')
    flags.DEFINE_bool('admm_gradient_checkpointing', False, 'Use gradient checkpointing for ADMM training. Set False when using FSDP')
    flags.DEFINE_float('lr', 1e-4, 'Learning rate (ADMM base optimizer / GMP peak LR).')
    flags.DEFINE_string('lr_scheduler', 'cosine', 'LR scheduler type. HF get_scheduler name for ADMM; GMP checks for "cosine" vs "constant"/"constant_with_warmup". "constant" alone ignores warmup_steps entirely (HF quirk) — use constant_with_warmup to keep the step-based warmup. Default is cosine-with-warmup (decays to ~0 by the last step) -- constant_with_warmup (flat LR, no decay) was the default from 2026-08-04 to 2026-08-08 and was found to make TR-GMP\'s final forced mask-to-target jump (gmp_trainer.py "final mask at full sparsity") land on a noisier, undertrained checkpoint, hurting eval scores (e.g. job 702463 vs cosine-scheduled 703120).')
    flags.DEFINE_integer('lr_warmup_steps', 256, 'LR warmup steps (ADMM scheduler warmup; GMP overrides gmp_warmup_ratio when > 0).')
    flags.DEFINE_float('admm_weight_decay', 0.0, 'Weight decay for ADMM base optimizer.')
    flags.DEFINE_enum('admm_precision', 'bf16', ['fp32', 'fp16', 'bf16'], 'Precision for ADMM training (fp16/bf16 enables Trainer autocast).')
    flags.DEFINE_enum('admm_projection_mode', 'momentum', ['identity', 'momentum'], 'objective-aware projection for ADMM.')
    flags.DEFINE_bool('admm_projection_bias_correction', False, 'Whether to use bias correction in obejctive-aware ADMM projection.')

    # ADMM Specific Config
    flags.DEFINE_float('admm_lmda', 0.01, 'Lambda penalty parameter for ADMM (for constant schedule).')
    flags.DEFINE_float('admm_init_lmda', 0.0, 'Initial lambda value for ADMM scheduling.')
    flags.DEFINE_float('admm_final_lmda', 0.01, 'Final lambda value for ADMM scheduling.')
    flags.DEFINE_bool('admm_init_lambda_from_inv_resid', False, 'Initialize lambda from inverse of initial residual.')
    flags.DEFINE_enum('admm_lmda_schedule_mode', 'constant', ['constant', 'linear', 'exponential', 'cosine'], 'Mode for lambda schedule (e.g., linear, cosine).')
    flags.DEFINE_integer('admm_interval', 2, 'Interval for ADMM projection and dual updates.')
    flags.DEFINE_bool('admm_tr_z_proj', False, 'Use global trust-region z-projection (Stage 1).')
    flags.DEFINE_float('admm_tr_kl_threshold', 0.1, 'KL(Q_x||Q_z) threshold (delta) for TR z-projection.')
    flags.DEFINE_integer('admm_tr_max_iters', 8, 'Max halving iterations per admm_interval for TR z-projection.')
    flags.DEFINE_float('admm_tr_init_delta', 0.05, 'Initial sparsity step size for TR z-projection.')
    flags.DEFINE_float('admm_tr_delta_min', 1e-3, 'Minimum sparsity delta before giving up in TR z-projection.')
    flags.DEFINE_enum('admm_tr_kl_reduce', 'mean', ['mean', 'quantile'], 'KL reduce mode for TR z-projection.')
    flags.DEFINE_enum('admm_z_schedule_mode', 'trust_region', ['trust_region', 'cubic', 'cosine'], "z-projection schedule: 'trust_region' (KL-gated, adaptive), 'cubic' (fixed schedule from admm-pruning/Boza et al. Algorithm 1, no KL check), or 'cosine' (fixed cosine ramp to final sparsity over admm_cubic_steps, no KL check).")
    flags.DEFINE_integer('admm_cubic_steps', 2048, 'ks: training step at which the cubic schedule reaches final sparsity (independent of admm_interval, which controls z-projection call cadence).')
    flags.DEFINE_bool('admm_z_layerwise', False, 'Compute the TR-z/cubic threshold per-parameter-tensor (like plain ELSA default projection) instead of one global threshold pooled across all params.')
    flags.DEFINE_bool('admm_tr_gate_at_target', True, 'Once trust-region sparsity reaches the final target, still KL-gate further mask reselection (swap/freeze) instead of reselecting unconditionally every interval.')
    flags.DEFINE_enum('admm_base_optimizer', 'adam', ['adam','adamw','adam8bit','adam4bit','sgd'], 'Base optimizer for ADMM primal update.')
    flags.DEFINE_enum('admm_dual_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM dual variable (fp32 or bf16).')
    flags.DEFINE_float('admm_lasso_lmda', 0.0, 'L1 penalty on pruned-position weights; 0 = disabled.')
    flags.DEFINE_bool('admm_use_fsdp', False, 'Wrap ADMM model with FSDP for multi-GPU training (requires torchrun).')
    flags.DEFINE_enum('admm_split_dtype', 'fp32', ['fp32','bf16', 'float8_e4m3fn', 'float8_e5m2'], 'Dtype for ADMM split variable (fp32 or bf16).')
    flags.DEFINE_bool('admm_nonuniform_sparsity', False, 'Whether to use non-uniform sparsity based on sensitivity scores in ADMM.')
    flags.DEFINE_string('admm_nonuniform_sparsity_config_file', None, 'Path to non-uniform sparsity configuration file (JSON format).')
    # Dynamic Barrier: replaces the fixed/scheduled admm_lmda penalty coefficient
    # with a per-step closed-form lambda_k that guarantees the ADMM residual
    # ||w - z + u||^2 makes progress toward a shrinking target while otherwise
    # staying as close as possible to the raw KD gradient. See lib/optimizers.py
    # ADMMOptimizer._compute_barrier_lambda / _dual_update.
    flags.DEFINE_bool('admm_dynamic_barrier', False, 'Replace the fixed/scheduled ADMM lambda with a per-step Dynamic Barrier coefficient (overrides admm_lmda_schedule_mode).')
    flags.DEFINE_float('admm_barrier_alpha', 0.5, 'Dynamic Barrier: how aggressively phi_k demands residual progress toward the target each step (0=no progress required, 1=full progress in one step).')
    flags.DEFINE_float('admm_barrier_beta', 0.8, 'Dynamic Barrier: shrink factor for the per-interval residual target c_t = beta * g_start (g_start = 0.5*||r||^2 right after the interval z/u refresh).')
    flags.DEFINE_float('admm_barrier_lambda_max', 100.0, 'Dynamic Barrier: safety clamp on the computed lambda_k.')
    # GMP (BEST-style)
    flags.DEFINE_bool('do_gmp', False, 'Use BEST-style gradual magnitude pruning with Fisher importance.')
    flags.DEFINE_bool('gmp_fixed_mask', False, 'Fix mask from pre-pruned model weights (for sparse SFT). Skips Fisher-based mask updates.')
    flags.DEFINE_bool('gmp_random_cot_ntp', False, 'Use random seqlen-token windows from CoT (no prompt masking) instead of MixedTextDataset.')
    flags.DEFINE_bool('gmp_use_fsdp', False, 'Wrap GMP model with FSDP for multi-GPU training (requires torchrun / accelerate launch).')
    flags.DEFINE_bool('gmp_pgd', False, 'Enable PGD projection after each optimizer step: re-project mask using Fisher saliency (v_t*w^2), logging pgd/revivals and pgd/prunings.')
    flags.DEFINE_bool('gmp_ste', False, 'Straight-through estimator masking: forward computes weight*mask (sparsity respected), backward passes gradient straight through unmasked -- param.data is never hard-reset, so Adam accumulates masked weights\' true trajectory (used with --gmp_pgd so revival/importance scoring sees a multi-step signal instead of a one-step-from-zero snapshot).')
    flags.DEFINE_float('gmp_pgd_max_swap_frac', 0.0, 'Trust-region cap on PGD mask churn per step, as a fraction of total masked params (0 = unlimited, PGD projects onto the full top-k set every step regardless of how many positions that flips). When capped, only the most-confident revivals/prunings are applied each step; the rest are re-evaluated next step. Ignored if --gmp_pgd_kl_budget > 0.')
    flags.DEFINE_float('gmp_pgd_kl_budget', 0.0, 'Alternative to --gmp_pgd_max_swap_frac: gate PGD prunings per step by measured self-KL instead of a fixed count. Bisects the number of (lowest-importance) prune candidates accepted this step so that self-KL(pre-prune || post-prune), measured on a small cached calibration batch, stays within this budget. Revive count is always set equal to the accepted prune count (existing invariant), so revival volume is bounded for free by the same search -- revival itself is never separately KL-checked since a masked weight is architecturally zero until it grows via later gradient steps, so its instantaneous swap has no measurable effect. 0 = disabled (use --gmp_pgd_max_swap_frac or uncapped instead). Only implemented for unstructured PGD (sparsity_type=unstructured), not yet for N:M.')
    flags.DEFINE_bool('gmp_pgd_kl_share', False, 'Cheaper alternative to --gmp_pgd_kl_budget: instead of measuring a fresh self-KL every PGD step (extra forward passes each step), reuse TR-GMP\'s own already-measured KL from its once-per-mask_interval growth check. Derives this window\'s PGD swap_frac as (1 - kl_spent/gmp_tr_kl_threshold) / gmp_mask_interval -- full TR budget unused this window means PGD gets a whole window\'s worth of swap room spread over its steps; TR using its full budget means PGD gets none. Requires --gmp_tr_enabled=true (needs a real TR-GMP growth check to share from). Takes priority over --gmp_pgd_max_swap_frac when both are set; ignored if --gmp_pgd_kl_budget > 0.')
    flags.DEFINE_integer('gmp_pgd_kl_calib_size', 4, 'Number of sequences in the small calibration batch used by --gmp_pgd_kl_budget, refreshed every gmp_mask_interval steps (not every PGD step) to amortize data-loading cost.')
    flags.DEFINE_integer('gmp_pgd_kl_calib_seqlen', 512, 'Sequence length to truncate the --gmp_pgd_kl_budget calibration batch to -- deliberately much shorter than the real training seqlen, since this check only needs to be a cheap proxy, not a faithful reproduction of training-time behavior.')
    flags.DEFINE_integer('gmp_pgd_kl_bisect_iters', 6, 'Bisection iterations for --gmp_pgd_kl_budget\'s per-step search over how many prune candidates to accept (each iteration costs one extra small-batch forward pass, so kept short unlike the 48-iteration threshold searches elsewhere that need no forward pass at all).')
    flags.DEFINE_float('gmp_dpo_lambda', 0.0, 'Weight for DPO loss (0 = disabled).')
    flags.DEFINE_float('gmp_dpo_beta', 0.1, 'DPO beta (temperature).')
    flags.DEFINE_integer('gmp_dpo_n_pairs', 1024, 'Number of chosen pairs to pre-generate.')
    flags.DEFINE_integer('gmp_dpo_gen_batch', 8, 'Batch size for DPO continuation generation.')
    flags.DEFINE_integer('gmp_dpo_max_new_tokens', 512, 'Continuation length for DPO pairs.')
    flags.DEFINE_float('gmp_dpo_temperature', 0.7, 'Sampling temperature for DPO generation.')
    flags.DEFINE_integer('gmp_dpo_start_step', 0, 'Step from which to start applying DPO loss.')
    flags.DEFINE_bool('gmp_dpo_reference_free', False, 'Reference-free DPO: set ref logprobs to 0 (no ref model). Used as ablation control vs pruning-aware DPO.')
    flags.DEFINE_string('gmp_dpo_loss_type', 'sigmoid', 'DPO/IPO loss type: sigmoid, ipo, hinge, robust, ca_ipo, etc.')
    flags.DEFINE_float('gmp_ca_ipo_eps_credit', 1e-6, 'Uniform fallback credit weight for CA-IPO (prevents zero-weight tokens).')
    flags.DEFINE_string('gmp_dpo_cache_dir', '/home1/doyoonkim/projects/elsa/.cache/dpo_chosen', 'Directory to persist chosen cache across runs. Set empty string to disable.')
    flags.DEFINE_bool('gmp_dpo_use_vllm_chosen', False, 'Use vLLM offline engine for fast chosen cache generation (offloads dense_model to CPU during generation).')
    flags.DEFINE_bool('gmp_dpo_use_vllm_rejected', False, 'Use vLLM engine for fast rejected generation each mask interval (weight-synced from student model).')
    flags.DEFINE_float('gmp_dpo_vllm_gpu_mem', 0.35, 'vLLM gpu_memory_utilization for rejected generation engine.')
    flags.DEFINE_integer('gmp_batch_size', 1, 'Per-device batch size for GMP.')
    flags.DEFINE_integer('gmp_grad_accum', 8, 'Gradient accumulation steps for GMP.')
    flags.DEFINE_float('gmp_warmup_ratio', 0.05, 'Fraction of steps for LR warmup in GMP (used only when lr_warmup_steps=0).')
    flags.DEFINE_integer('gmp_dense_warmup_steps', 0, 'Steps to train fully dense before GMP pruning schedule starts (gates mask application, TR-GMP growth, cubic sparsity ramp, PGD, and DPO-queue refill alike).')
    flags.DEFINE_float('gmp_pruning_end_ratio', 1.0, 'Fraction of steps at which pruning completes; remaining steps do sparse training with fixed mask. Ignored when gmp_sparse_train_steps > 0.')
    flags.DEFINE_integer('gmp_sparse_train_steps', 512, 'Steps of fixed-mask sparse training at the end of the run (mask frozen at final sparsity). Pruning completes at steps - gmp_sparse_train_steps, so the cubic ramp fills the time between gmp_dense_warmup_steps and that point. 0 = derive from gmp_pruning_end_ratio instead.')
    flags.DEFINE_integer('gmp_post_target_steps', -1, "TR-GMP only: stop training this many steps after tr_reached first flips True (dynamic -- based on when the trust-region growth actually hits final_sparsity, not a precomputed step), instead of continuing for the full remaining `steps` budget with the mask frozen. -1 (default) = tie to gmp_mask_interval (stop after exactly one more mask-update cycle). 0 = explicitly disabled (old behavior: train all the way to `steps`).")
    flags.DEFINE_integer('gmp_mask_interval', 32, 'Steps between mask updates in GMP.')
    flags.DEFINE_float('gmp_fisher_beta', 0.999, 'EMA beta for Fisher diagonal accumulation.')
    flags.DEFINE_enum('gmp_saliency', 'fisher', ['fisher', 'magnitude', 'spa', 'sqrt_fisher', 'wanda'],
                      'Importance score for GMP pruning: fisher=F_hat*w^2 (Adam 2nd moment), magnitude=w^2, '
                      'spa=h*u^2 where u is the next unconstrained Adam iterate and h=sqrt(v_hat)+eps '
                      '(Sparse Projected Adam saliency -- exact solution to projecting u onto a sparse '
                      'support under the Adam metric), sqrt_fisher=sqrt(F_hat)*w^2 (spa\'s lr->0 limit -- '
                      'same cost as fisher, no momentum state needed), wanda=|w|*sqrt(scaler_row) '
                      '(Wanda-style weight*activation-norm from a one-batch forward-hook snapshot -- '
                      'NOT comparable across layers, use with --gmp_pruning_scope=layer only).')
    flags.DEFINE_enum('gmp_fisher_source', 'adam', ['adam', 'opd_empirical'],
                      'Fisher source for TR saliency: adam=exp_avg_sq (default), opd_empirical=grad^2 on OPD cal_batch.')
    flags.DEFINE_enum('gmp_pruning_scope', 'global', ['global', 'layer', 'block'],
                      'Pruning scope: global=single threshold across all layers, layer=per-layer threshold (each layer hits target sparsity exactly), block=per-block-of-layers threshold (each group of --gmp_blockwise_init_block/current block_size consecutive decoder layers gets its own independent threshold; only meaningful with --gmp_blockwise_squarehead=true, whose block_size this scope reuses -- non-layer params (embeddings, final norm) are pooled into their own group). Not yet implemented under FSDP.')
    flags.DEFINE_string('gmp_save_path', '/home1/doyoonkim/projects/elsa/models', 'Directory to save GMP pruned model.')
    flags.DEFINE_enum('gmp_base_optimizer', 'adamw', ['adamw', 'activation_metric_pgd'],
                      'Base optimizer for GMP training. activation_metric_pgd projects the gradient step onto the '
                      'active (non-pruned) coordinates using a running per-group activation-covariance metric '
                      '(see lib/activation_metric_projected_sgd.py) -- an online per-step analogue of PCG\'s '
                      'post-hoc reconstruction correction.')
    flags.DEFINE_float('gmp_pgd_lam', 1e-3, 'activation_metric_pgd: relative damping on the active-block covariance solve.')
    flags.DEFINE_integer('gmp_pgd_group_size', 4, 'activation_metric_pgd: input-dim column block size for the activation-covariance metric.')
    flags.DEFINE_float('gmp_pgd_trust_ratio', 5.0, 'activation_metric_pgd: cap the preconditioned step at this multiple of the plain-SGD step.')
    flags.DEFINE_float('gmp_pgd_momentum', 0.0, 'activation_metric_pgd: classical momentum on the gradient before projection (0 = none).')
    flags.DEFINE_integer('gmp_max_prompt_len', 512, 'Max prompt length for GMP NTP dataset.')
    flags.DEFINE_float('gmp_ntp_lambda', 1.0, 'NTP loss weight for GMP (default 1.0).')
    flags.DEFINE_float('gmp_kd_lambda', 0.0, 'KD loss weight for GMP (0 = NTP only).')
    flags.DEFINE_float('gmp_kd_temperature', 2.0, 'Temperature for GMP token-level KD.')
    flags.DEFINE_integer('gmp_kd_topk', 0, 'Top-K for KD KL divergence (0 = full vocab).')
    flags.DEFINE_integer('gmp_kl_chunk_size', 0, 'Chunk the sequence dimension into pieces of this many tokens when computing full-vocab KL loss (0 = disabled, compute the whole sequence at once). Full-vocab log-softmax tensors at seqlen=8192 are ~5GB each in fp32 regardless of GPU count (not FSDP-sharded) -- fits an 80GB card but OOMs a 40GB one; chunking trades a Python loop for a bounded peak at the same total FLOPs.')
    flags.DEFINE_bool('gmp_kd_only', False, 'Use KD loss only (no NTP loss).')
    flags.DEFINE_float('gmp_hidden_lambda', 0.0, 'Weight for final hidden matching loss vs dense teacher.')
    flags.DEFINE_boolean('gmp_blockwise_squarehead', False,
                          'Anchor-based SquareHead-style per-layer distillation loss (Kurtic et al., '
                          '"Sparse Fine-tuning for Inference Acceleration of Large Language Models", '
                          'arxiv.org/abs/2310.06927, code at github.com/IST-DASLab/SparseFinetuning) with '
                          'ADAPTIVE anchor spacing (block size): starts at gmp_blockwise_init_block '
                          '(1 = anchor every layer, matching the paper exactly) and widens by '
                          'gmp_blockwise_widen_factor (fewer anchors, more inter-layer compensation '
                          'freedom for the layers between surviving anchors) whenever TR-GMP\'s own '
                          'trust-region growth stalls completely (no delta accepted down to '
                          'gmp_tr_delta_min). Requires --gmp_tr_enabled=true and a teacher model. The '
                          'accept/reject decision for growth itself is UNCHANGED -- still output-level '
                          'KL via _tr_mask_update on a fresh calibration batch every call, never the '
                          'blockwise loss value itself (that would be circular: the model was just '
                          'directly optimized to minimize it, so a low value there says nothing about '
                          'generalization, only about fitting the calibration batch it was computed on).')
    flags.DEFINE_float('gmp_blockwise_hardness', 1.0, 'Weight of the blockwise SquareHead loss term.')
    flags.DEFINE_integer('gmp_blockwise_init_block', 1,
                          'Initial anchor spacing for --gmp_blockwise_squarehead (1 = every layer).')
    flags.DEFINE_integer('gmp_blockwise_widen_factor', 2,
                          'Multiplicative factor for widening block size (fewer anchors) on a TR-GMP stall.')
    flags.DEFINE_bool('gmp_blockwise_delay_global_signal', False,
                       'Hold NTP/KD/OPKD lambdas at 0 (SquareHead loss alone drives training AND Fisher '
                       'importance) until block_size widens all the way to every decoder layer (no more '
                       'widening possible), then switch NTP/KD/OPKD back on at their configured lambdas. '
                       'Tests whether local per-layer distillation is sufficient on its own to keep growth '
                       'safe -- with NTP/KD/OPKD always on, TR-GMP\'s KL check is trivially satisfied and '
                       'widening was observed to basically never fire. Requires --gmp_blockwise_squarehead=true.')
    flags.DEFINE_bool('gmp_hidden_only', False, 'Use final hidden matching loss only (no NTP, no logit KD).')
    flags.DEFINE_string('gmp_hidden_mode', 'cosine', 'Loss for hidden matching: cosine (default), nmse, or mse.')
    flags.DEFINE_string('gmp_hidden_mask', 'cot', 'Mask for hidden matching: cot (labels!=-100) or all (attention_mask, prompt+CoT).')
    flags.DEFINE_string('gmp_hidden_layers', 'final', 'Layer scope: final (last layer only) or anneal_all_to_final (coarse-to-fine).')
    flags.DEFINE_float('gmp_onpolicy_kd_lambda', 0.0, 'Weight for on-policy KD loss in GMP (0 = disabled).')
    flags.DEFINE_integer('gmp_onpolicy_kd_interval', 1, 'Optimizer steps between on-policy KD generations.')
    flags.DEFINE_integer('gmp_onpolicy_max_new_tokens', 256, 'Max new tokens for on-policy student generation.')
    flags.DEFINE_integer('gmp_onpolicy_kd_topk', 0, 'Top-K for on-policy KL divergence (0 = full vocab).')
    flags.DEFINE_float('gmp_onpolicy_temperature', 0.6, 'Sampling temperature for on-policy generation.')
    flags.DEFINE_integer('gmp_onpolicy_grad_accum', 1, 'Number of on-policy generate+KL micro-steps to accumulate per interval.')
    flags.DEFINE_float('gmp_onpolicy_grad_clip', 1.0, 'Gradient clip norm applied after each on-policy rollout backward.')
    flags.DEFINE_boolean('gmp_onpolicy_reverse_kl', False, 'Use reverse KL D(S||T) for on-policy KD instead of forward KL.')
    flags.DEFINE_boolean('gmp_opkd_reuse_ipo_rollouts', False, 'If True, OPKD reuses IPO rejected rollouts (from RejectedQueue.rollout_pool) instead of generating new ones.')
    flags.DEFINE_boolean('gmp_opkd_prev_mask_teacher', False, 'Use pre-mask-update model snapshot as OPKD teacher instead of the dense teacher.')
    flags.DEFINE_float('gmp_prevmask_opkd_lambda', 0.0, 'Weight for prev-mask-teacher OPKD loss added on top of dense teacher OPKD (0=disabled).')
    flags.DEFINE_float('gmp_opkd_vllm_gpu_mem', 0.35, 'GPU memory utilization for the OPKD vLLM engine.')
    flags.DEFINE_integer('gmp_opkd_vllm_gpu_index', -1, 'CUDA device index for the OPKD vLLM subprocess. -1 (default) = dedicated GPU at index=world_size. >=0 shares that training rank\'s physical GPU instead (no extra GPU needed).')
    flags.DEFINE_integer('gmp_opkd_vllm_tp_size', 1, 'Tensor-parallel size for the OPKD vLLM engine. >1 spreads vLLM weights+KV cache evenly across that many training GPUs (indices 0..tp_size-1) instead of piling onto one -- takes priority over gmp_opkd_vllm_gpu_index.')
    flags.DEFINE_boolean('gmp_opkd_vllm_enforce_eager', False, 'If True, disable vLLM CUDA graph capture (enforce_eager=True) to save peak memory.')
    flags.DEFINE_boolean('gmp_gradient_checkpointing', False, 'If True, enable gradient checkpointing to reduce activation memory (trades compute for memory).')
    # TR-GMP: trust-region gradual mask selection
    flags.DEFINE_boolean('gmp_tr_enabled', False, 'Use trust-region KL-constrained mask updates instead of cubic sparsity schedule.')
    flags.DEFINE_float('gmp_tr_kl_threshold', 0.01, 'TR-GMP: max KL(old||cand) per token to accept a mask update.')
    flags.DEFINE_float('gmp_tr_delta_init', 0.05, 'TR-GMP: initial sparsity step size (fraction of total params).')
    flags.DEFINE_float('gmp_tr_delta_min', 0.005, 'TR-GMP: minimum sparsity step size; line search stops halving below this.')
    flags.DEFINE_string('gmp_tr_kl_reduce', 'mean', "TR-GMP: KL aggregation over tokens — 'mean' or 'quantile'.")
    flags.DEFINE_float('gmp_tr_kl_quantile', 0.95, 'TR-GMP: quantile level when gmp_tr_kl_reduce=quantile.')
    flags.DEFINE_bool('gmp_cubic_log_kl', False, 'Cubic-schedule (--gmp_tr_enabled=false) diagnostic: measure KL(old||candidate) at every mask-update boundary using the same _compute_tr_kl the TR path uses to accept/reject growth -- purely logged (cubic/kl_before_after, cubic/sparsity), never gates the update. For a fair cubic-vs-trust-region comparison: how far outside a TR budget does the cubic schedule\'s forced growth actually land.')
    flags.DEFINE_enum('gmp_growth_schedule', 'cubic', ['cubic', 'cosine'], 'Fixed sparsity ramp shape used when --gmp_tr_enabled=false (ignored otherwise): \'cubic\' (fast start, slow finish) or \'cosine\' (slow start/end, steepest in the middle). Reaches --sparsity_ratio at step (steps - gmp_sparse_train_steps).')
    flags.DEFINE_string('gmp_milestone_sparsities', '', 'Comma-separated sparsity checkpoints (e.g. "0.5,0.6,0.7"). Saves model at each level; post-hoc eval runs after training.')
    flags.DEFINE_boolean('gmp_pcg_correct', False, 'TR-GMP: after every mask update, apply an ALPS-style PCG backsolve (mask fixed, no ADMM search) using the dense teacher as reconstruction target -- lib/gmp_trainer.py _pcg_correct_masked_weights. Non-FSDP only.')
    flags.DEFINE_integer('gmp_pcg_maxiter', 5, 'Max conjugate-gradient iterations per layer for gmp_pcg_correct (kept small since this runs every mask update).')
    flags.DEFINE_float('gmp_pcg_damp', 0.01, 'Ridge damping coefficient (relative to mean diagonal of X^TX) for gmp_pcg_correct.')
    flags.DEFINE_boolean('gmp_pcg_sequential', False, 'gmp_pcg_correct: use the ALPS-style sequential per-layer variant (re-forward after each layer so later layers see the actual post-correction input) instead of the single-snapshot-forward default. Costs ~num_layers extra forward passes per mask update.')
    flags.DEFINE_boolean('gmp_measure_grad_conflict', False, 'If True, measure cosine similarity between OPKD and IPO gradients at every OPKD step and log to wandb.')
    flags.DEFINE_boolean('gmp_filter_grad_conflict', False, 'If True, filter OPKD gradient when cos_sim(g_opkd, g_ntp+g_ipo) < 0.')
    flags.DEFINE_boolean('gmp_opkd_project_onto_combined', False, 'If True, project g_OPKD onto (g_NTP+g_DPO) direction: g̃_OPKD = scalar*(g_NTP+g_DPO).')
    flags.DEFINE_boolean('gmp_opkd_filter_combined', False, 'If True, drop g_OPKD entirely when cos_sim(g_OPKD, g_NTP+g_DPO) < 0.')
    flags.DEFINE_boolean('gmp_teacher_gen_kd', False, 'Pre-generate teacher rollouts once (total_steps*gbs entries) and use them for KD sequentially each micro-step.')
    flags.DEFINE_boolean('gmp_onpolicy_pg', False, 'Add MiniLLM-style long-term policy gradient loss to on-policy KD.')
    flags.DEFINE_float('gmp_onpolicy_pg_lambda', 1.0, 'Weight for the long-term PG loss.')
    flags.DEFINE_float('gmp_onpolicy_mixed_alpha', 0.0, 'Teacher-mixed sampling alpha: sample from alpha*p_T+(1-alpha)*q_S. 0=pure student.')
    flags.DEFINE_float('gmp_onpolicy_pg_cliprange', 0.2, 'PPO clip range epsilon for on-policy PG loss.')
    flags.DEFINE_float('gmp_onpolicy_pg_gamma', 0.99, 'Discount factor gamma for on-policy PG cumulative reward.')
    flags.DEFINE_integer('gmp_rollout_buffer_size', 0, 'Rollout buffer size for PPO reuse (MiniLLM-style). 0=disabled (inline backward). When >0, collect rollouts and run ppo_epochs updates per buffer fill.')
    flags.DEFINE_integer('gmp_ppo_epochs', 2, 'Number of PPO optimization epochs per rollout buffer fill.')
    flags.DEFINE_float('gmp_pg_reward_clip', 10.0, 'Clip rewards to [-clip, +clip] before discounted cumsum. 0=disabled.')
    flags.DEFINE_float('gmp_pg_reward_scale', 0.0, 'Divide rewards by this value before clipping (MiniLLM reward_scaling). 0=disabled.')
    flags.DEFINE_string('gmp_prompt_path', None, 'Path to math prompts JSONL for on-policy GMP KD (defaults to data_path).')
    flags.DEFINE_float('gmp_anchor_kd_lambda', 0.0, 'Weight for anchored KD loss (CoT prefix + student continuation).')
    flags.DEFINE_integer('gmp_anchor_kd_interval', 32, 'Optimizer steps between anchored KD generations.')
    flags.DEFINE_integer('gmp_anchor_prefix_len', 1536, 'CoT prefix length (tokens) for anchored KD.')
    flags.DEFINE_integer('gmp_anchor_max_new_tokens', 512, 'Max new tokens for anchored student generation.')
    flags.DEFINE_bool('gmp_teacher_seqkd', False, 'SeqKD: teacher generates sequences, student does NTP on them. No CoT dataset NTP, no KL divergence.')

    # GRPO-OPKD: GRPO-style on-policy KD
    flags.DEFINE_bool('do_grpo_opkd', False, 'Use GRPO-style on-policy KD (lib/grpo_opkd.py).')
    flags.DEFINE_integer('gmp_grpo_num_rollouts', 4, 'Number of rollouts per prompt G for GRPO-OPKD.')
    flags.DEFINE_integer('gmp_grpo_interval', 8, 'Optimizer steps between GRPO-OPKD updates.')
    flags.DEFINE_float('gmp_grpo_lambda', 1.0, 'Weight for GRPO-OPKD loss.')
    flags.DEFINE_float('gmp_grpo_eps_clip', 0.2, 'PPO clip range epsilon for GRPO-OPKD.')
    flags.DEFINE_bool('gmp_correctness_reward', False,
                      'Use correctness+format reward instead of log-ratio reward in GRPO-OPKD.')
    flags.DEFINE_float('gmp_format_reward', 0.1,
                       'Partial reward for correct format (</think>+\\boxed{}) but wrong answer.')

    # Chunk-GRPO-OPKD: chunk-wise teacher-verified on-policy KD
    flags.DEFINE_bool('do_chunk_grpo_opkd', False, 'Use chunk-wise GRPO-OPKD (lib/grpo_opkd.py).')
    flags.DEFINE_integer('gmp_chunk_size', 32, 'Tokens per chunk K for chunk-GRPO-OPKD.')
    flags.DEFINE_float('gmp_chunk_adv_clip', 2.0, 'Advantage clamp value for chunk-GRPO-OPKD.')
    flags.DEFINE_bool('gmp_chunk_reward_logratio', True, 'Reward = log p_T - log q_old (True) or log p_T only (False).')
    flags.DEFINE_float('gmp_chunk_kd_lambda', 0.0, 'Weight for on-policy reverse KL on full generated sequence after chunk loop.')
    flags.DEFINE_float('gmp_l1_lambda', 0.0, 'L1 regularization weight. 0=disabled.')
    flags.DEFINE_bool('gmp_l1_structured', True, 'True=bottom-2 per group L1 (2:4 structured), False=use gmp_l1_mode.')
    flags.DEFINE_enum('gmp_l1_mode', 'plain', ['plain', 'inv_fisher_sqrt'],
                      'L1 mode when gmp_l1_structured=False. plain=mean|w|, inv_fisher_sqrt=|w|/sqrt(clamp(f/mean_f)).')
    flags.DEFINE_bool('gmp_l1_open_groups_only', False,
                      'When True (and sparsity_type is N:M), restrict gmp_l1_mode L1 to weights in '
                      '2:4 groups that have not yet reached their prune_n cap, instead of all alive '
                      'weights layer-wide. Concentrates L1 pressure on the shrinking pool of still-'
                      'prunable weights as a layer approaches its target sparsity. No effect when '
                      'gmp_l1_structured=True.')
    flags.DEFINE_float('gmp_l1_fisher_clip_min', 0.1, 'Min clamp for normalized Fisher in inv_fisher_sqrt L1.')
    flags.DEFINE_float('gmp_l1_fisher_clip_max', 10.0, 'Max clamp for normalized Fisher in inv_fisher_sqrt L1.')

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
    flags.DEFINE_bool('kd_use_cot_dataset', False, 'Use MixedTextDataset (provides CoT NTP labels + prompt for KD).')
    flags.DEFINE_bool('kd_offpolicy_ntp', False, 'Hybrid NTP + dataset-based KD: KL(student||teacher) on CoT answer tokens, no generation.')
    flags.DEFINE_bool('kd_triple_loss', False, 'Triple loss: NTP + dataset KD + on-policy KD (each weighted by ntp/kd/opkd lambda).')
    flags.DEFINE_float('kd_opkd_lambda', 0.0, 'Weight for on-policy KD loss in triple loss mode.')
    flags.DEFINE_bool('admm_tr_use_opkd_rollout', False, 'Use OPKD student rollout as TR-z calibration batch instead of CoT text.')
    flags.DEFINE_bool('cot_append_eos', False, 'Append EOS token to each sample in MixedTextDataset.')
    flags.DEFINE_bool('kd_use_random_cot_ntp', False, 'Use random 2048-token CoT windows for NTP; use separate MixedPromptDataset for KD prompts.')
    flags.DEFINE_integer('kd_step_interval', 1, 'Apply KD loss every N optimizer steps (1=every step). Reduces teacher forward cost.')
    flags.DEFINE_string('kd_ntp_dataset', 'mixed_cot', 'Dataset for NTP in random CoT mode: mixed_cot or c4.')
    flags.DEFINE_integer('kd_buffer_size', 0, 'Rollout buffer size: generate this many prompts in one vLLM batch (0 = disabled).')
    flags.DEFINE_integer('kd_buffer_refresh_interval', 32, 'Refresh rollout buffer every N steps (default: align with admm_interval).')

    # OPD (On-Policy Distillation inside ADMM with z-masked rollouts)
    flags.DEFINE_bool('opd_enabled', False, 'Enable OPD: generate z-masked rollouts for backward KL inside ADMM training.')
    flags.DEFINE_float('opd_lambda', 0.0, 'OPD loss weight. In the NTP+KD+OPD hybrid path, NTP/KD/OPD each get opd_lambda/3; in the KD-only offpolicy path (do_offpolicy_kd_admm), it is OPD\'s own weight applied directly (paired with kd_lambda for KD\'s weight, no NTP term).')
    flags.DEFINE_integer('opd_vllm_max_tokens', 256, 'Max tokens per OPD rollout generation.')
    flags.DEFINE_string('opd_prompt_path', '', 'Prompt source for OPD on-policy rollouts. Defaults to kd_data_path if empty -- set this to a disjoint file so OPD never rolls out on a prompt the KD loss is also training on.')
    flags.DEFINE_float('opd_vllm_gpu_mem', 0.25, 'GPU memory fraction for OPD vLLM engine (single-GPU mode).')

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
    flags.DEFINE_bool('eval_full_bench', True, 'Whether to run the lighteval benchmark suite after training.')
    flags.DEFINE_enum('eval_profile', 'quick', ['quick', 'official'],
                       "'quick' (default, 8192 budget, 5 tasks, no AIME24/25 -- fast enough for every sweep job) "
                       "or 'official' (official Qwen3 budgets incl. AIME24/25, ~2-4x slower per benchmark -- only "
                       "use for a specific run you already know is going in a results table, not a routine sweep).")
    flags.DEFINE_float('gmp_offline_ipo_lambda', 0.0, 'Weight for offline IPO loss (0 = disabled).')
    flags.DEFINE_float('gmp_offline_ipo_beta', 0.1, 'IPO beta for offline preference pairs.')
    flags.DEFINE_string('gmp_offline_ipo_datasets', 'HuggingFaceH4/ultrafeedback_binarized', 'Comma-separated HF preference dataset names.')
    flags.DEFINE_string('gmp_offline_ipo_splits', 'train_prefs', 'Comma-separated splits for each offline IPO dataset.')
    flags.DEFINE_string('gmp_offline_ipo_per_max', '8000', 'Comma-separated max samples per dataset (or single int applied to all).')
    flags.DEFINE_integer('gmp_offline_ipo_max_length', 2048, 'Max sequence length for offline IPO pairs.')
    flags.DEFINE_integer('gmp_offline_ipo_max_prompt_length', 1024, 'Max prompt length for offline IPO pairs.')
    flags.DEFINE_string('math500_model_path', None, 'Path to saved pruned model for lighteval eval. If None, saves model to temp dir.')
    flags.DEFINE_integer('math500_max_new_tokens', 4096, 'max_new_tokens for MATH-500 generation.')
    flags.DEFINE_integer('math500_max_samples', 0, 'Max samples for MATH-500 eval (0 = all 500).')
    flags.DEFINE_bool('wandb', False, 'Whether to use wandb for logging.')
    flags.DEFINE_string('wandb_project', None, 'wandb project name.')
    flags.DEFINE_string('run_name_suffix', '', 'Appended verbatim to the auto-built wandb run_name -- for disambiguating runs that share identical FLAGS but differ in code (e.g. a bugfix between two otherwise-identical launches).')
    flags.DEFINE_bool('push_to_hub', False, 'Whether to push the pruned model to HuggingFace Hub after eval.')
    flags.DEFINE_string('hub_model_id', None, 'HuggingFace Hub repo id (e.g. username/model-name) to push pruned model.')

    # SAFE pruning (layer-by-layer ADMM + SAM)
    flags.DEFINE_bool('do_safe', False, 'Use SAFE layer-by-layer pruning (ADMM + SAM).')
    flags.DEFINE_float('safe_lr', 2e-4, 'Learning rate for SAFE optimizer.')
    flags.DEFINE_float('safe_lmda', 1e-3, 'Lambda penalty for SAFE ADMM.')
    flags.DEFINE_float('safe_rho', 0.05, 'SAM perturbation size (rho) for SAFE.')
    flags.DEFINE_integer('safe_epochs', 30, 'Number of epochs per layer for SAFE.')
    flags.DEFINE_integer('safe_warmup_epochs', 2, 'Warmup epochs per layer for SAFE.')
    flags.DEFINE_integer('safe_interval', 32, 'ADMM dual update interval for SAFE.')
    flags.DEFINE_integer('safe_batch_size', 4, 'Batch size for SAFE calibration dataloader.')
    flags.DEFINE_integer('safe_accumulation_steps', 1, 'Gradient accumulation steps for SAFE.')

    app.run(main)
