"""Generate all sweep yamls for Qwen3-1.7B experiments."""
import os, yaml

OUT_DIR = "sweep_configs/qwen3_1.7b"
MODEL_PATH = "/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PROJECT = "reasoning_qwen3_1.7b"
ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"
DATA_PATH = "/home1/doyoonkim/projects/elsa/data/ot3_fineweb_20k.jsonl"
SAVE_PATH = "/home1/doyoonkim/projects/elsa/models"

SPARSITIES = [0.50, 0.60, 0.70]

COMMON = {
    "seed": 42,
    "model": MODEL_PATH,
    "dataset": "mixed_cot",
    "data_path": DATA_PATH,
    "do_gmp": True,
    "gmp_batch_size": 1,
    "gmp_grad_accum": 8,
    "lr": 1e-4,
    "gmp_warmup_ratio": 0.05,
    "gmp_mask_interval": 32,
    "gmp_fisher_beta": 0.999,
    "gmp_kd_lambda": 1.0,
    "seqlen": 2048,
    "gmp_max_prompt_len": 512,
    "gmp_dpo_lambda": 0.0,
    "gmp_save_path": SAVE_PATH,
    "save_model": True,
    "eval_math500": False,
    "eval_full_bench": True,
    "eval_zero_shot": True,
    "wandb": True,
    "wandb_project": PROJECT,
    "push_to_hub": True,
}

def make_yaml(name, sparsity, extra_params):
    sp_tag = f"s{int(sparsity*100)}pct"
    fname = f"{name}_{sp_tag}.yaml"
    params = {"seed": {"value": COMMON["seed"]}}
    params["sparsity_ratio"] = {"value": sparsity}
    for k, v in COMMON.items():
        if k in ("seed",):
            continue
        params[k] = {"value": v}
    for k, v in extra_params.items():
        if isinstance(v, list):
            params[k] = {"values": v}
        else:
            params[k] = {"value": v}

    doc = {
        "program": "main.py",
        "method": "grid",
        "project": PROJECT,
        "name": f"{name}_{sp_tag}",
        "entity": ENTITY,
        "command": [
            "/home1/doyoonkim/miniconda3/envs/rac/bin/python",
            "${program}",
            "${args}",
        ],
        "metric": {"name": "math500_pass@1", "goal": "maximize"},
        "parameters": params,
    }
    path = os.path.join(OUT_DIR, fname)
    with open(path, "w") as f:
        yaml.dump(doc, f, default_flow_style=False, sort_keys=False)
    return path, fname

TR_KL_4K  = [0.005, 0.01, 0.02]   # TR-GMP / TR-GMP+OPKD 4k
TR_KL_8K  = [0.005, 0.01]          # PrevMask / Dual 8k (kl=0.0025은 32k 별도)
TR_KL_32K = [0.0025]               # PrevMask / Dual 32k

configs = []  # (yaml_path, n_runs)

for sp in SPARSITIES:
    sp_tag = f"s{int(sp*100)}pct"

    # 1. GMP NTP+KD (no TR, no OPKD)
    p, _ = make_yaml("gmp_ntp_kd_qwen3_1.7b", sp, {
        "steps": 8192,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 0.0,
        "gmp_opkd_use_vllm": False,
        "gmp_tr_enabled": False,
    })
    configs.append((p, 1))

    # 2. GMP NTP+KD+OPKD (no TR)
    p, _ = make_yaml("gmp_ntp_kd_opkd_qwen3_1.7b", sp, {
        "steps": 8192,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": False,
    })
    configs.append((p, 1))

    # 3. TR-GMP NTP+KD (no OPKD)
    p, _ = make_yaml("gmp_tr_ntp_kd_qwen3_1.7b", sp, {
        "steps": 4096,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 0.0,
        "gmp_opkd_use_vllm": False,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": TR_KL_4K,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, len(TR_KL_4K)))

    # 4. TR-GMP+OPKD 4k (dense teacher, kl_reduce=mean)
    p, _ = make_yaml("gmp_tr_opkd_ntp_kd_qwen3_1.7b", sp, {
        "steps": 4096,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_opkd_prev_mask_teacher": False,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": TR_KL_4K,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, len(TR_KL_4K)))

    # 5. TR-GMP+OPKD PrevMask 8k (kl=0.005, 0.01)
    p, _ = make_yaml("gmp_tr_opkd_prevmask_ntp_kd_qwen3_1.7b", sp, {
        "steps": 8192,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_opkd_prev_mask_teacher": True,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": TR_KL_8K,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, len(TR_KL_8K)))

    # 5b. TR-GMP+OPKD PrevMask 32k (kl=0.0025)
    p, _ = make_yaml("gmp_tr_opkd_prevmask_ntp_kd_32k_qwen3_1.7b", sp, {
        "steps": 32768,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_opkd_prev_mask_teacher": True,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": 0.0025,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, 1))

    # 6. TR-GMP+OPKD Dual 8k (kl=0.005, 0.01)
    p, _ = make_yaml("gmp_tr_opkd_dual_ntp_kd_qwen3_1.7b", sp, {
        "steps": 8192,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_opkd_prev_mask_teacher": False,
        "gmp_prevmask_opkd_lambda": 1.0,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": TR_KL_8K,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, len(TR_KL_8K)))

    # 6b. TR-GMP+OPKD Dual 32k (kl=0.0025)
    p, _ = make_yaml("gmp_tr_opkd_dual_ntp_kd_32k_qwen3_1.7b", sp, {
        "steps": 32768,
        "gmp_prompt_path": DATA_PATH,
        "gmp_onpolicy_kd_lambda": 1.0,
        "gmp_opkd_use_vllm": True,
        "gmp_opkd_vllm_gpu_mem": 0.35,
        "gmp_opkd_prev_mask_teacher": False,
        "gmp_prevmask_opkd_lambda": 1.0,
        "gmp_onpolicy_max_new_tokens": 256,
        "gmp_tr_enabled": True,
        "gmp_tr_delta_init": 0.05,
        "gmp_tr_delta_min": 0.005,
        "gmp_tr_kl_threshold": 0.0025,
        "gmp_tr_kl_reduce": "mean",
    })
    configs.append((p, 1))

total_runs = sum(n for _, n in configs)
print(f"Generated {len(configs)} sweep yamls → {total_runs} total runs")
for p, n in configs:
    print(f"  {os.path.basename(p)}  ({n} run{'s' if n>1 else ''})")
