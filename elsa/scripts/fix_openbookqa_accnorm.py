"""
OpenBook QA acc_norm 재평가 후 wandb run summary 덮어쓰기.
GMP계열 run들이 acc,none을 로깅했으나 acc_norm,none이 올바른 메트릭.
SparseGPT / Dense는 이미 올바른 값이므로 스킵.
"""
import torch, gc, json, os
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

ENTITY  = "dyk6208-gwangju-institute-of-science-and-technology"
PROJECT = "reasoning_pruning_v2"

# (wandb_run_id, hf_model_id)
# SparseGPT(tvatqxx4,6qftdjlo,mgxa9eus)와 Dense(4t6quvd1)는 이미 acc_norm이므로 제외
RUNS = [
    # ── S50 ──────────────────────────────────────────────────────────
    ("ipubu9cq", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260705_130752"),   # GMP NTP+KD
    ("2je0fyab", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260706_020934"),   # GMP NTP+KD+OPKD
    ("zktj96b7", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_013118"),   # TR-GMP kl=0.005
    ("ph42mgbi", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_011305"),   # TR-GMP kl=0.01
    ("q8us6c6n", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_010755"),   # TR-GMP kl=0.02
    ("ul11t38q", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_121943"),   # TR-GMP+OPKD 4k kl=0.005
    ("w0j7gvlz", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_114525"),   # TR-GMP+OPKD 4k kl=0.01
    ("6337e12s", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260707_114011"),   # TR-GMP+OPKD 4k kl=0.02
    ("mjfwh085", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260708_123251"),   # cfg8k dense kl=0.005
    ("v85j5bdl", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260708_120835"),   # cfg8k dense kl=0.01
    ("15btrmnv", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260708_120030"),   # cfg8k dense kl=0.02
    ("u1i8zgs4", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_071332"),   # prevmask kl=0.005
    ("q5kzdkf6", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_064323"),   # prevmask kl=0.01
    # ── S60 ──────────────────────────────────────────────────────────
    ("3w3ubwja", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260705_131614"),   # GMP NTP+KD
    ("ebf8oski", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260706_022219"),   # GMP NTP+KD+OPKD
    ("3i7n0udd", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_020505"),   # TR-GMP kl=0.005
    ("yxho0vmq", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_012832"),   # TR-GMP kl=0.01
    ("1krya5s0", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_031928"),   # TR-GMP kl=0.02
    ("qq3dhm8n", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_124907"),   # TR-GMP+OPKD kl=0.005
    ("rdzjtvpv", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_120906"),   # TR-GMP+OPKD kl=0.01
    ("ch788zxg", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260707_115022"),   # TR-GMP+OPKD kl=0.02
    # ── S70 ──────────────────────────────────────────────────────────
    ("ov87u2y0", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260705_132250"),   # GMP NTP+KD
    ("xn5x8fdy", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260706_023531"),   # GMP NTP+KD+OPKD
    # ze6qflhi TR-GMP kl=0.005 s70: HF 모델 없음, 스킵
    ("hbwmkmyr", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260707_020027"),   # TR-GMP kl=0.01
    ("b7btobbb", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260707_013419"),   # TR-GMP kl=0.02
    ("bblrnf9d", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260707_145811"),   # TR-GMP+OPKD kl=0.005
    ("v4bbime0", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260707_130931"),   # TR-GMP+OPKD kl=0.01
    ("guma0zsg", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260707_122404"),   # TR-GMP+OPKD kl=0.02
    # ── S50 Dual / PrevMask (신규) ────────────────────────────────────
    ("qd1117ir", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_171528"),   # Dual kl=0.0025
    ("skv34ivb", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_144853"),   # Dual kl=0.005
    ("vavkcbhd", "cosmos1030/gmp-kd1e0-s50pct-lr1e-4_20260710_184120"),   # Dual kl=0.01
    # ── S60 PrevMask / Dual (신규) ────────────────────────────────────
    ("2tvw63dd", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_211955"),   # PrevMask kl=0.0025
    ("lr6irpqw", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_165658"),   # PrevMask kl=0.005
    ("qmskp5i3", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_170933"),   # PrevMask kl=0.01 (collapsed)
    ("72gcoqxh", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_210736"),   # Dual kl=0.0025
    ("3db52ubg", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_165553"),   # Dual kl=0.005
    ("k2m9qpwz", "cosmos1030/gmp-kd1e0-s60pct-lr1e-4_20260710_203741"),   # Dual kl=0.01
    # ── S70 PrevMask / Dual (신규) ────────────────────────────────────
    ("qpuf8u1s", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_214929"),   # PrevMask kl=0.0025
    ("ihrr6nt5", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_174108"),   # PrevMask kl=0.005
    ("10laq7t2", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_171018"),   # PrevMask kl=0.01
    ("4zssrw84", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_220516"),   # Dual kl=0.0025
    ("1wak5534", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_174446"),   # Dual kl=0.005
    ("vni8ewy4", "cosmos1030/gmp-kd1e0-s70pct-lr1e-4_20260710_170350"),   # Dual kl=0.01
]

OUT_PATH = "/home1/doyoonkim/projects/elsa/eval_outputs/openbookqa_accnorm_fix.json"
results_map = {}

# 이미 처리한 것 있으면 로드
if os.path.exists(OUT_PATH):
    with open(OUT_PATH) as f:
        results_map = json.load(f)
    print(f"Loaded {len(results_map)} cached results from {OUT_PATH}")

api = wandb.Api()

for run_id, hf_id in RUNS:
    if run_id in results_map:
        print(f"[skip] {run_id} already done (acc_norm={results_map[run_id]:.4f})")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating {run_id}: {hf_id}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            hf_id, torch_dtype=torch.bfloat16, device_map="cuda"
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size="auto")

        res = evaluator.simple_evaluate(
            model=lm,
            tasks=["openbookqa"],
            num_fewshot=0,
            log_samples=False,
            batch_size="auto",
            random_seed=42,
            numpy_random_seed=42,
            torch_random_seed=42,
            cache_requests=False,
            check_integrity=False,
        )

        acc_norm = res["results"]["openbookqa"]["acc_norm,none"]
        print(f"  acc_norm = {acc_norm:.4f}")
        results_map[run_id] = acc_norm

        # wandb run summary 업데이트
        run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
        run.summary["global_admm/openbookqa_acc"] = acc_norm
        run.summary.update()
        print(f"  wandb updated: global_admm/openbookqa_acc = {acc_norm:.4f}")

        # 중간 저장
        with open(OUT_PATH, "w") as f:
            json.dump(results_map, f, indent=2)

    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        try:
            del model, lm
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("All done. Results:")
for rid, v in sorted(results_map.items()):
    print(f"  {rid}: {v:.4f}")

with open(OUT_PATH, "w") as f:
    json.dump(results_map, f, indent=2)
print(f"Saved to {OUT_PATH}")
