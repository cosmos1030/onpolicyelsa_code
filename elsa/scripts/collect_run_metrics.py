#!/usr/bin/env python3
"""
wandb run에서 아티팩트 DB용 JSON 한 줄 출력.

Usage:
  python scripts/collect_run_metrics.py <wbid> [--project reasoning_qwen3_4b]

Output (stdout):
  {"wbid":"xxx","math":83.8,"lcb":22.8,"gpqa":42.4,"ifeval":71.0,
   "wino":64.7,"rte":71.8,"race":40.4,"piqa":74.4,"ob":38.6,
   "hella":64.6,"boolq":83.5,"arce":76.1,"arcc":49.0,
   "c4":19.41,"wt2":14.17}

null = 해당 metric이 wandb에 아직 없음.
"""
import argparse, json, sys

def pct(v):
    """wandb 값(0~1 비율 또는 이미 퍼센트)을 퍼센트로 변환, 소수점 1자리."""
    if v is None or (isinstance(v, float) and (v != v)):  # nan
        return None
    v = float(v)
    if v <= 1.0:
        v *= 100.0
    return round(v, 1)

# wandb summary key → artifact field
LIGHTEVAL_KEYS = {
    "math":   ["lighteval/math500", "math500/pass_at_1"],
    "lcb":    ["lighteval/lcb"],
    "gpqa":   ["lighteval/gpqa_diamond"],
    "ifeval": ["lighteval/ifeval_prompt"],
}

ZEROSHOT_KEYS = {
    "wino":  ["eval/winogrande_acc", "winogrande/acc,none", "zeroshot/winogrande"],
    "rte":   ["eval/rte_acc", "rte/acc,none", "zeroshot/rte"],
    "race":  ["eval/race_acc", "race/acc,none", "zeroshot/race"],
    "piqa":  ["eval/piqa_acc_norm", "piqa/acc_norm,none", "zeroshot/piqa"],
    "ob":    ["eval/openbookqa_acc_norm", "openbookqa/acc_norm,none", "zeroshot/openbookqa"],
    "hella": ["eval/hellaswag_acc_norm", "hellaswag/acc_norm,none", "zeroshot/hellaswag"],
    "boolq": ["eval/boolq_acc", "boolq/acc,none", "zeroshot/boolq"],
    "arce":  ["eval/arc_easy_acc_norm", "arc_easy/acc_norm,none", "zeroshot/arc_easy"],
    "arcc":  ["eval/arc_challenge_acc_norm", "arc_challenge/acc_norm,none", "zeroshot/arc_challenge"],
}

PPL_KEYS = {
    "c4":  ["ppl/c4"],
    "wt2": ["ppl/wikitext2"],
}


def get_first(summary, keys):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            return v
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("wbid", help="eval wandb run ID")
    p.add_argument("--train_wbid", default=None, help="training wandb run ID (다르면 명시, 같으면 생략)")
    p.add_argument("--project", default="reasoning_qwen3_4b")
    p.add_argument("--entity", default="dyk6208-gwangju-institute-of-science-and-technology")
    p.add_argument("--full", action="store_true", help="include wbid + state in output")
    args = p.parse_args()

    import wandb
    api = wandb.Api()
    try:
        run = api.run(f"{args.entity}/{args.project}/{args.wbid}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    s = run.summary._json_dict

    # training run summary도 merge (eval run에 없는 metric은 여기서 fallback)
    train_s = {}
    if args.train_wbid and args.train_wbid != args.wbid:
        try:
            train_run = api.run(f"{args.entity}/{args.project}/{args.train_wbid}")
            train_s = train_run.summary._json_dict
        except Exception as e:
            print(f"WARNING: train run fetch failed ({e})", file=sys.stderr)

    def get_first_merged(keys):
        v = get_first(s, keys)
        if v is None and train_s:
            v = get_first(train_s, keys)
        return v

    result = {}
    if args.full:
        result["wbid"] = args.wbid
        if args.train_wbid and args.train_wbid != args.wbid:
            result["train_wbid"] = args.train_wbid
        result["state"] = run.state

    # Reasoning (lighteval) — eval run 우선, 없으면 training run fallback
    for field, keys in LIGHTEVAL_KEYS.items():
        v = get_first_merged(keys)
        result[field] = pct(v)

    # Zero-shot (9 tasks)
    for field, keys in ZEROSHOT_KEYS.items():
        v = get_first_merged(keys)
        if v is not None:
            v = float(v)
            # eval_full.py logs as 0~1; global_admm/* logs as 0~100
            result[field] = round(v * 100 if v <= 1.0 else v, 1)
        else:
            result[field] = None

    # PPL — raw value (lower=better, no scaling)
    for field, keys in PPL_KEYS.items():
        v = get_first_merged(keys)
        result[field] = round(float(v), 2) if v is not None else None

    print(json.dumps(result, separators=(",", ":")))

    # 누락 필드 경고
    missing = [k for k, v in result.items() if v is None and k not in ("wbid", "state")]
    if missing:
        print(f"WARNING: 누락 필드 → {missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
