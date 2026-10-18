"""Re-score saved lighteval generations without running the model again.

Reads a details parquet written by an earlier eval (one row per sample: the
Doc, the ModelResponse, the metric), re-applies the reasoning-tag removal --
including the orphan-close-tag rule from lighteval_patched_runner.py, which
OLMo 3 Think needs because its chat template opens <think> in the prompt --
and recomputes the task's own lighteval metrics. CPU only.

    python elsa/scripts/rescore_lighteval_details.py \
        --details <out>/ifeval/details/<model>/<date>/details_lighteval|ifeval|0_<date>.parquet \
        --task "lighteval|ifeval|0"
"""
import argparse
import json

import numpy as np


def remove_reasoning_tags(text, tag_pairs=(("<think>", "</think>"),)):
    # same rule as lighteval_patched_runner._patch_orphan_reasoning_close
    for start_tag, end_tag in tag_pairs:
        end = text.find(end_tag)
        if end != -1 and start_tag not in text[:end]:
            text = text[end + len(end_tag):]
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag):]
            else:
                break
    return result


def _stock_remove(text, tag_pairs=(("<think>", "</think>"),)):
    result = text
    for start_tag, end_tag in tag_pairs:
        while start_tag in result and end_tag in result:
            start = result.find(start_tag)
            end = result.find(end_tag, start)
            if start != -1 and end != -1:
                result = result[:start] + result[end + len(end_tag):]
            else:
                break
    return result


def _plain(x):
    """parquet round-trip turns lists into numpy arrays; undo that recursively."""
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, np.generic):
        return x.item()
    return x


def _doc_types(x):
    """Doc fields also lose their types in parquet: ints in nullable columns come
    back as floats and None as NaN (IFEval's instruction kwargs: num_words=101.0,
    unused keys NaN), which the instruction checkers then use as list indices."""
    if isinstance(x, float):
        if x != x:
            return None
        if x.is_integer():
            return int(x)
        return x
    if isinstance(x, list):
        return [_doc_types(v) for v in x]
    if isinstance(x, dict):
        return {k: _doc_types(v) for k, v in x.items()}
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", required=True)
    ap.add_argument("--task", required=True, help='e.g. "lighteval|ifeval|0"')
    ap.add_argument("--no_orphan_rule", action="store_true",
                    help="stock lighteval tag removal (to check this reproduces the original score)")
    args = ap.parse_args()

    import pandas as pd
    from lighteval.metrics import apply_metric
    from lighteval.models.model_output import ModelResponse
    from lighteval.tasks.registry import Registry
    from lighteval.tasks.requests import Doc

    task = Registry(tasks=args.task).load_tasks()[args.task]
    df = pd.read_parquet(args.details)
    docs, responses = [], []
    for _, row in df.iterrows():
        d = _doc_types(_plain(row["doc"]))
        docs.append(Doc(**{k: v for k, v in d.items() if k in Doc.__dataclass_fields__}))
        r = _plain(row["model_response"])
        mr = ModelResponse(**{k: v for k, v in r.items() if k in ModelResponse.__dataclass_fields__})
        mr.text_post_processed = [(_stock_remove(t) if args.no_orphan_rule else remove_reasoning_tags(t))
                                  for t in mr.text]
        responses.append(mr)

    metrics = [m for m in task.metrics]
    outputs = apply_metric(responses=responses, docs=docs, metrics=metrics)
    per_key = {}
    for o in outputs:
        for k, v in o.items():
            per_key.setdefault(k, []).append(v)
    result = {}
    for m in metrics:
        for name, fn in m.get_corpus_aggregations().items():
            if name in per_key:
                result[name] = float(fn(per_key[name]))
    print("RESCORED", args.task, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
