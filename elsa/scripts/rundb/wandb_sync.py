"""Fetch metrics from wandb and update DB entries."""
from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"

# wandb summary key candidates → DB field
LIGHTEVAL_KEYS = {
    "math":   ["lighteval/math500", "math500/pass_at_1", "eval/math500"],
    "lcb":    ["lighteval/lcb", "eval/lcb"],
    "gpqa":   ["lighteval/gpqa_diamond", "eval/gpqa_diamond"],
    "ifeval": ["lighteval/ifeval_prompt", "eval/ifeval"],
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


def _get_first(summary: dict, keys: list[str]):
    for k in keys:
        v = summary.get(k)
        if v is not None:
            return v
    return None


def _pct(v) -> Optional[float]:
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v:  # nan
        return None
    if v <= 1.0:
        v *= 100.0
    return round(v, 1)


def _ppl(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def fetch_run_metrics(wbid: str, project: str, entity: str = ENTITY,
                      extra_wbid: Optional[str] = None) -> dict:
    """
    Fetch metrics for a completed wandb run.
    extra_wbid: optional second run to merge (e.g., training run that has PPL/zeroshot).
    Returns dict with DB field names as keys.
    """
    import wandb
    api = wandb.Api()

    def _load_summary(rid):
        run = api.run(f"{entity}/{project}/{rid}")
        return run.summary._json_dict, run.state

    try:
        s, state = _load_summary(wbid)
    except Exception as e:
        logger.error(f"Could not load run {wbid}: {e}")
        return {}

    if state not in ("finished", "crashed"):
        logger.warning(f"Run {wbid} state={state}, metrics may be incomplete")

    s2 = {}
    if extra_wbid and extra_wbid != wbid:
        try:
            s2, _ = _load_summary(extra_wbid)
        except Exception as e:
            logger.warning(f"Could not load extra run {extra_wbid}: {e}")

    def get(keys):
        v = _get_first(s, keys)
        if v is None and s2:
            v = _get_first(s2, keys)
        return v

    result = {}
    for field, keys in LIGHTEVAL_KEYS.items():
        result[field] = _pct(get(keys))
    for field, keys in ZEROSHOT_KEYS.items():
        v = get(keys)
        if v is not None:
            v = float(v)
            result[field] = round(v * 100 if v <= 1.0 else v, 1)
        else:
            result[field] = None
    for field, keys in PPL_KEYS.items():
        result[field] = _ppl(get(keys))

    missing = [k for k, v in result.items() if v is None]
    if missing:
        logger.warning(f"Run {wbid}: missing fields → {missing}")

    return result


def sync_entry(entry, project: str, entity: str = ENTITY) -> bool:
    """
    Pull metrics from wandb into a RunEntry. Returns True if any field was filled in.
    Uses entry.wbid; falls back to search by entry.wandb_name if wbid is None.
    """
    import wandb
    api = wandb.Api()

    wbid = entry.wbid
    if wbid is None and entry.wandb_name:
        # search by name
        proj = entry.wandb_project or project
        try:
            runs = api.runs(f"{entity}/{proj}",
                            filters={"display_name": entry.wandb_name, "state": "finished"})
            runs = list(runs)
            if runs:
                wbid = runs[0].id
                entry.wbid = wbid
                logger.info(f"Resolved '{entry.wandb_name}' → {wbid}")
        except Exception as e:
            logger.error(f"Run search failed: {e}")
            return False

    if not wbid:
        logger.warning(f"No wbid for entry '{entry.name}', skipping")
        return False

    proj = entry.wandb_project or project
    metrics = fetch_run_metrics(wbid, proj, entity,
                                extra_wbid=entry.train_wbid)
    if not metrics:
        return False

    updated = False
    for field, val in metrics.items():
        if val is not None and getattr(entry, field, None) is None:
            setattr(entry, field, val)
            updated = True

    if updated:
        entry.pending = False
    return updated


def sync_all_pending(db, verbose: bool = True) -> int:
    """Sync all pending entries in the DB. Returns count of updated entries."""
    pending = db.pending_entries()
    if not pending:
        if verbose:
            logger.info("No pending entries.")
        return 0

    updated = 0
    for model_key, tier, entry in pending:
        project = db.models[model_key].wb_project
        logger.info(f"Syncing {model_key}/{tier}/{entry.name} (wbid={entry.wbid})")
        if sync_entry(entry, project):
            updated += 1
            logger.info(f"  → updated: math={entry.math} lcb={entry.lcb} gpqa={entry.gpqa}")
        else:
            logger.warning(f"  → no update")

    return updated


def scan_new_runs(db, projects: list[str], entity: str = ENTITY,
                  limit: int = 50) -> list[dict]:
    """
    Scan wandb projects for finished runs not yet in DB.
    Returns list of dicts describing untracked runs.
    """
    import wandb
    api = wandb.Api()

    known_wbids = set()
    for model in db.models.values():
        for tier in ["S50", "S60", "S70"]:
            for e in getattr(model, tier):
                if e.wbid:
                    known_wbids.add(e.wbid)

    untracked = []
    for project in projects:
        try:
            runs = api.runs(f"{entity}/{project}",
                            filters={"state": "finished"},
                            order="-created_at")
        except Exception as e:
            logger.error(f"Could not scan {project}: {e}")
            continue

        count = 0
        for run in runs:
            if count >= limit:
                break
            if run.id in known_wbids:
                continue
            cfg = run.config
            untracked.append({
                "wbid": run.id,
                "name": run.name,
                "project": project,
                "sparsity": cfg.get("sparsity_ratio", cfg.get("sparsity")),
                "method": cfg.get("gmp_saliency", cfg.get("method", "unknown")),
                "model": cfg.get("model", cfg.get("model_path", "")),
                "created_at": run.created_at,
            })
            count += 1

    return untracked
