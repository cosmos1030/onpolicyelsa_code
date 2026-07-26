#!/usr/bin/env python3
"""
rundb CLI — manage pruning experiment results and artifact.

Commands:
  sync          Pull pending metrics from wandb, rebuild & redeploy artifact
  add-run       Add a new pending run entry to the DB
  build         Rebuild artifact HTML from DB (no wandb sync)
  status        Show DB summary and pending runs
  fetch-run     Fetch metrics for a single wandb run ID (prints JSON)
  scan          Scan wandb projects for untracked finished runs
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCRIPTS_DIR)

from rundb.schema import DB, DB_PATH, TIERS, MODELS, METRIC_FIELDS
from rundb.builder import build_artifact_html

ARTIFACT_OUTPUT = os.path.join(SCRIPTS_DIR, "artifact_built.html")
ARTIFACT_URL = "https://claude.ai/code/artifact/b81a7023-3366-4d46-b5eb-02efbb37da8d"


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_db() -> DB:
    return DB.load(DB_PATH)


def _rebuild(db, output_path: str = ARTIFACT_OUTPUT) -> str:
    html = build_artifact_html(db, output_path=output_path)
    logger.info(f"Artifact HTML written → {output_path}")
    return html


# ─── commands ─────────────────────────────────────────────────────────────────

def cmd_sync(args):
    """Pull all pending/incomplete metrics from wandb, save DB, rebuild HTML."""
    from rundb.wandb_sync import sync_all_pending

    db = _load_db()
    updated = sync_all_pending(db, verbose=True)

    if updated > 0 or args.force:
        db.save()
        logger.info(f"DB saved ({updated} entries updated)")
        _rebuild(db)
    else:
        logger.info("Nothing to update.")

    if args.status:
        cmd_status(args)

    return updated


def cmd_add_run(args):
    """Register a new run as pending so sync will fill it in later."""
    db = _load_db()
    model = db.models.get(args.model)
    if model is None:
        logger.error(f"Unknown model '{args.model}'. Valid: {list(db.models.keys())}")
        sys.exit(1)

    tier = f"S{int(args.sparsity * 100)}"
    if tier not in TIERS:
        logger.error(f"Sparsity {args.sparsity} → tier '{tier}' not in {TIERS}")
        sys.exit(1)

    from rundb.schema import RunEntry
    entry = RunEntry(
        badge=args.badge,
        name=args.name,
        sub=args.sub or "",
        wbid=args.wbid,
        wandb_name=args.wandb_name,
        wandb_project=args.project or model.wb_project,
        hf=args.hf,
        steps=args.steps,
        pending=True,
    )
    getattr(model, tier).append(entry)
    db.save()
    logger.info(f"Added {args.model}/{tier}: '{args.name}' (wbid={args.wbid or '?'}, pending=True)")

    if args.sync:
        from rundb.wandb_sync import sync_entry
        if sync_entry(entry, model.wb_project):
            db.save()
            _rebuild(db)
            logger.info("Synced immediately and rebuilt.")


def cmd_build(args):
    """Rebuild artifact HTML from current DB (no wandb sync)."""
    db = _load_db()
    _rebuild(db)


def cmd_status(args):
    """Print DB summary."""
    db = _load_db()
    print("\n=== rundb status ===")
    for model_key, model in db.models.items():
        print(f"\n{model_key} ({model.display})")
        for tier in TIERS:
            entries = getattr(model, tier)
            pending = [e for e in entries if e.pending or (e.wbid and e.missing_metrics())]
            complete = [e for e in entries if not e.pending and not e.missing_metrics()]
            print(f"  {tier}: {len(entries)} total  "
                  f"{len(complete)} complete  {len(pending)} pending/incomplete")
            for e in pending:
                missing = e.missing_metrics()
                print(f"    ⚠ [{e.badge}] {e.name}  wbid={e.wbid}  missing={missing[:4]}{'...' if len(missing)>4 else ''}")
    print()


def cmd_fetch_run(args):
    """Fetch metrics for a single wandb run and print as JSON."""
    from rundb.wandb_sync import fetch_run_metrics
    metrics = fetch_run_metrics(args.wbid, args.project)
    print(json.dumps(metrics, indent=2))


def cmd_register(args):
    """
    Register a completed run: find existing entry by name+badge or create new one,
    set its wbid, then immediately sync metrics. Supports multiple sparsity tiers
    (milestone runs that produce S50/S60/S70 from one training run).
    """
    from rundb.schema import RunEntry
    from rundb.wandb_sync import sync_entry

    db = _load_db()
    model = db.models.get(args.model)
    if model is None:
        logger.error(f"Unknown model '{args.model}'. Valid: {list(db.models.keys())}")
        sys.exit(1)

    sparsities = [float(s) for s in args.sparsities.split(",")]
    project = args.project or model.wb_project
    updated_any = False

    for sp in sparsities:
        tier = f"S{int(sp * 100)}"
        entries = getattr(model, tier, [])

        # Find existing entry with matching badge and name
        match = None
        for e in entries:
            if e.badge == args.badge and e.name == args.name:
                match = e
                break

        if match:
            if match.wbid and match.wbid != args.wbid:
                logger.warning(f"  {tier}: entry '{args.name}' already has wbid={match.wbid}, overwriting with {args.wbid}")
            match.wbid = args.wbid
            match.wandb_project = project
            match.pending = True
            entry = match
            logger.info(f"  {tier}: updated existing entry wbid={args.wbid}")
        else:
            entry = RunEntry(
                badge=args.badge,
                name=args.name,
                sub=args.sub or "",
                wbid=args.wbid,
                wandb_project=project,
                hf=args.hf,
                pending=True,
            )
            entries.append(entry)
            logger.info(f"  {tier}: added new entry wbid={args.wbid}")

        # Sync metrics immediately
        if sync_entry(entry, project):
            logger.info(f"  {tier}: synced — math={entry.math} gpqa={entry.gpqa}")
            updated_any = True
        else:
            logger.warning(f"  {tier}: sync failed (metrics may appear later)")

    db.save()
    if updated_any:
        _rebuild(db)
        logger.info("DB saved and artifact rebuilt.")
    else:
        db.save()
        logger.info("DB saved (no metrics fetched yet, will retry on next sync).")


def cmd_scan(args):
    """Scan wandb for untracked finished runs."""
    from rundb.wandb_sync import scan_new_runs
    db = _load_db()
    projects = args.projects or ["reasoning_qwen3_4b", "reasoning_pruning_v1"]
    untracked = scan_new_runs(db, projects, limit=args.limit)
    if not untracked:
        print("No untracked finished runs found.")
        return
    print(f"\n{len(untracked)} untracked finished runs:\n")
    for r in untracked:
        print(f"  [{r['project']}] {r['wbid']}  name={r['name']}  "
              f"sparsity={r['sparsity']}  method={r['method']}")
    print()
    print("Add with: python rundb/cli.py add-run --model <model> --sparsity <s> "
          "--badge <badge> --name <name> --wbid <id> --sync")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="rundb: manage pruning experiment results")
    sub = p.add_subparsers(dest="cmd")

    # sync
    ps = sub.add_parser("sync", help="Pull pending metrics, save DB, rebuild artifact")
    ps.add_argument("--force", action="store_true", help="Rebuild even if nothing changed")
    ps.add_argument("--status", action="store_true", help="Print status after sync")

    # add-run
    pa = sub.add_parser("add-run", help="Register a new pending run")
    pa.add_argument("--model", required=True, choices=MODELS)
    pa.add_argument("--sparsity", type=float, required=True, help="e.g. 0.5")
    pa.add_argument("--badge", required=True,
                    help="badge key: dense/sgpt/wanda/gmp/opkd/tr/tropkd/tropkd_mag/tropkd_layer/elsa/safe/alps")
    pa.add_argument("--name", required=True, help="Display name")
    pa.add_argument("--sub", default="", help="Subtitle")
    pa.add_argument("--wbid", default=None, help="wandb run ID (eval)")
    pa.add_argument("--wandb-name", dest="wandb_name", default=None,
                    help="wandb run display name (for lookup when wbid unknown)")
    pa.add_argument("--project", default=None, help="wandb project override")
    pa.add_argument("--hf", default=None, help="HuggingFace repo")
    pa.add_argument("--steps", type=int, default=None)
    pa.add_argument("--sync", action="store_true", help="Immediately sync from wandb")

    # build
    sub.add_parser("build", help="Rebuild artifact HTML from current DB")

    # status
    sub.add_parser("status", help="Show DB summary and pending entries")

    # fetch-run
    pf = sub.add_parser("fetch-run", help="Fetch metrics for one wandb run")
    pf.add_argument("wbid")
    pf.add_argument("--project", default="reasoning_qwen3_4b")

    # register
    pr = sub.add_parser("register", help="Register a completed run (find/create entry + sync)")
    pr.add_argument("--model", required=True, choices=MODELS)
    pr.add_argument("--sparsities", required=True,
                    help="Comma-separated sparsity tiers, e.g. '0.5,0.6,0.7'")
    pr.add_argument("--badge", required=True)
    pr.add_argument("--name", required=True, help="Display name (must match existing entry to update)")
    pr.add_argument("--sub", default="")
    pr.add_argument("--wbid", required=True, help="wandb run ID")
    pr.add_argument("--project", default=None)
    pr.add_argument("--hf", default=None)

    # scan
    psc = sub.add_parser("scan", help="Scan wandb for untracked finished runs")
    psc.add_argument("--projects", nargs="+", default=None)
    psc.add_argument("--limit", type=int, default=50)

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return

    dispatch = {
        "sync": cmd_sync,
        "add-run": cmd_add_run,
        "register": cmd_register,
        "build": cmd_build,
        "status": cmd_status,
        "fetch-run": cmd_fetch_run,
        "scan": cmd_scan,
    }
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
