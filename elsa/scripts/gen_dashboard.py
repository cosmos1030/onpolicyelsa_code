#!/usr/bin/env python3
"""
Generate dashboard.html from live data:
  - wandb API  → runs, metrics, sweep state
  - squeue     → slurm job status
  - output_qwen/ log files → timing, results not in wandb

Usage:
  python scripts/gen_dashboard.py
  python scripts/gen_dashboard.py --open   # open in browser after generation
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"
PROJECTS = ["elsa_qwen3_0.6b", "qwen3_gkd", "qwen3_dense_gkd_eval",
            "rac_qwen3_0.6_pruning", "prune_eval_test", "qwen3_gkd_test"]
LOG_DIRS = [
    "/home1/doyoonkim/projects/elsa/output_qwen",
    "/home1/doyoonkim/projects/RAC/open-r1-main/output_qwen",
]
OUT_HTML = Path("/home1/doyoonkim/projects/elsa/dashboard.html")

# ── known manual entries (things not in wandb) ──────────────────────────────
MANUAL_RESULTS = [
    {
        "model": "Qwen3-0.6B", "sparsity": "0%", "method": "Dense",
        "math500": 0.726, "note": "Job 299607, 28.3min, max_new_tokens=32768",
        "wandb": None,
    },
    {
        "model": "Qwen3-0.6B", "sparsity": "30%", "method": "SparseGPT (C4)",
        "math500": 0.542, "note": "Job 296842, ⚠ max_new_tokens=4096 (낮게 나왔을 수 있음)",
        "wandb": None,
    },
    {
        "model": "DS-R1-Distill-1.5B", "sparsity": "0%", "method": "Dense",
        "math500": 0.832, "note": "RAC 논문", "wandb": None,
    },
    {
        "model": "DS-R1-Distill-1.5B", "sparsity": "50%", "method": "RAC (prompt+CoT)",
        "math500": 0.664, "note": "RAC 논문", "wandb": None,
    },
    {
        "model": "DS-R1-Distill-7B", "sparsity": "0%", "method": "Dense",
        "math500": 0.936, "note": "RAC 논문", "wandb": None,
    },
    {
        "model": "DS-R1-Distill-7B", "sparsity": "50%", "method": "RAC (prompt+CoT)",
        "math500": 0.900, "note": "RAC 논문", "wandb": None,
    },
]

# ── slurm ────────────────────────────────────────────────────────────────────
def get_slurm_jobs() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", os.environ.get("USER", "doyoonkim"),
             "--format=%i|%j|%T|%M|%l|%R|%P", "--noheader"],
            text=True
        )
        jobs = []
        for line in out.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 7:
                continue
            jobs.append({
                "job_id": parts[0].strip(),
                "name":   parts[1].strip(),
                "state":  parts[2].strip(),
                "time":   parts[3].strip(),
                "limit":  parts[4].strip(),
                "reason": parts[5].strip(),
                "partition": parts[6].strip(),
            })
        return jobs
    except Exception as e:
        return []


# ── log file parsing ─────────────────────────────────────────────────────────
def parse_log(job_id: str) -> dict:
    info = {}
    for log_dir in LOG_DIRS:
        log = Path(log_dir) / f"{job_id}.out"
        if not log.exists():
            continue
        text = log.read_text(errors="replace")

        # MATH-500 result
        m = re.search(r"MATH-500 pass@1\s*=\s*([0-9.]+)", text)
        if m:
            info["math500"] = float(m.group(1))

        # RESULT: pass@1
        m = re.search(r"RESULT:\s*pass@1\s*=\s*([0-9.]+)", text)
        if m:
            info.setdefault("math500", float(m.group(1)))

        # elapsed time
        m = re.search(r"ELAPSED:\s*([0-9.]+)s\s*\(([^)]+)\)", text)
        if m:
            info["elapsed"] = m.group(2)

        # total time from footer
        m = re.search(r"Total:\s*(\d+)\s*min", text)
        if m:
            info.setdefault("elapsed", f"{m.group(1)} min")

        # wandb run URL
        m = re.search(r"View run at:\s*(https://wandb\.ai/\S+)", text)
        if m:
            info["wandb_url"] = m.group(1)

        # wandb run ID from URL
        m = re.search(r"wandb\.ai/[^/]+/[^/]+/runs/([a-z0-9]+)", text)
        if m:
            info["wandb_run_id"] = m.group(1)

        info["log_path"] = str(log)
        break
    return info


# ── wandb ────────────────────────────────────────────────────────────────────
def get_wandb_runs() -> list[dict]:
    try:
        import wandb
        api = wandb.Api(timeout=30)
        all_runs = []
        for proj in PROJECTS:
            try:
                runs = api.runs(f"{ENTITY}/{proj}", per_page=50)
                for r in runs:
                    cfg = dict(r.config)
                    summary = dict(r.summary)
                    all_runs.append({
                        "project":    proj,
                        "id":         r.id,
                        "name":       r.name,
                        "state":      r.state,
                        "created_at": r.created_at,
                        "url":        r.url,
                        "math500":    summary.get("math500_pass@1") or summary.get("eval/math500_pass@1"),
                        "model":      cfg.get("model", ""),
                        "sparsity":   cfg.get("sparsity_ratio", ""),
                        "dataset":    cfg.get("dataset", ""),
                        "admm_lr":    cfg.get("admm_lr", ""),
                        "admm_lmda":  cfg.get("admm_lmda", ""),
                        "admm_steps": cfg.get("admm_steps", ""),
                        "sweep_id":   r.sweep.id if r.sweep else None,
                        "sweep_name": r.sweep.name if r.sweep else None,
                    })
            except Exception:
                continue
        return all_runs
    except Exception:
        return []


def get_wandb_sweeps() -> list[dict]:
    """Pull sweeps via runs: collect unique sweep IDs from run metadata."""
    try:
        import wandb
        api = wandb.Api(timeout=30)
        sweeps: dict[str, dict] = {}
        for proj in PROJECTS:
            try:
                runs = api.runs(f"{ENTITY}/{proj}", per_page=100)
                for r in runs:
                    if not r.sweep:
                        continue
                    sid = r.sweep.id
                    if sid not in sweeps:
                        sweeps[sid] = {
                            "project": proj,
                            "id": sid,
                            "name": r.sweep.name if hasattr(r.sweep, "name") else sid,
                            "state": "unknown",
                            "run_count": 0,
                            "best_metric": None,
                            "url": f"https://wandb.ai/{ENTITY}/{proj}/sweeps/{sid}",
                        }
                    sweeps[sid]["run_count"] += 1
                    math500 = (dict(r.summary).get("math500_pass@1")
                               or dict(r.summary).get("eval/math500_pass@1"))
                    if math500 is not None:
                        prev = sweeps[sid]["best_metric"]
                        if prev is None or float(math500) > float(prev):
                            sweeps[sid]["best_metric"] = math500
                    # infer state from run states
                    if r.state == "running":
                        sweeps[sid]["state"] = "running"
                    elif sweeps[sid]["state"] != "running" and r.state == "finished":
                        sweeps[sid]["state"] = "finished"
            except Exception:
                continue
        return list(sweeps.values())
    except Exception:
        return []


# ── HTML helpers ─────────────────────────────────────────────────────────────
STATE_BADGE = {
    "RUNNING":   ("running",   "running"),
    "PENDING":   ("pending",   "pending"),
    "COMPLETED": ("done",      "done"),
    "FAILED":    ("failed",    "failed"),
    "CANCELLED": ("cancelled", "cancelled"),
    "running":   ("running",   "running"),
    "finished":  ("done",      "done"),
    "crashed":   ("failed",    "crashed"),
    "failed":    ("failed",    "failed"),
}

def badge(state: str) -> str:
    cls, label = STATE_BADGE.get(state, ("pending", state))
    return f'<span class="badge {cls}">{label}</span>'

def score_html(v) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
        cls = "result-good" if f >= 0.7 else "result-ok" if f >= 0.5 else "result-bad"
        return f'<span class="{cls}">{f:.3f}</span>'
    except Exception:
        return str(v)

def wandb_link(url: Optional[str], label: str = "↗") -> str:
    if not url:
        return "—"
    return f'<a href="{url}" target="_blank">{label} ↗</a>'

def code(s) -> str:
    return f"<code>{s}</code>" if s else ""

def path_td(p: str) -> str:
    return f'<span class="path">{p}</span>'


# ── HTML generation ───────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #222; padding: 24px; }
h1 { font-size: 1.6rem; margin-bottom: 4px; }
.subtitle { color: #666; font-size: 0.9rem; margin-bottom: 24px; }
.section { background: #fff; border-radius: 10px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.section h2 { font-size: 1.1rem; margin-bottom: 14px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 0.86rem; }
th { background: #f6f8fa; text-align: left; padding: 8px 10px; font-weight: 600; border-bottom: 2px solid #e0e0e0; white-space: nowrap; }
td { padding: 8px 10px; border-bottom: 1px solid #f0f0f0; vertical-align: top; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
.badge.running  { background: #e3f2fd; color: #1565c0; }
.badge.done     { background: #e8f5e9; color: #2e7d32; }
.badge.failed   { background: #ffebee; color: #c62828; }
.badge.pending  { background: #fff8e1; color: #f57f17; }
.badge.cancelled{ background: #f3e5f5; color: #6a1b9a; }
.result-good { color: #2e7d32; font-weight: 700; }
.result-ok   { color: #e65100; font-weight: 700; }
.result-bad  { color: #c62828; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 3px; font-size: 0.85em; font-family: monospace; }
.path { font-family: monospace; font-size: 0.80em; color: #555; word-break: break-all; }
.note { color: #777; font-size: 0.80em; margin-top: 3px; }
a { color: #1a73e8; text-decoration: none; }
a:hover { text-decoration: underline; }
.tag { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.78rem; background: #eef2ff; color: #3730a3; margin: 1px; }
"""


def render(slurm_jobs, wandb_runs, wandb_sweeps) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    slurm_map = {j["job_id"]: j for j in slurm_jobs}

    # ── Section 1: Slurm Jobs ────────────────────────────────────────────────
    slurm_rows = ""
    for j in slurm_jobs:
        log_info = parse_log(j["job_id"])
        wandb_url = log_info.get("wandb_url")
        wandb_run_id = log_info.get("wandb_run_id")
        wandb_cell = wandb_link(wandb_url, wandb_run_id or "run") if wandb_url else "—"
        log_path = log_info.get("log_path", "")
        math500 = score_html(log_info.get("math500"))
        elapsed = log_info.get("elapsed", "")

        slurm_rows += f"""
        <tr>
          <td><b>{j["job_id"]}</b></td>
          <td>{badge(j["state"])}</td>
          <td>{j["name"]}<div class="note">{j["partition"]} | {j["time"]} / {j["limit"]}</div></td>
          <td>{path_td(log_path) if log_path else "—"}</td>
          <td>{wandb_cell}</td>
          <td>{math500}<div class="note">{elapsed}</div></td>
        </tr>"""

    # ── Section 2: wandb Runs ────────────────────────────────────────────────
    # filter: only recent meaningful runs (has model config)
    meaningful = [r for r in wandb_runs if r.get("model")]
    meaningful.sort(key=lambda r: r["created_at"], reverse=True)

    wandb_rows = ""
    for r in meaningful[:30]:
        cfg_tags = ""
        if r["model"]:
            cfg_tags += f'<span class="tag">{r["model"].split("/")[-1]}</span>'
        if r["sparsity"]:
            cfg_tags += f'<span class="tag">s={r["sparsity"]}</span>'
        if r["dataset"]:
            cfg_tags += f'<span class="tag">{r["dataset"]}</span>'
        if r["admm_lr"]:
            cfg_tags += f'<span class="tag">lr={r["admm_lr"]}</span>'
        if r["admm_lmda"]:
            cfg_tags += f'<span class="tag">λ={r["admm_lmda"]}</span>'
        sweep_cell = wandb_link(
            f"https://wandb.ai/{ENTITY}/{r['project']}/sweeps/{r['sweep_id']}" if r["sweep_id"] else None,
            r["sweep_id"] or ""
        )
        created = r["created_at"][:10] if r["created_at"] else ""
        wandb_rows += f"""
        <tr>
          <td>{code(r["id"])}<div class="note">{created}</div></td>
          <td>{badge(r["state"])}</td>
          <td>{r["project"]}</td>
          <td>{r["name"]}<br>{cfg_tags}</td>
          <td>{sweep_cell}</td>
          <td>{wandb_link(r["url"], "run ↗")}</td>
          <td>{score_html(r["math500"])}</td>
        </tr>"""

    # ── Section 3: Sweeps ────────────────────────────────────────────────────
    sweep_rows = ""
    for s in wandb_sweeps:
        sweep_rows += f"""
        <tr>
          <td>{code(s["id"])}</td>
          <td>{s["project"]}</td>
          <td>{s["name"]}</td>
          <td>{badge(s.get("state","?"))}</td>
          <td>{s.get("run_count","?")}</td>
          <td>{score_html(s.get("best_metric"))}</td>
          <td>{wandb_link(s["url"])}</td>
        </tr>"""

    if not sweep_rows:
        sweep_rows = "<tr><td colspan='7' style='color:#aaa'>wandb sweep 정보 없음</td></tr>"

    # ── Section 4: Known Results ─────────────────────────────────────────────
    result_rows = ""
    # also pull from wandb runs that have math500
    for r in meaningful:
        if r.get("math500") is not None:
            model_short = r["model"].split("/")[-1] if r["model"] else "?"
            sparsity = f"{int(float(r['sparsity'])*100)}%" if r["sparsity"] else "?"
            method = f"ELSA ({r['dataset']}, lr={r['admm_lr']}, λ={r['admm_lmda']})"
            MANUAL_RESULTS.append({
                "model": model_short, "sparsity": sparsity, "method": method,
                "math500": r["math500"], "note": f"run {r['id']}",
                "wandb": r["url"],
            })

    seen = set()
    for row in MANUAL_RESULTS:
        key = (row["model"], row["sparsity"], row["method"])
        if key in seen:
            continue
        seen.add(key)
        result_rows += f"""
        <tr>
          <td>{row["model"]}</td>
          <td>{row["sparsity"]}</td>
          <td>{row["method"]}</td>
          <td>{score_html(row["math500"])}</td>
          <td>{wandb_link(row.get("wandb"))}<div class="note">{row.get("note","")}</div></td>
        </tr>"""

    slurm_section = (
        "<p style='color:#aaa'>실행 중인 job 없음</p>" if not slurm_jobs else
        f"<table><tr><th>Job ID</th><th>상태</th><th>이름 / 파티션 / 시간</th><th>로그</th><th>wandb</th><th>결과</th></tr>{slurm_rows}</table>"
    )
    runs_section = (
        "<p style='color:#aaa'>wandb 연결 실패</p>" if not wandb_runs else
        f"<table><tr><th>Run ID</th><th>상태</th><th>Project</th><th>이름 / 설정</th><th>Sweep</th><th>wandb</th><th>MATH-500</th></tr>{wandb_rows}</table>"
    )

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8"/>
  <title>ELSA Dashboard</title>
  <style>{CSS}</style>
</head>
<body>
<h1>ELSA Experiment Dashboard</h1>
<p class="subtitle">자동생성: {now} | <a href="elsa_math.html">수식 설명 ↗</a></p>

<div class="section">
  <h2>⚡ 현재 Slurm Jobs</h2>
  {slurm_section}
</div>

<div class="section">
  <h2>🔄 wandb Sweeps</h2>
  <table>
    <tr><th>ID</th><th>Project</th><th>이름</th><th>상태</th><th>Runs</th><th>Best math500</th><th>링크</th></tr>
    {sweep_rows}
  </table>
</div>

<div class="section">
  <h2>📋 wandb Runs (최근 30개)</h2>
  {runs_section}
</div>

<div class="section">
  <h2>📈 알려진 결과 (MATH-500 pass@1)</h2>
  <table>
    <tr><th>모델</th><th>Sparsity</th><th>방법</th><th>MATH-500</th><th>출처</th></tr>
    {result_rows}
  </table>
</div>

<div class="section">
  <h2>📁 주요 경로</h2>
  <table>
    <tr><th>설명</th><th>경로</th></tr>
    <tr><td>ELSA</td><td>{path_td("/home1/doyoonkim/projects/elsa/")}</td></tr>
    <tr><td>RAC</td><td>{path_td("/home1/doyoonkim/projects/RAC/open-r1-main/")}</td></tr>
    <tr><td>Math CoT data</td><td>{path_td("/home1/doyoonkim/projects/elsa/data/math_220k_cot.jsonl")}</td></tr>
    <tr><td>Math Prompts data</td><td>{path_td("/home1/doyoonkim/projects/elsa/data/math_220k_prompts.jsonl")}</td></tr>
    <tr><td>모델 저장</td><td>{path_td("/home1/doyoonkim/projects/elsa/models/")}</td></tr>
    <tr><td>ELSA 로그</td><td>{path_td("/home1/doyoonkim/projects/elsa/output_qwen/")}</td></tr>
    <tr><td>RAC 로그</td><td>{path_td("/home1/doyoonkim/projects/RAC/open-r1-main/output_qwen/")}</td></tr>
    <tr><td>Sweep config</td><td>{path_td("/home1/doyoonkim/projects/elsa/config/sweep_qwen_ntp_cot.yaml")}</td></tr>
    <tr><td>대시보드 생성 스크립트</td><td>{path_td("/home1/doyoonkim/projects/elsa/scripts/gen_dashboard.py")}</td></tr>
    <tr><td>수식 설명</td><td>{path_td("/home1/doyoonkim/projects/elsa/elsa_math.html")}</td></tr>
    <tr><td>Dense Qwen3-0.6B 캐시</td><td>{path_td("/home1/doyoonkim/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")}</td></tr>
  </table>
</div>

</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--open", action="store_true", help="Open in browser after generation")
    args = parser.parse_args()

    print("Fetching slurm jobs...", flush=True)
    slurm_jobs = get_slurm_jobs()
    print(f"  {len(slurm_jobs)} jobs found")

    print("Fetching wandb runs...", flush=True)
    wandb_runs = get_wandb_runs()
    print(f"  {len(wandb_runs)} runs found")

    print("Fetching wandb sweeps...", flush=True)
    wandb_sweeps = get_wandb_sweeps()
    print(f"  {len(wandb_sweeps)} sweeps found")

    html = render(slurm_jobs, wandb_runs, wandb_sweeps)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"Written: {OUT_HTML}")

    if args.open:
        import webbrowser
        webbrowser.open(str(OUT_HTML))


if __name__ == "__main__":
    main()