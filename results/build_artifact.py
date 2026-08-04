#!/usr/bin/env python3
"""
Build the pruning results artifact from runs_db.json.

Usage:
    python build_artifact.py               # render HTML from current runs_db.json
    python build_artifact.py --sync        # pull wandb metrics for pending/missing runs first
    python build_artifact.py --sync --all  # re-fetch ALL runs (not just pending/missing)

The artifact HTML is written to deepseek_results.html in the same directory.
"""
import argparse, json, math, os, sys
from pathlib import Path

DB_PATH  = Path(__file__).parent / "runs_db.json"
OUT_PATH = Path(__file__).parent / "deepseek_results.html"

ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"

# wandb summary key candidates (tried in order, first hit wins)
METRIC_KEYS = {
    "math":  ["math500_pass@1", "lighteval/math500"],
    "lcb":   ["lighteval/lcb"],
    "gpqa":  ["lighteval/gpqa_diamond"],
    "ifeval":["lighteval/ifeval_prompt"],
    "c4":    ["ppl_test(c4)", "ppl/c4"],
    "wt2":   ["ppl_test(wikitext2)", "ppl/wikitext2"],
    "wino":  ["global_admm/winogrande_acc", "zero_shot/winogrande", "zeroshot/winogrande"],
    "rte":   ["global_admm/rte_acc",        "zero_shot/rte",        "zeroshot/rte"],
    "race":  ["global_admm/race_acc",        "zero_shot/race",       "zeroshot/race"],
    "piqa":  ["global_admm/piqa_acc",        "zero_shot/piqa",       "zeroshot/piqa"],
    "ob":    ["global_admm/openbookqa_acc",  "zero_shot/openbookqa", "zeroshot/openbookqa"],
    "hella": ["global_admm/hellaswag_acc",   "zero_shot/hellaswag",  "zeroshot/hellaswag"],
    "boolq": ["global_admm/boolq_acc",       "zero_shot/boolq",      "zeroshot/boolq"],
    "arce":  ["global_admm/arc_easy_acc",    "zero_shot/arc_easy",   "zeroshot/arc_easy"],
    "arcc":  ["global_admm/arc_challenge_acc","zero_shot/arc_challenge","zeroshot/arc_challenge"],
    "steps": ["step"],
}
# these raw values are 0-1 fractions → multiply by 100
SCALE_100 = {"math","lcb","gpqa","ifeval","wino","rte","race","piqa","ob","hella","boolq","arce","arcc"}

REASONING_COLS = ["math","lcb","gpqa","ifeval"]
ZEROSHOT_COLS  = ["wino","rte","race","piqa","ob","hella","boolq","arce","arcc"]
PPL_COLS       = ["c4","wt2"]


def fetch_run(api, project, wbid):
    """Pull metrics from wandb for a single run. Returns dict of metric → value."""
    try:
        r = api.run(f"{ENTITY}/{project}/{wbid}")
        s = r.summary
        out = {}
        for metric, candidates in METRIC_KEYS.items():
            for key in candidates:
                v = s.get(key)
                if v is not None:
                    val = round(v * 100, 1) if metric in SCALE_100 else round(float(v), 2)
                    out[metric] = val
                    break
        hf_id = s.get("hub_model_id")
        if hf_id:
            out["hf"] = hf_id
        return out, r.state
    except Exception as e:
        print(f"  WARN: could not fetch {wbid}: {e}", file=sys.stderr)
        return {}, None


def sync_db(db, sync_all=False):
    """Pull wandb metrics into runs_db for pending or missing-metric runs."""
    import wandb
    api = wandb.Api()
    changed = False

    for model_key, model in db["models"].items():
        project = model["wb_project"]
        for section in ["S50", "S60", "S70"]:
            for run in model.get(section, []):
                wbid = run.get("wbid")
                if not wbid:
                    continue
                # decide whether to fetch
                needs_fetch = (
                    sync_all
                    or run.get("pending")
                    or any(run.get(m) is None for m in REASONING_COLS)
                )
                if not needs_fetch:
                    continue

                print(f"  syncing {model_key}/{section}/{wbid} ...")
                metrics, state = fetch_run(api, project, wbid)
                if not metrics:
                    continue

                for k, v in metrics.items():
                    if k == "hf":
                        if not run.get("hf"):
                            run["hf"] = v
                    else:
                        run[k] = v

                # mark as no longer pending if we got the key reasoning metrics
                if all(run.get(m) is not None for m in REASONING_COLS):
                    if run.get("pending"):
                        run["pending"] = False
                        print(f"    → resolved (state={state})")
                changed = True

    return changed


# ── HTML template ─────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<title>Pruning Results</title>
<style>
  :root {
    --bg:#0D0F18;--surface:#161926;--surface2:#1D2135;--border:#252840;
    --text:#DDE1F0;--muted:#5A607E;--accent:#1BC4B6;--accent-dim:rgba(27,196,182,0.13);
    --dense-row:rgba(255,255,255,0.03);--best-glow:rgba(27,196,182,0.18);
    --running:#F59E0B;--link-w:#F97316;--link-hf:#FBBF24;
    --badge-dense:#6B7280;--badge-sgpt:#D97706;--badge-gmp:#3B82F6;
    --badge-opkd:#8B5CF6;--badge-tr:#1BC4B6;--badge-tropkd:#10B981;
  }
  @media(prefers-color-scheme:light){:root{
    --bg:#F0F2FA;--surface:#FFFFFF;--surface2:#EEF0F8;--border:#D8DCF0;
    --text:#1A1D2E;--muted:#8A90AB;--accent:#0B9D96;--accent-dim:rgba(11,157,150,0.10);
    --dense-row:rgba(0,0,0,0.03);--best-glow:rgba(11,157,150,0.14);
    --link-w:#EA580C;--link-hf:#D97706;
  }}
  :root[data-theme="dark"]{--bg:#0D0F18;--surface:#161926;--surface2:#1D2135;--border:#252840;--text:#DDE1F0;--muted:#5A607E;--accent:#1BC4B6;--accent-dim:rgba(27,196,182,0.13);--dense-row:rgba(255,255,255,0.03);--best-glow:rgba(27,196,182,0.18);--link-w:#F97316;--link-hf:#FBBF24;}
  :root[data-theme="light"]{--bg:#F0F2FA;--surface:#FFFFFF;--surface2:#EEF0F8;--border:#D8DCF0;--text:#1A1D2E;--muted:#8A90AB;--accent:#0B9D96;--accent-dim:rgba(11,157,150,0.10);--dense-row:rgba(0,0,0,0.03);--best-glow:rgba(11,157,150,0.14);--link-w:#EA580C;--link-hf:#D97706;}

  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,'Helvetica Neue',Arial,sans-serif;font-size:13px;line-height:1.5;min-height:100vh}
  .page{max-width:1800px;margin:0 auto;padding:32px 24px 64px}

  .page-eyebrow{font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent);margin-bottom:8px;font-weight:600}
  .page-title{font-size:22px;font-weight:700;color:var(--text);letter-spacing:-.02em;margin-bottom:6px}
  .page-subtitle{color:var(--muted);font-size:12.5px;margin-bottom:20px}

  .tab-bar{display:flex;gap:0;margin-bottom:24px;border-bottom:1px solid var(--border)}
  .tab-btn{padding:8px 18px;font-size:12px;font-weight:600;color:var(--muted);border:none;background:none;cursor:pointer;border-bottom:2px solid transparent;margin-bottom:-1px;letter-spacing:.02em;transition:color .1s,border-color .1s;font-family:inherit}
  .tab-btn.active{color:var(--accent);border-bottom-color:var(--accent)}
  .tab-btn:hover:not(.active){color:var(--text)}

  .method-guide{margin-bottom:28px;display:flex;flex-direction:column;gap:6px}
  .mg-row{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
  .badge{display:inline-flex;align-items:center;gap:5px;padding:3px 9px;border-radius:4px;font-size:11px;font-weight:600;letter-spacing:.01em;border:1px solid;white-space:nowrap;flex-shrink:0}
  .badge-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
  .badge.dense{border-color:var(--badge-dense);color:var(--badge-dense)}.badge-dot.dense{background:var(--badge-dense)}
  .badge.sgpt{border-color:var(--badge-sgpt);color:var(--badge-sgpt)}.badge-dot.sgpt{background:var(--badge-sgpt)}
  .badge.gmp{border-color:var(--badge-gmp);color:var(--badge-gmp)}.badge-dot.gmp{background:var(--badge-gmp)}
  .badge.opkd{border-color:var(--badge-opkd);color:var(--badge-opkd)}.badge-dot.opkd{background:var(--badge-opkd)}
  .badge.tr{border-color:var(--badge-tr);color:var(--badge-tr)}.badge-dot.tr{background:var(--badge-tr)}
  .badge.tropkd{border-color:var(--badge-tropkd);color:var(--badge-tropkd)}.badge-dot.tropkd{background:var(--badge-tropkd)}
  .mg-desc{font-size:11.5px;color:var(--muted);line-height:1.4}

  .section{margin-bottom:36px}
  .section-label{display:flex;align-items:center;gap:12px;margin-bottom:10px}
  .section-label-tag{font-size:11px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--accent);background:var(--accent-dim);padding:3px 10px;border-radius:4px;border:1px solid var(--accent)}
  .section-label-sparsity{font-size:11px;color:var(--muted)}
  .section-label-line{flex:1;height:1px;background:var(--border)}

  .tbl-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:8px;background:var(--surface)}
  table{border-collapse:collapse;width:max-content;min-width:100%}

  .colgroup-header th{background:var(--surface2);color:var(--muted);font-size:9.5px;font-weight:700;letter-spacing:.09em;text-transform:uppercase;padding:5px 8px;border-bottom:1px solid var(--border);text-align:center;white-space:nowrap}
  .colgroup-header th:first-child{text-align:left}
  thead tr.col-headers th{background:var(--surface2);color:var(--muted);font-size:10px;font-weight:600;letter-spacing:.04em;padding:7px 8px 6px;border-bottom:2px solid var(--border);white-space:nowrap;text-align:right;position:sticky;top:0;z-index:2}
  thead tr.col-headers th:first-child{text-align:left;position:sticky;left:0;z-index:3}
  thead tr.colgroup-header th:first-child{position:sticky;left:0;z-index:3;background:var(--surface2)}

  tbody tr{border-bottom:1px solid var(--border);transition:background .08s}
  tbody tr:last-child{border-bottom:none}
  tbody tr:hover{background:rgba(128,140,180,.05)}
  tbody tr.dense-row{background:var(--dense-row)}
  tbody td{padding:7px 8px;white-space:nowrap;font-variant-numeric:tabular-nums;text-align:right;color:var(--text);font-size:12.5px}
  tbody td:first-child{text-align:left;position:sticky;left:0;background:inherit;z-index:1;min-width:220px;max-width:260px;border-right:1px solid var(--border);padding-right:12px}
  tbody tr:hover td:first-child{background:var(--surface)}
  tbody tr.dense-row td:first-child{background:var(--surface)}

  .csep{border-left:1px solid var(--border)!important}
  .method-cell{display:flex;align-items:center;gap:7px}
  .method-stripe{width:3px;height:20px;border-radius:2px;flex-shrink:0}
  .method-stripe.dense{background:var(--badge-dense)}.method-stripe.sgpt{background:var(--badge-sgpt)}
  .method-stripe.gmp{background:var(--badge-gmp)}.method-stripe.opkd{background:var(--badge-opkd)}
  .method-stripe.tr{background:var(--badge-tr)}.method-stripe.tropkd{background:var(--badge-tropkd)}
  .method-name{font-size:12px;font-weight:500;color:var(--text);overflow:hidden;text-overflow:ellipsis}
  .method-sub{font-size:10px;color:var(--muted);display:block;margin-top:1px}

  td.best{color:var(--accent)!important;font-weight:700;background:var(--best-glow);border-radius:3px}
  td.second{font-weight:600;opacity:.7;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px}
  td.ppl-col{color:var(--muted)}
  td.ppl-col.best-ppl{color:var(--accent)!important;font-weight:700;background:var(--best-glow)}
  td.ppl-col.second-ppl{font-weight:600;opacity:.7;text-decoration:underline;text-decoration-style:dotted;text-underline-offset:3px}
  td.running{color:var(--running);font-style:italic;font-size:11px}
  td.dash{color:var(--muted)}
  tr.failed-target td{opacity:.45}
  tr.failed-target td:first-child{opacity:1}
  .failed-badge{display:inline-block;font-size:9px;font-weight:700;color:#F87171;border:1px solid #F87171;border-radius:3px;padding:0 4px;margin-left:4px;vertical-align:middle;letter-spacing:.02em}
  td.step-col{color:var(--muted);font-size:11.5px}
  td.link-cell{text-align:center;padding:5px 6px}
  a.link-btn{display:inline-block;padding:2px 7px;border-radius:3px;font-size:10px;font-weight:700;letter-spacing:.03em;text-decoration:none;border:1px solid;transition:opacity .1s}
  a.link-btn:hover{opacity:.75}
  a.link-btn.wb{color:var(--link-w);border-color:var(--link-w)}
  a.link-btn.hf{color:var(--link-hf);border-color:var(--link-hf)}
  span.no-link{color:var(--muted);font-size:10px}

  .notes{margin-top:14px;display:flex;flex-direction:column;gap:5px}
  .note{font-size:11px;color:var(--muted);display:flex;align-items:center;gap:6px}
  .note-dot{width:6px;height:6px;border-radius:50%;background:var(--running);flex-shrink:0;animation:pulse 1.4s ease-in-out infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
  @media(prefers-reduced-motion:reduce){.note-dot{animation:none}}
</style>

<div class="page">
  <div class="page-eyebrow" id="page-eyebrow"></div>
  <div class="page-title">Pruning Results</div>
  <div class="page-subtitle" id="page-subtitle"></div>

  <div class="tab-bar" id="tab-bar"></div>

  <div class="method-guide">
    <div class="mg-row"><span class="badge dense"><span class="badge-dot dense"></span>Dense</span><span class="mg-desc">No pruning — baseline model</span></div>
    <div class="mg-row"><span class="badge sgpt"><span class="badge-dot sgpt"></span>SparseGPT</span><span class="mg-desc">One-shot Hessian-based unstructured pruning · calibration: OT 80% + FW 20% · no fine-tuning</span></div>
    <div class="mg-row"><span class="badge gmp"><span class="badge-dot gmp"></span>GMP NTP+KD</span><span class="mg-desc">Gradual Magnitude Pruning · NTP on CoT data + KD from dense teacher (λ_kd=1) · sparsity linearly scheduled over training</span></div>
    <div class="mg-row"><span class="badge opkd"><span class="badge-dot opkd"></span>GMP NTP+KD+OPKD</span><span class="mg-desc">GMP NTP+KD + On-Policy KD · student rollouts distilled against dense teacher · Prev Mask: previous-step sparse checkpoint (prev_mask_teacher=True)</span></div>
    <div class="mg-row"><span class="badge tr"><span class="badge-dot tr"></span>TR-GMP NTP+KD</span><span class="mg-desc">GMP NTP+KD with task regularization · pruning halts when KL(student‖teacher) > threshold · prevents over-pruning collapse</span></div>
    <div class="mg-row"><span class="badge tropkd"><span class="badge-dot tropkd"></span>TR-GMP+OPKD</span><span class="mg-desc">TR-GMP NTP+KD + On-Policy KD · Dual (prev_mask_teacher=False): original dense teacher · Prev Mask: previous-step sparse checkpoint</span></div>
  </div>

  <div id="app"></div>
  <div class="notes" id="notes"></div>
</div>

<script>
const DB = __DB_JSON__;
const HF0 = DB.meta.hf_base;

const MODEL_KEYS = Object.keys(DB.models);
let currentModel = MODEL_KEYS[0];

// ── tab bar ──────────────────────────────────────────────────────────────────
(function buildTabs() {
  const bar = document.getElementById('tab-bar');
  MODEL_KEYS.forEach((k, i) => {
    const btn = document.createElement('button');
    btn.className = 'tab-btn' + (i === 0 ? ' active' : '');
    btn.dataset.m = k;
    btn.textContent = DB.models[k].display;
    btn.onclick = () => switchModel(k);
    bar.appendChild(btn);
  });
})();

// ── table header ─────────────────────────────────────────────────────────────
const THEAD = `<thead>
  <tr class="colgroup-header">
    <th style="min-width:220px">Method</th>
    <th class="csep" colspan="4">Reasoning</th>
    <th class="csep" colspan="10">Zero-shot</th>
    <th class="csep" colspan="2">Perplexity ↓</th>
    <th class="csep">Steps</th>
    <th class="csep" colspan="2">Links</th>
  </tr>
  <tr class="col-headers">
    <th>Method</th>
    <th class="csep" data-higher>MATH-500</th><th data-higher>LCB</th><th data-higher>GPQA</th><th data-higher>IFEval</th>
    <th class="csep" data-higher>ZS Avg</th><th data-higher>Wino.</th><th data-higher>RTE</th><th data-higher>RACE</th><th data-higher>PIQA</th><th data-higher>OpenBook</th><th data-higher>HellaSwag</th><th data-higher>BoolQ</th><th data-higher>ARC-E</th><th data-higher>ARC-C</th>
    <th class="csep" data-lower>C4</th><th data-lower>WT2</th>
    <th class="csep">#step</th>
    <th class="csep">W&B</th><th>HF</th>
  </tr>
</thead>`;

function mcell(r) {
  const failBadge = r.failed_target
    ? `<span class="failed-badge">sp=${r.actual_sp}%</span>`
    : '';
  return `<div class="method-cell"><span class="method-stripe ${r.badge}"></span><div><span class="method-name">${r.name}${failBadge}</span><span class="method-sub">${r.sub||''}</span></div></div>`;
}

function makeRow(r, wbBase) {
  const tr = document.createElement('tr');
  if (r.isDense) tr.className = 'dense-row';
  else if (r.failed_target) tr.classList.add('failed-target');
  const wbLink = r.wbid
    ? `<td class="csep link-cell"><a class="link-btn wb" href="${wbBase}${r.wbid}" target="_blank">W&B</a></td>`
    : `<td class="csep link-cell"><span class="no-link">—</span></td>`;
  const hfHref = r.hf ? (r.hf.startsWith('http') ? r.hf : HF0 + r.hf) : null;
  const hfLink = hfHref
    ? `<td class="link-cell"><a class="link-btn hf" href="${hfHref}" target="_blank">HF</a></td>`
    : `<td class="link-cell"><span class="no-link">—</span></td>`;

  if (r.pending) {
    tr.innerHTML = `<td>${mcell(r)}</td>
      <td class="csep running">—</td><td class="running">—</td><td class="running">—</td><td class="running">—</td>
      <td class="csep dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td><td class="dash">—</td>
      <td class="csep ppl-col dash">—</td><td class="ppl-col dash">—</td>
      <td class="csep step-col dash">—</td>${wbLink}<td class="link-cell"><span class="no-link">—</span></td>`;
    return tr;
  }

  const f = v => (v === null || v === undefined) ? '<span class="dash">—</span>' : v;
  const lcbTd = r.lcbNull ? `<td class="dash">—</td>` : `<td>${f(r.lcb)}</td>`;
  const obVal = r.obNote ? `${f(r.ob)}${r.obNote}` : f(r.ob);
  const stepsTd = (r.steps == null)
    ? `<td class="csep step-col dash">—</td>`
    : `<td class="csep step-col">${r.steps}</td>`;

  const zsVals = [r.wino, r.rte, r.race, r.piqa, r.ob, r.hella, r.boolq, r.arce, r.arcc]
    .filter(v => v !== null && v !== undefined);
  const zsAvg = zsVals.length === 9
    ? (zsVals.reduce((a, b) => a + b, 0) / 9).toFixed(1)
    : null;

  tr.innerHTML = `<td>${mcell(r)}</td>
    <td class="csep">${f(r.math)}</td>${lcbTd}<td>${f(r.gpqa)}</td><td>${f(r.ifeval)}</td>
    <td class="csep">${f(zsAvg)}</td><td>${f(r.wino)}</td><td>${f(r.rte)}</td><td>${f(r.race)}</td><td>${f(r.piqa)}</td><td>${obVal}</td><td>${f(r.hella)}</td><td>${f(r.boolq)}</td><td>${f(r.arce)}</td><td>${f(r.arcc)}</td>
    <td class="csep ppl-col">${f(r.c4)}</td><td class="ppl-col">${f(r.wt2)}</td>
    ${stepsTd}${wbLink}${hfLink}`;
  return tr;
}

function highlightBest(id) {
  const tbl = document.getElementById(id);
  if (!tbl) return;
  const ths = Array.from(tbl.querySelector('tr.col-headers').querySelectorAll('th'));
  const rows = Array.from(tbl.querySelectorAll('tbody tr')).filter(r => !r.classList.contains('dense-row') && !r.classList.contains('failed-target'));
  ths.forEach((th, ci) => {
    const hi = th.hasAttribute('data-higher'), lo = th.hasAttribute('data-lower');
    if (!hi && !lo) return;
    const cells = rows.map(r => r.querySelectorAll('td')[ci]).filter(Boolean);
    const vals = cells.map(td => { const n = parseFloat(td.textContent); return isNaN(n) ? null : n; });
    const valid = vals.filter(v => v !== null);
    if (!valid.length) return;
    const sorted = [...new Set(valid)].sort((a, b) => hi ? b - a : a - b);
    const best = sorted[0], second = sorted[1];
    cells.forEach((td, i) => {
      if (vals[i] === best) td.classList.add(lo ? 'best-ppl' : 'best');
      else if (second !== undefined && vals[i] === second) td.classList.add(lo ? 'second-ppl' : 'second');
    });
  });
}

function renderSection(sec, model) {
  const runs = model[sec] || [];
  const wbBase = `https://wandb.ai/${DB.meta.entity}/${model.wb_project}/runs/`;
  const sparsity = {S50:'50%',S60:'60%',S70:'70%'}[sec];
  const id = 'tbl-' + sec.toLowerCase();
  const wrap = document.createElement('div');
  wrap.className = 'section';
  wrap.innerHTML = `
    <div class="section-label">
      <span class="section-label-tag">${sec.toLowerCase()}</span>
      <span class="section-label-sparsity">${sparsity} sparsity</span>
      <div class="section-label-line"></div>
    </div>
    <div class="tbl-wrap"><table id="${id}">${THEAD}<tbody></tbody></table></div>`;
  const tbody = wrap.querySelector('tbody');
  runs.forEach(r => tbody.appendChild(makeRow(r, wbBase)));
  return wrap;
}

function renderNotes(notesList) {
  const notesEl = document.getElementById('notes');
  notesEl.innerHTML = '';
  notesList.forEach(n => {
    const d = document.createElement('div');
    d.className = 'note';
    d.innerHTML = `<span class="note-dot"></span>${n}`;
    notesEl.appendChild(d);
  });
}

function switchModel(k) {
  currentModel = k;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.m === k));
  const model = DB.models[k];
  document.getElementById('page-eyebrow').textContent = `${model.display} · ${model.wb_project}`;
  document.getElementById('page-subtitle').textContent = 'A100-80GB · eval: MATH-500, LCB, GPQA, IFEval + 9 zero-shot tasks';
  const app = document.getElementById('app');
  app.innerHTML = '';
  ['S50','S60','S70'].forEach(s => {
    app.appendChild(renderSection(s, model));
    highlightBest('tbl-' + s.toLowerCase());
  });
  renderNotes(model.notes || []);
}

switchModel(MODEL_KEYS[0]);
</script>
"""


def render(db):
    db_json = json.dumps(db, ensure_ascii=False, separators=(',', ':'))
    return HTML_TEMPLATE.replace('__DB_JSON__', db_json)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync", action="store_true", help="Pull pending/missing metrics from wandb")
    parser.add_argument("--all",  action="store_true", help="With --sync: re-fetch all runs, not just pending")
    args = parser.parse_args()

    with open(DB_PATH) as f:
        db = json.load(f)

    if args.sync:
        print("Syncing with wandb...")
        changed = sync_db(db, sync_all=args.all)
        if changed:
            with open(DB_PATH, "w") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)
            print(f"Updated {DB_PATH}")

    html = render(db)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"Written {OUT_PATH} ({len(html)//1024}KB)")


if __name__ == "__main__":
    main()
