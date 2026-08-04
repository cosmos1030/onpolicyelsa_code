"""DB schema and helpers. results_db.json is the source of truth."""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "results_db.json")

METRIC_FIELDS = ["math", "lcb", "gpqa", "ifeval",
                 "wino", "rte", "race", "piqa", "ob", "hella", "boolq", "arce", "arcc",
                 "c4", "wt2"]

TIERS = ["S50", "S60", "S70", "N24"]
MODELS = ["deepseek_1.5b", "qwen3_1.7b", "qwen3_4b", "qwen3_8b"]

ENTITY = "dyk6208-gwangju-institute-of-science-and-technology"
WANDB_PROJECTS = {
    "reasoning_qwen3_4b": "qwen3_4b",
    "reasoning_pruning_v1": None,   # multi-model
    "reasoning_pruning_v2": None,
    "elsa_qwen3_0.6b": None,
}


@dataclass
class RunEntry:
    """One row in the results table."""
    badge: str
    name: str
    sub: str = ""
    wbid: Optional[str] = None          # eval wandb run id
    train_wbid: Optional[str] = None    # training wandb run id (if different)
    hf: Optional[str] = None
    steps: Optional[int] = None
    isDense: bool = False
    # metrics
    math: Optional[float] = None
    lcb: Optional[float] = None
    gpqa: Optional[float] = None
    ifeval: Optional[float] = None
    wino: Optional[float] = None
    rte: Optional[float] = None
    race: Optional[float] = None
    piqa: Optional[float] = None
    ob: Optional[float] = None
    hella: Optional[float] = None
    boolq: Optional[float] = None
    arce: Optional[float] = None
    arcc: Optional[float] = None
    c4: Optional[float] = None
    wt2: Optional[float] = None
    # internal tracking
    pending: bool = False               # True = metrics not yet fetched
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None    # run name for lookup (when wbid unknown)

    def missing_metrics(self) -> list[str]:
        return [f for f in METRIC_FIELDS if getattr(self, f) is None]

    def is_complete(self) -> bool:
        return not self.pending and len(self.missing_metrics()) == 0

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if v is not None and v is not False}
        # always keep isDense if True
        if self.isDense:
            d["isDense"] = True
        # remove internal fields from artifact output
        for internal in ("pending", "wandb_project", "wandb_name", "train_wbid"):
            d.pop(internal, None)
        return d


@dataclass
class ModelEntry:
    display: str
    wb_project: str = "reasoning_qwen3_4b"
    notes: list[str] = field(default_factory=list)
    S50: list[RunEntry] = field(default_factory=list)
    S60: list[RunEntry] = field(default_factory=list)
    S70: list[RunEntry] = field(default_factory=list)
    N24: list[RunEntry] = field(default_factory=list)


class DB:
    """In-memory view of results_db.json with load/save/sync helpers."""

    def __init__(self, path: str = DB_PATH):
        self.path = path
        self.meta: dict = {}
        self.models: dict[str, ModelEntry] = {}
        self._raw: dict = {}

    @classmethod
    def load(cls, path: str = DB_PATH) -> "DB":
        db = cls(path)
        with open(path) as f:
            raw = json.load(f)
        db._raw = raw
        db.meta = raw.get("meta", {})
        for model_key, model_data in raw.get("models", {}).items():
            entries: dict[str, list[RunEntry]] = {}
            for tier in TIERS:
                entries[tier] = []
                for row in model_data.get(tier, []):
                    e = RunEntry(
                        badge=row["badge"],
                        name=row["name"],
                        sub=row.get("sub", ""),
                        wbid=row.get("wbid"),
                        train_wbid=row.get("train_wbid"),
                        hf=row.get("hf"),
                        steps=row.get("steps"),
                        isDense=row.get("isDense", False),
                        pending=row.get("pending", False),
                        wandb_project=row.get("wandb_project"),
                        wandb_name=row.get("wandb_name"),
                    )
                    for mf in METRIC_FIELDS:
                        if mf in row:
                            setattr(e, mf, row[mf])
                    entries[tier].append(e)
            db.models[model_key] = ModelEntry(
                display=model_data["display"],
                wb_project=model_data.get("wb_project", "reasoning_qwen3_4b"),
                notes=model_data.get("notes", []),
                S50=entries["S50"],
                S60=entries["S60"],
                S70=entries["S70"],
                N24=entries["N24"],
            )
        return db

    def save(self):
        data = {
            "meta": self.meta,
            "models": {},
        }
        for model_key, model in self.models.items():
            data["models"][model_key] = {
                "display": model.display,
                "wb_project": model.wb_project,
                "notes": model.notes,
            }
            for tier in TIERS:
                rows = getattr(model, tier)
                tier_list = []
                for e in rows:
                    row_dict = e.to_dict()
                    # keep pending/wandb_project/wandb_name in the JSON file even if omitted from artifact
                    if e.pending:
                        row_dict["pending"] = True
                    if e.wandb_project:
                        row_dict["wandb_project"] = e.wandb_project
                    if e.wandb_name:
                        row_dict["wandb_name"] = e.wandb_name
                    if e.train_wbid:
                        row_dict["train_wbid"] = e.train_wbid
                    tier_list.append(row_dict)
                data["models"][model_key][tier] = tier_list
        # preserve any unknown top-level keys from original
        for k, v in self._raw.items():
            if k not in ("meta", "models"):
                data[k] = v
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def pending_entries(self) -> list[tuple[str, str, RunEntry]]:
        """Return (model_key, tier, entry) for all entries with pending=True or missing metrics."""
        result = []
        for model_key, model in self.models.items():
            for tier in TIERS:
                for entry in getattr(model, tier):
                    if entry.pending or (entry.wbid and entry.missing_metrics()):
                        result.append((model_key, tier, entry))
        return result

    def for_artifact(self) -> dict:
        """Return dict suitable for JSON injection into artifact HTML."""
        data = {"meta": self.meta, "models": {}}
        for model_key, model in self.models.items():
            data["models"][model_key] = {
                "display": model.display,
                "wb_project": model.wb_project,
                "notes": model.notes,
            }
            for tier in TIERS:
                data["models"][model_key][tier] = [
                    e.to_dict() for e in getattr(model, tier)
                ]
        return data
