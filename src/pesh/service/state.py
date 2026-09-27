"""Run state and persistence for the quote/enforcement service."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class RunRecord:
    run_id: str
    task_id: str
    model_version: str
    features: dict
    quote: float
    base_log: float
    budget: float
    exploration: bool
    spent: float = 0.0
    steps: int = 0
    context: float = 0.0
    last_step_cost: float = 0.0
    repeat_ewma: float = 0.0
    degraded: bool = False
    status: str = "running"          # running | stopped | capped | finished
    success: bool | None = None
    closed: bool = False             # outcome reported via /finish
    decisions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class RunStore:
    """In-memory run table with optional SQLite persistence of finished runs."""

    def __init__(self, db_path: str | None = None, monitor_window: int = 500):
        self.runs: dict[str, RunRecord] = {}
        self.lock = threading.Lock()
        self.recent: dict[str, deque] = {}
        self.window = monitor_window
        self.db_path = db_path
        if db_path:
            with sqlite3.connect(db_path) as con:
                con.execute("""CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY, task_id TEXT, model_version TEXT, features TEXT,
                    quote REAL, cap REAL, cost REAL, capped INTEGER, exploration INTEGER,
                    success INTEGER, steps INTEGER, degraded INTEGER, status TEXT)""")

    def new_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def add(self, rec: RunRecord):
        with self.lock:
            self.runs[rec.run_id] = rec

    def get(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def record_outcome(self, rec: RunRecord, covered: bool):
        self.recent.setdefault(rec.model_version, deque(maxlen=self.window)).append(bool(covered))
        if self.db_path:
            with sqlite3.connect(self.db_path) as con:
                con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    rec.run_id, rec.task_id, rec.model_version, json.dumps(rec.features), rec.quote,
                    rec.budget, rec.spent, int(rec.status == "capped"), int(rec.exploration),
                    None if rec.success is None else int(rec.success), rec.steps, int(rec.degraded), rec.status))

    def rolling_coverage(self, model_version: str) -> tuple[float, int]:
        d = self.recent.get(model_version)
        if not d:
            return float("nan"), 0
        return float(np.mean(d)), len(d)

    def export_runs(self):
        """Finished runs in the generic run-log schema (for retraining)."""
        import pandas as pd

        rows = []
        for r in self.runs.values():
            if not r.closed:
                continue
            # capped and controller-stopped runs are right-censored: true cost >= what was spent
            censored = r.status in ("capped", "stopped")
            cost = max(r.spent, 1e-9)
            cap = max(r.budget, cost) if r.status == "capped" else (cost if censored else float("inf"))
            rows.append({"task_id": r.task_id, "run_id": r.run_id, "cost": cost,
                         "success": bool(r.success), "capped": censored, "cap": cap,
                         "stopped_early": r.status == "stopped",
                         "steps": r.steps, "model_version": r.model_version, "exploration": r.exploration,
                         **r.features})
        return pd.DataFrame(rows)
