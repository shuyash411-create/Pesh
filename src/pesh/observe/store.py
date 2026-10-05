"""Observation store: the `observations` table in the existing SQLite file (see service/state.py)."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

import pandas as pd

COLUMNS = [
    "task_id", "kind", "run_idx", "model", "provider", "status", "error",
    "predicted_cost", "predicted_ceiling",
    "actual_cost", "actual_input_tokens", "actual_output_tokens", "wall_ms", "steps", "success",
    "within_ceiling", "abs_error", "pct_error", "timestamp", "features",
]
_SCHEMA = """CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL, kind TEXT NOT NULL, run_idx INTEGER NOT NULL,
    model TEXT NOT NULL, provider TEXT, status TEXT NOT NULL, error TEXT,
    predicted_cost REAL, predicted_ceiling REAL,
    actual_cost REAL, actual_input_tokens INTEGER, actual_output_tokens INTEGER,
    wall_ms REAL, steps INTEGER, success INTEGER,
    within_ceiling INTEGER, abs_error REAL, pct_error REAL,
    timestamp TEXT, features TEXT,
    UNIQUE (task_id, model, kind, run_idx))"""


class ObservationStore:
    """Append-only log of (prediction, actual) pairs. Safe to share between threads."""

    def __init__(self, db_path: str | Path = "pesh_dashboard.sqlite"):
        self.db_path = str(db_path)
        self.lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._con() as con:
            con.execute(_SCHEMA)

    def _con(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def record(self, obs: dict) -> None:
        """Insert one observation. A repeat of the same (task, model, kind, run_idx) replaces it,
        which is how a previously errored run is retried."""
        row = {c: obs.get(c) for c in COLUMNS}
        if isinstance(row["features"], dict):
            row["features"] = json.dumps(row["features"])
        for c in ("success", "within_ceiling"):
            if row[c] is not None:
                row[c] = int(bool(row[c]))
        with self.lock, self._con() as con:
            con.execute(f"INSERT OR REPLACE INTO observations ({','.join(COLUMNS)}) "
                        f"VALUES ({','.join('?' * len(COLUMNS))})", [row[c] for c in COLUMNS])

    def done_keys(self) -> set[tuple]:
        """(task_id, model, kind, run_idx) of runs that completed; errored runs are retried."""
        with self._con() as con:
            rows = con.execute("SELECT task_id, model, kind, run_idx FROM observations "
                               "WHERE status='ok'").fetchall()
        return {tuple(r) for r in rows}

    def count(self, status: str | None = None) -> int:
        with self._con() as con:
            q, a = "SELECT COUNT(*) FROM observations", ()
            if status:
                q, a = q + " WHERE status=?", (status,)
            return con.execute(q, a).fetchone()[0]

    def frame(self, task_ids: list[str] | None = None) -> pd.DataFrame:
        with self._con() as con:
            df = pd.read_sql_query("SELECT * FROM observations ORDER BY id", con)
        if task_ids is not None:
            df = df[df["task_id"].isin(task_ids)]
        for c in ("success", "within_ceiling"):
            df[c] = df[c].map(lambda v: None if pd.isna(v) else bool(v)).astype(object)
        return df.reset_index(drop=True)

    def export(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.frame()
        if path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path
