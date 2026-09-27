"""Loading, saving and leakage-free splitting of run logs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import validate_runs


def load_runs(path: str | Path, validate: bool = True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix in (".csv", ".txt"):
        df = pd.read_csv(path)
    elif path.suffix in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError(f"unsupported file type: {path.suffix}")
    return validate_runs(df) if validate else df


def save_runs(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def split_by_task(df: pd.DataFrame, fractions=(0.5, 0.25, 0.25), seed: int = 0,
                  key: str = "task_id") -> list[pd.DataFrame]:
    """Split so that all runs of a task land in the same part (no task leakage across splits)."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("fractions must sum to 1")
    tasks = pd.unique(df[key])
    rng = np.random.default_rng(seed)
    tasks = tasks[rng.permutation(len(tasks))]
    cuts = (np.cumsum(fractions)[:-1] * len(tasks)).astype(int)
    parts = np.split(tasks, cuts)
    return [df[df[key].isin(set(p))].reset_index(drop=True) for p in parts]


def one_run_per_task(df: pd.DataFrame, seed: int = 0, key: str = "task_id") -> pd.DataFrame:
    """Keep one random run per task (i.i.d. sample for conformal calibration)."""
    return (df.sample(frac=1.0, random_state=seed).drop_duplicates(key)
            .sort_values(key).reset_index(drop=True))
