"""Generic run-log schema.

One row per agent run. Anything that is known before execution (features) may be
added as extra columns; the quote engine is told which ones to use.

Required:  task_id, run_id, cost (USD, realised or capped), success (bool)
Optional:  capped (bool; cost is a lower bound on the uncapped cost), cap (USD ceiling in force),
           steps, input_tokens, output_tokens, model_version, exploration (bool)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["task_id", "run_id", "cost", "success"]
DEFAULTS = {
    "capped": False,
    "cap": np.inf,
    "steps": np.nan,
    "input_tokens": np.nan,
    "output_tokens": np.nan,
    "model_version": "unknown",
    "exploration": False,
}

STEP_REQUIRED = ["task_id", "run_id", "step", "input_tokens", "output_tokens"]


class SchemaError(ValueError):
    pass


def validate_runs(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"run log missing required columns: {missing}")
    out = df.copy()
    for col, default in DEFAULTS.items():
        if col not in out.columns:
            out[col] = default
    out["cost"] = pd.to_numeric(out["cost"], errors="raise").astype(float)
    if (out["cost"] <= 0).any():
        raise SchemaError("cost must be strictly positive (log-cost models)")
    out["success"] = out["success"].astype(bool)
    out["capped"] = out["capped"].fillna(False).astype(bool)
    out["cap"] = pd.to_numeric(out["cap"]).fillna(np.inf).astype(float)
    bad = out["capped"] & ~np.isfinite(out["cap"])
    if bad.any():
        raise SchemaError(f"{int(bad.sum())} capped rows have no finite cap")
    if (out["cost"] > out["cap"] * (1 + 1e-9)).any():
        raise SchemaError("cost exceeds the cap in force for some rows")
    out["model_version"] = out["model_version"].astype(str)
    return out


def validate_steps(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STEP_REQUIRED if c not in df.columns]
    if missing:
        raise SchemaError(f"step log missing required columns: {missing}")
    return df.sort_values(["task_id", "run_id", "step"]).reset_index(drop=True)
