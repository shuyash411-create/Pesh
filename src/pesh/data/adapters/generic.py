"""Generic per-step agent logs -> run logs (and feasibility training frames).

Input: one record per model call (JSONL/CSV/Parquet) with at least
    task_id, run_id, step, input_tokens, output_tokens
and optionally
    cache_read_tokens, cache_write_tokens, cost, repeated_views, success, model_version,
    plus any pre-execution feature columns (constant within a run).

If `cost` is absent it is computed from token counts with Eq. (4) prices:
    cost = p_in*uncached + w*p_in*cache_write + delta*p_in*cache_read + p_out*output
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...config import PricingParams
from ...schema import validate_runs, validate_steps


def load_steps(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return validate_steps(df)


def step_costs(steps: pd.DataFrame, pricing: PricingParams | None = None) -> pd.Series:
    pr = pricing or PricingParams()
    read = steps.get("cache_read_tokens", pd.Series(0.0, index=steps.index)).fillna(0.0)
    write = steps.get("cache_write_tokens", pd.Series(0.0, index=steps.index)).fillna(0.0)
    uncached = (steps["input_tokens"] - read - write).clip(lower=0)
    computed = (pr.p_in * uncached + pr.w * pr.p_in * write + pr.delta * pr.p_in * read
                + pr.p_out * steps["output_tokens"])
    if "cost" in steps:
        return steps["cost"].fillna(computed)
    return computed


def enrich_steps(steps: pd.DataFrame, pricing: PricingParams | None = None, ewma: float = 0.3) -> pd.DataFrame:
    """Add step_cost, cumulative spent, context, last_step_cost and repeat_ewma signals."""
    s = validate_steps(steps).copy()
    s["step_cost"] = step_costs(s, pricing)
    g = s.groupby(["task_id", "run_id"], sort=False)
    s["spent"] = g["step_cost"].cumsum()
    s["context"] = s["input_tokens"]
    s["last_step_cost"] = s["step_cost"]
    views = s["repeated_views"] if "repeated_views" in s else pd.Series(0.0, index=s.index)
    s["repeat_ewma"] = views.groupby([s["task_id"], s["run_id"]]).transform(
        lambda v: v.ewm(alpha=ewma, adjust=False).mean())
    if "degraded" not in s:
        s["degraded"] = False
    return s


def runs_from_steps(steps: pd.DataFrame, outcomes: pd.DataFrame | None = None,
                    pricing: PricingParams | None = None, feature_cols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate step logs into the generic run-log schema.

    outcomes: optional frame (task_id, run_id, success[, capped, cap]) if success is not in steps.
    """
    s = enrich_steps(steps, pricing)
    agg = {"step": "max", "input_tokens": "sum", "output_tokens": "sum", "step_cost": "sum"}
    for c in ("success", "capped", "cap", "model_version"):
        if c in s:
            agg[c] = "last"
    for c in feature_cols or []:
        agg[c] = "first"
    runs = s.groupby(["task_id", "run_id"], sort=False).agg(agg).reset_index()
    runs = runs.rename(columns={"step": "steps", "step_cost": "cost"})
    if outcomes is not None:
        runs = runs.drop(columns=[c for c in outcomes.columns if c in runs and c not in ("task_id", "run_id")])
        runs = runs.merge(outcomes, on=["task_id", "run_id"], how="left")
    runs["total_tokens"] = runs["input_tokens"] + runs["output_tokens"]
    return validate_runs(runs)


def feasibility_frame(steps: pd.DataFrame, runs: pd.DataFrame, seed: int = 0,
                      budget_quantiles=(0.70, 0.99)) -> pd.DataFrame:
    """Step states labelled for FeasibilityModel.fit.

    Uncapped logs get a random ceiling per run (as in simulation); capped runs keep their cap.
    """
    s = enrich_steps(steps) if "spent" not in steps else steps
    r = runs[["task_id", "run_id", "cost", "success"] + (["cap"] if "cap" in runs else [])].rename(
        columns={"cost": "final_cost"})
    lo, hi = np.log(np.quantile(runs["cost"], budget_quantiles))
    rng = np.random.default_rng(seed)
    rand_b = np.exp(rng.uniform(lo, hi, len(r)))
    cap = r["cap"].to_numpy(float) if "cap" in r else np.full(len(r), np.inf)
    r = r.assign(budget=np.where(np.isfinite(cap), cap, rand_b))
    out = s.drop(columns=[c for c in ("success",) if c in s]).merge(r, on=["task_id", "run_id"], how="inner")
    return out[out["spent"] <= out["budget"]].reset_index(drop=True)
