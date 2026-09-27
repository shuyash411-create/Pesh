"""Offline A/B of execution policies on common random numbers (hypothesis H6)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ControllerConfig, DGPParams, PricingParams
from ..econometrics.risk import cvar
from ..sim.dgp import execute


def policy_metrics(runs: pd.DataFrame, B: float | None = None) -> dict:
    c = runs["cost"].to_numpy(float)
    s = runs["success"].to_numpy(bool)
    out = {
        "mean_cost": float(c.mean()),
        "cvar95": cvar(c, 0.95),
        "success": float(s.mean()),
        "cost_per_success": float(c.sum() / max(s.sum(), 1)),
        "capped_share": float(runs["capped"].mean()),
        "stopped_share": float(runs["stopped_early"].mean()),
        "degraded_share": float(runs["degraded"].mean()),
    }
    if B is not None:
        out["p_exceed_B"] = float((c > B + 1e-12).mean())
    return out


def compare_policies(plan, policies: dict, params: DGPParams | None = None,
                     pricing: PricingParams | None = None, config: ControllerConfig | None = None,
                     B: float | None = None) -> pd.DataFrame:
    cfg = config or ControllerConfig()
    rows = {}
    for name, pol in policies.items():
        res = execute(plan, pol, params, pricing, degrade_price_mult=cfg.degrade_price_mult,
                      compact_to=cfg.compact_to, degrade_extra_fail=cfg.degrade_extra_fail)
        rows[name] = policy_metrics(res.runs, B)
    df = pd.DataFrame(rows).T
    base = df.iloc[0]
    df["mean_cost_change"] = df["mean_cost"] / base["mean_cost"] - 1
    df["cvar95_change"] = df["cvar95"] / base["cvar95"] - 1
    df["success_change_pp"] = (df["success"] - base["success"]) * 100
    return df


def replay_logged_runs(steps: pd.DataFrame, controller, budget: float) -> pd.DataFrame:
    """Replay recorded step logs (task_id, run_id, step, step_cost, repeat_ewma, context) under a
    controller, without degradation (logged trajectories cannot be re-generated after a model
    switch). Returns per-run cost and whether the run was stopped or capped before its logged end.
    """
    out = []
    for (task, run), g in steps.sort_values("step").groupby(["task_id", "run_id"], sort=False):
        spent, stopped, capped = 0.0, False, False
        last = 0.0
        for row in g.itertuples(index=False):
            if spent + row.step_cost > budget:
                capped = True
                break
            spent += row.step_cost
            last = row.step_cost
            d = controller.decide_one(int(row.step), spent, float(getattr(row, "repeat_ewma", 0.0)),
                                      float(getattr(row, "context", 0.0)), last, False, budget)
            if d == "stop" and row.step < g["step"].max():
                stopped = True
                break
        out.append({"task_id": task, "run_id": run, "cost": spent, "stopped_early": stopped,
                    "capped": capped, "logged_cost": float(g["step_cost"].sum())})
    return pd.DataFrame(out)


def ceiling_from_quantile(costs: np.ndarray, q: float = 0.9) -> float:
    return float(np.quantile(np.asarray(costs, float), q))
