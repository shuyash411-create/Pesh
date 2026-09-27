"""Mid-run feasibility model pi^(sigma_t) = Pr(success | state after t steps).

BAGEN (Lin et al., 2026) finds that *feasibility* is learnable mid-run while remaining
*amounts* are not; the controller therefore needs a calibrated pi^ and only a coarse
estimate of remaining cost (Section 4.6).

Target: success *within the remaining budget* (success and final cost <= B), which is the
paper's pi(sigma_t). Features are observable trajectory signals only: step index, spend to
date, spend as a share of B, context size, the last step's cost, and an EWMA of repeated file
views / edits (loop signal), and whether the run has been degraded. Training is on-policy: step
logs come from runs executed under a hard cap with and without degradation, with a random
ceiling per run, so one model serves any B and every policy that uses it.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = ["step", "spent", "repeat_ewma", "context", "last_step_cost", "budget", "degraded"]


def _design(step, spent, repeat_ewma, context, last_step_cost, budget, degraded) -> np.ndarray:
    step = np.asarray(step, float)
    frac = np.clip(np.asarray(spent, float) / np.asarray(budget, float), 0.0, 1.0)
    deg = np.broadcast_to(np.asarray(degraded, float), step.shape)
    return np.column_stack([
        deg,
        deg * frac,
        frac,
        frac**2,
        np.log(np.asarray(budget, float)),
        np.log1p(step),
        np.log1p(np.asarray(spent, float) * 100),
        np.asarray(repeat_ewma, float),
        np.log1p(np.asarray(context, float)),
        np.log1p(np.asarray(last_step_cost, float) * 100),
        np.asarray(repeat_ewma, float) * np.log1p(step),
    ])


class FeasibilityModel:
    def __init__(self, C: float = 1.0):
        self.clf = LogisticRegression(C=C, max_iter=2000)
        self.metrics: dict = {}

    def fit(self, steps: pd.DataFrame, label: str = "feasible") -> "FeasibilityModel":
        """steps needs FEATURES plus `label` (default: success & final_cost <= budget)."""
        if label == "feasible" and label not in steps:
            steps = steps.assign(feasible=steps["success"] & (steps["final_cost"] <= steps["budget"]))
        X = _design(*(steps[c] for c in FEATURES))
        y = steps[label].astype(int).to_numpy()
        self.clf.fit(X, y)
        p = self.clf.predict_proba(X)[:, 1]
        self.metrics = {"n_steps": int(len(y)), "base_rate": float(y.mean()),
                        "auc_in_sample": float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")}
        return self

    def predict(self, step, spent, repeat_ewma, context, last_step_cost, budget, degraded=False) -> np.ndarray:
        X = _design(step, spent, repeat_ewma, context, last_step_cost, budget, degraded)
        return self.clf.predict_proba(X)[:, 1]

    def predict_state(self, state) -> np.ndarray:
        return self.predict(state.t, state.spent, state.repeat_ewma, state.context, state.last_step_cost,
                            state.budget, state.degraded)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "FeasibilityModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def train_from_simulation(n_tasks: int = 3000, runs: int = 2, seed: int = 101, params=None, pricing=None,
                          max_runs_logged: int = 6000, config=None) -> FeasibilityModel:
    """Train pi^ on-policy from simulated step logs.

    Each run gets a random ceiling between the 70th and 99th percentile of uncontrolled cost
    and is executed twice on common random numbers: under a hard cap, and under cap + degrade.
    """
    from ..config import ControllerConfig
    from ..sim.dgp import execute, plan_runs, simulate_tasks
    from .controller import CostController

    cfg = config or ControllerConfig()
    tasks = simulate_tasks(n_tasks, params, seed)
    plan = plan_runs(tasks, runs, params, seed)
    free = execute(plan, None, params, pricing).runs["cost"]
    lo, hi = np.log(np.quantile(free, [0.70, 0.99]))
    B = np.exp(np.random.default_rng(seed).uniform(lo, hi, plan.n))
    frames = []
    for degrade in (False, True):
        res = execute(plan, CostController(B, None, degrade, cfg), params, pricing,
                      degrade_price_mult=cfg.degrade_price_mult, compact_to=cfg.compact_to,
                      degrade_extra_fail=cfg.degrade_extra_fail, record_steps=max_runs_logged)
        frames.append(res.steps.assign(budget=B[res.steps["row"].to_numpy()]))
    return FeasibilityModel().fit(pd.concat(frames, ignore_index=True))
