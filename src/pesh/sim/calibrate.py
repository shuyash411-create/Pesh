"""Simulated method of moments (SMM) calibration of the DGP to published moments.

The paper hand-tunes the DGP by coordinate descent (Appendix B). Here the same targets
(Table 4, Bai et al. 2026) are matched by minimising a weighted squared relative-error
distance with Nelder-Mead, holding simulation draws fixed across evaluations (common
random numbers) so the objective is deterministic.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import kendalltau

from ..config import DGPParams, PricingParams
from .dgp import simulate

# Published targets (paper Table 4) plus the paper's simulated ICC as a target for rho_tau.
TARGETS = {
    "mean_tokens_m": 4.03,
    "mean_cost": 2.07,
    "in_out_ratio": 148.0,
    "success": 0.651,
    "kendall_tau": 0.33,
    "maxmin_mean": 4.1,
    "icc": 0.80,
}
WEIGHTS = {"mean_tokens_m": 1.0, "mean_cost": 1.0, "in_out_ratio": 0.5, "success": 2.0,
           "kendall_tau": 1.0, "maxmin_mean": 0.5, "icc": 2.0}

FREE = ["mu_T", "lam", "sigma_u", "label_noise", "s0", "q0", "scale_L", "a_log_mean", "o_log_mean", "sigma_G"]


def moments(df: pd.DataFrame) -> dict:
    g = df.groupby("task_id")
    logc = np.log(df["cost"].clip(lower=1e-9))
    within = logc.groupby(df["task_id"]).var().mean()
    ratio = g["cost"].max() / g["cost"].min()
    return {
        "mean_tokens_m": df["total_tokens"].mean() / 1e6,
        "mean_cost": df["cost"].mean(),
        "in_out_ratio": df["input_tokens"].sum() / df["output_tokens"].sum(),
        "success": df["success"].mean(),
        "kendall_tau": kendalltau(g["human_label"].first(), g["total_tokens"].mean()).statistic,
        "maxmin_mean": ratio.mean(),
        "maxmin_p99": ratio.quantile(0.99),
        "icc": 1.0 - within / logc.var(),
    }


def distance(m: dict, targets: dict = TARGETS, weights: dict = WEIGHTS) -> float:
    return float(sum(weights[k] * ((m[k] - v) / v) ** 2 for k, v in targets.items()))


def calibrate(start: DGPParams | None = None, n_tasks: int = 1500, runs: int = 4, seed: int = 7,
              maxiter: int = 300, pricing: PricingParams | None = None, verbose: bool = False):
    start = start or DGPParams()
    x0 = np.array([getattr(start, k) for k in FREE], float)

    def to_params(x):
        vals = dict(zip(FREE, x))
        vals["label_noise"] = abs(vals["label_noise"])
        vals["scale_L"] = abs(vals["scale_L"])
        vals["sigma_G"] = abs(vals["sigma_G"])
        vals["sigma_u"] = abs(vals["sigma_u"])
        return replace(start, **vals)

    def obj(x):
        df = simulate(n_tasks, runs, to_params(x), pricing, seed=seed)
        val = distance(moments(df))
        if verbose:
            print(f"{val:.4f}", np.round(x, 3))
        return val

    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4, "adaptive": True})
    best = to_params(res.x)
    final = moments(simulate(n_tasks * 2, runs, best, pricing, seed=seed + 1))
    return best, final, res
