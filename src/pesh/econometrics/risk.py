"""Risk measures, pooling (Proposition 6) and the buyer's value of a ceiling (Eq. 12)."""

from __future__ import annotations

import numpy as np


def var(x, level: float = 0.95) -> float:
    return float(np.quantile(np.asarray(x, float), level))


def cvar(x, level: float = 0.95) -> float:
    """Expected shortfall: mean of the worst (1 - level) share (Rockafellar & Uryasev, 2000)."""
    x = np.sort(np.asarray(x, float))
    k = max(1, int(np.ceil(round((1 - level) * len(x), 9))))
    return float(x[-k:].mean())


def mean_cvar_disutility(spend, eta: float, level: float = 0.95) -> float:
    s = np.asarray(spend, float)
    m = s.mean()
    return float(m + eta * (cvar(s, level) - m))


def willingness_to_pay(spend_uncapped, spend_capped, success_uncapped: float, success_capped: float,
                       n_tasks: int, eta: float, value_per_success: float, level: float = 0.95) -> dict:
    """Eq. (12): expected-cost saving + risk-premium saving - quality cost."""
    s0 = np.asarray(spend_uncapped, float)
    s1 = np.asarray(spend_capped, float)
    exp_saving = s0.mean() - s1.mean()
    risk_saving = eta * ((cvar(s0, level) - s0.mean()) - (cvar(s1, level) - s1.mean()))
    quality_cost = value_per_success * n_tasks * (success_uncapped - success_capped)
    wtp = exp_saving + risk_saving - quality_cost
    return {"wtp": float(wtp), "expected_saving": float(exp_saving), "risk_saving": float(risk_saving),
            "quality_cost": float(quality_cost), "wtp_share_of_spend": float(wtp / s0.mean())}


def pooling_loading(costs: np.ndarray, sizes, eps: float = 0.01, reps: int = 2000, seed: int = 0,
                    shock_mult: float | None = None, shock_prob: float = 0.0,
                    expected: float | None = None) -> dict:
    """Relative loading (q_{1-eps}[mean_N] - E C) / E C for portfolios of size N.

    With shock_mult, each portfolio's costs are multiplied by shock_mult with prob shock_prob
    (a common provider-side shock; Prop. 6 iv). `expected` defaults to the unconditional mean
    including the shock.
    """
    rng = np.random.default_rng(seed)
    costs = np.asarray(costs, float)
    ec = costs.mean() if expected is None else expected
    if shock_mult is not None and expected is None:
        ec = costs.mean() * (1 + shock_prob * (shock_mult - 1))
    out = {}
    for N in sizes:
        means = np.empty(reps)
        for r0 in range(0, reps, 200):
            r1 = min(reps, r0 + 200)
            idx = rng.integers(0, len(costs), size=(r1 - r0, N))
            means[r0:r1] = costs[idx].mean(axis=1)
        if shock_mult is not None:
            means *= np.where(rng.random(reps) < shock_prob, shock_mult, 1.0)
        out[int(N)] = float((np.quantile(means, 1 - eps) - ec) / ec)
    return out


def loglog_slope(loading: dict) -> float:
    n = np.log(np.array(list(loading.keys()), float))
    v = np.log(np.maximum(np.array(list(loading.values()), float), 1e-12))
    return float(np.polyfit(n, v, 1)[0])


def gaussian_loading(sigma: float, n: int, eps: float = 0.01) -> float:
    """Prop. 6(i): l_N ~ z_{1-eps} sigma / sqrt(N) (absolute)."""
    from scipy.stats import norm

    return float(norm.ppf(1 - eps) * sigma / np.sqrt(n))
