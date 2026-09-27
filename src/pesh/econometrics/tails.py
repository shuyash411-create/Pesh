"""Heavy tails (Proposition 2, hypothesis H2).

Hill estimator over a grid of upper order statistics and a Clauset-Shalizi-Newman
power-law fit (xmin chosen by minimum KS distance). Indices are reported on the
survival scale: Pr(X > x) ~ x^{-alpha}.
"""

from __future__ import annotations

import numpy as np


def hill(x: np.ndarray, k: int) -> float:
    """Hill estimate of the tail index alpha using the k largest observations."""
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    n = len(x)
    if not 1 <= k < n:
        raise ValueError("need 1 <= k < n")
    top = x[n - k:]
    threshold = x[n - k - 1]
    return float(1.0 / np.mean(np.log(top / threshold)))


def hill_curve(x: np.ndarray, fractions=(0.005, 0.01, 0.02, 0.03, 0.05)) -> dict[float, float]:
    n = len(x)
    return {f: hill(x, max(10, int(f * n))) for f in fractions}


def hill_range(x: np.ndarray, fractions=(0.01, 0.02)) -> tuple[float, float]:
    vals = list(hill_curve(x, fractions).values())
    return float(min(vals)), float(max(vals))


def powerlaw_fit(x: np.ndarray, n_candidates: int = 60, min_tail: int = 50) -> dict:
    """Continuous power-law MLE with xmin selected by KS distance (Clauset et al., 2009)."""
    x = np.sort(np.asarray(x, float))
    x = x[x > 0]
    cands = np.unique(np.quantile(x, np.linspace(0.5, 1 - min_tail / len(x), n_candidates)))
    best = None
    for xmin in cands:
        tail = x[x >= xmin]
        m = len(tail)
        if m < min_tail:
            continue
        a = 1.0 + m / np.sum(np.log(tail / xmin))       # density exponent
        emp = np.arange(m) / m
        theo = 1.0 - (tail / xmin) ** (1.0 - a)
        ks = float(np.max(np.abs(emp - theo)))
        if best is None or ks < best["ks"]:
            best = {"xmin": float(xmin), "alpha_density": float(a), "alpha": float(a - 1.0), "ks": ks, "n_tail": m}
    return best or {}


def tail_ratio(cost: np.ndarray, steps: np.ndarray, fraction: float = 0.01) -> dict:
    """alpha_C / alpha_T; Proposition 2 predicts ~0.5 while context accumulates freely."""
    k_c = max(10, int(fraction * len(cost)))
    k_t = max(10, int(fraction * len(steps)))
    a_c = hill(cost, k_c)
    a_t = hill(steps, k_t)
    return {"alpha_cost": a_c, "alpha_steps": a_t, "ratio": a_c / a_t}
