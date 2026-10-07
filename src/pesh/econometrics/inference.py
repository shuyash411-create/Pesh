"""Confidence intervals for H1 (predictability ceiling) and H2 (tail indices).

Repeated runs of one task are correlated, so every interval here is a *cluster* bootstrap:
tasks are resampled with replacement and all of a resampled task's runs come along.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .icc import icc_anova
from .tails import hill, powerlaw_fit


def _group_index(groups) -> tuple[np.ndarray, list[np.ndarray]]:
    codes, uniques = pd.factorize(pd.Series(groups), sort=False)
    order = np.argsort(codes, kind="stable")
    bounds = np.cumsum(np.bincount(codes, minlength=len(uniques)))[:-1]
    return codes, np.split(order, bounds)


def _icc_from_stats(n: np.ndarray, s: np.ndarray, ss: np.ndarray) -> np.ndarray:
    """Vectorised ANOVA ICC; rows are bootstrap replicates, columns resampled groups."""
    N = n.sum(axis=1)
    k = n.shape[1]
    grand = s.sum(axis=1) / N
    means = s / n
    ssb = (n * (means - grand[:, None]) ** 2).sum(axis=1)
    ssw = (ss - s**2 / n).sum(axis=1)
    msb = ssb / max(k - 1, 1)
    msw = ssw / np.maximum(N - k, 1)
    n0 = (N - (n**2).sum(axis=1) / N) / max(k - 1, 1)
    s2u = np.maximum((msb - msw) / n0, 0.0)
    tot = s2u + msw
    return np.where(tot > 0, s2u / np.where(tot > 0, tot, 1), 0.0)


def icc_ci(y, groups, n_boot: int = 1000, level: float = 0.95, seed: int = 0) -> dict:
    """rho_tau with a percentile cluster-bootstrap interval (tasks resampled)."""
    y = np.asarray(y, float)
    point = icc_anova(y, groups)
    codes, members = _group_index(groups)
    n = np.bincount(codes).astype(float)
    s = np.bincount(codes, weights=y)
    ss = np.bincount(codes, weights=y**2)
    keep = n >= 2                       # single-run tasks carry no within-task information
    n, s, ss = n[keep], s[keep], ss[keep]
    rng = np.random.default_rng(seed)
    k = len(n)
    reps = []
    for b0 in range(0, n_boot, 200):
        idx = rng.integers(0, k, size=(min(200, n_boot - b0), k))
        reps.append(_icc_from_stats(n[idx], s[idx], ss[idx]))
    reps = np.concatenate(reps)
    a = (1 - level) / 2
    lo, hi = np.quantile(reps, [a, 1 - a])
    return {"rho": point.rho, "rho_lo": float(lo), "rho_hi": float(hi),
            "corr_ceiling": point.corr_ceiling, "corr_ceiling_lo": float(np.sqrt(lo)),
            "corr_ceiling_hi": float(np.sqrt(hi)), "n_tasks": int(k), "n_runs": int(n.sum()),
            "runs_per_task": float(n.mean()), "n_boot": n_boot}


def _k(n: int, fraction: float) -> int:
    return max(10, int(fraction * n))


def drop_cap_mass(x: np.ndarray, min_share: float = 0.002) -> tuple[np.ndarray, float]:
    """Remove runs sitting at a hard cap (e.g. an agent's max-iterations limit).

    Many ties at the maximum mean the variable was truncated there, so the Hill estimator
    is undefined at that point; the tail is estimated on the runs below the cap instead.
    Returns (values kept, share of runs at the cap).
    """
    x = np.asarray(x, float)
    at_max = x == x.max()
    share = float(at_max.mean())
    if at_max.sum() > 1 and share >= min_share:
        return x[~at_max], share
    return x, 0.0


def _hill_safe(x: np.ndarray, k: int) -> float:
    with np.errstate(divide="ignore"):
        return hill(x, k)


def tail_ci(cost, steps=None, groups=None, fraction: float = 0.02, n_boot: int = 500, level: float = 0.95,
            seed: int = 0, sensitivity=(0.01, 0.02, 0.05)) -> dict:
    """Hill tail indices of cost (and steps) with cluster-bootstrap intervals and the ratio alpha_C/alpha_T.

    `sensitivity` reports the point estimate at other tail fractions, because Hill estimates
    move with the number of order statistics used.
    """
    cost = np.asarray(cost, float)
    n = len(cost)
    out = {"fraction": fraction, "k": _k(n, fraction), "n": n,
           "alpha_cost": hill(cost, _k(n, fraction)),
           "alpha_cost_by_fraction": {f: hill(cost, _k(n, f)) for f in sensitivity if _k(n, f) < n}}
    st = None if steps is None else np.asarray(steps, float)
    if st is not None:
        cap = st.max()
        kept, out["steps_at_cap_share"] = drop_cap_mass(st)
        m = len(kept)
        out["alpha_steps"] = _hill_safe(kept, _k(m, fraction))
        out["alpha_steps_by_fraction"] = {f: _hill_safe(kept, _k(m, f)) for f in sensitivity if _k(m, f) < m}
        out["ratio"] = out["alpha_cost"] / out["alpha_steps"]
    if groups is None:
        groups = np.arange(n)
    _, members = _group_index(groups)
    rng = np.random.default_rng(seed)
    reps_c, reps_s = [], []
    for _ in range(n_boot):
        pick = rng.integers(0, len(members), len(members))
        idx = np.concatenate([members[g] for g in pick])
        kk = _k(len(idx), fraction)
        if kk >= len(idx):
            continue
        c_rep = _hill_safe(cost[idx], kk)
        s_rep = np.nan
        if st is not None:
            s_vals = st[idx][st[idx] < cap] if out["steps_at_cap_share"] else st[idx]
            ks = _k(len(s_vals), fraction)
            s_rep = _hill_safe(s_vals, ks) if ks < len(s_vals) else np.nan
        reps_c.append(c_rep)
        reps_s.append(s_rep)
    a = (1 - level) / 2
    rc, rs = np.array(reps_c), np.array(reps_s)
    fin = np.isfinite(rc)
    out["alpha_cost_lo"], out["alpha_cost_hi"] = map(float, np.quantile(rc[fin], [a, 1 - a]))
    if st is not None:
        ok = np.isfinite(rs) & (rs > 0)
        out["alpha_steps_lo"], out["alpha_steps_hi"] = map(float, np.quantile(rs[ok], [a, 1 - a]))
        ratio = (rc / np.where(ok, rs, np.nan))[ok & fin]
        out["ratio_lo"], out["ratio_hi"] = map(float, np.quantile(ratio, [a, 1 - a]))
    out["variance_verdict"] = ("infinite" if out["alpha_cost_hi"] < 2 else
                               "finite" if out["alpha_cost_lo"] > 2 else "inconclusive")
    out["powerlaw"] = powerlaw_fit(cost)
    return out


def h1h2_report(runs: pd.DataFrame, by: str | None = "model_version", n_boot: int = 1000,
                fraction: float = 0.02, seed: int = 0) -> pd.DataFrame:
    """One row per agent configuration (and an 'all' row): H1 and H2 with 95% intervals."""
    groups = [("all", runs)]
    if by and by in runs and runs[by].nunique() > 1:
        groups = [(str(k), g) for k, g in runs.groupby(by, sort=True)] + groups
    rows = []
    for name, g in groups:
        row = {"group": name, "n_runs": len(g), "n_tasks": g["task_id"].nunique(),
               "mean_cost": float(g["cost"].mean()), "median_cost": float(g["cost"].median()),
               "success": float(g["success"].mean())}
        counts = g.groupby("task_id").size()
        if (counts >= 2).sum() >= 10:
            row.update(icc_ci(np.log(g["cost"]), g["task_id"], n_boot=n_boot, seed=seed))
        has_steps = "steps" in g and g["steps"].notna().all()
        if len(g) >= 200:
            t = tail_ci(g["cost"], g["steps"] if has_steps else None, g["task_id"], fraction=fraction,
                        n_boot=max(200, n_boot // 2), seed=seed)
            row.update({k: v for k, v in t.items() if not isinstance(v, dict)})
            row["powerlaw_alpha"] = t["powerlaw"].get("alpha")
            row["powerlaw_ks"] = t["powerlaw"].get("ks")
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")
