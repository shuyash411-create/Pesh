"""Predictability ceiling (Proposition 3, hypothesis H1).

One-way random effects  Y_ir = nu + u_i + e_ir  on Y = log cost.
rho_tau = sigma_u^2 / (sigma_u^2 + sigma_e^2) bounds the R^2 of *any* pre-execution
predictor, and sqrt(rho_tau) bounds its correlation with realised log cost.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class ICCResult:
    rho: float
    sigma2_between: float
    sigma2_within: float
    corr_ceiling: float
    n_tasks: int
    n_runs: int
    method: str

    def to_dict(self):
        return asdict(self)


def icc_anova(y: np.ndarray, groups: np.ndarray) -> ICCResult:
    """ANOVA (method-of-moments) estimator for unbalanced one-way random effects."""
    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    stats = df.groupby("g")["y"].agg(["mean", "count", "var"])
    stats = stats[stats["count"] >= 1]
    N, k = int(stats["count"].sum()), len(stats)
    grand = df["y"].mean()
    ssb = float((stats["count"] * (stats["mean"] - grand) ** 2).sum())
    ssw = float(((stats["count"] - 1) * stats["var"].fillna(0.0)).sum())
    msb = ssb / max(k - 1, 1)
    msw = ssw / max(N - k, 1)
    n0 = (N - (stats["count"] ** 2).sum() / N) / max(k - 1, 1)
    s2u = max((msb - msw) / n0, 0.0)
    rho = s2u / (s2u + msw) if s2u + msw > 0 else 0.0
    return ICCResult(rho, s2u, msw, float(np.sqrt(rho)), k, N, "anova")


def icc_reml(y: np.ndarray, groups: np.ndarray) -> ICCResult:
    """REML via statsmodels MixedLM (random intercept)."""
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"y": np.asarray(y, float), "g": np.asarray(groups)})
    fit = smf.mixedlm("y ~ 1", df, groups=df["g"]).fit(reml=True)
    s2u = float(fit.cov_re.iloc[0, 0])
    s2e = float(fit.scale)
    rho = s2u / (s2u + s2e)
    return ICCResult(rho, s2u, s2e, float(np.sqrt(rho)), df["g"].nunique(), len(df), "reml")


def predictability_report(y, groups, predictions=None, method: str = "anova") -> dict:
    """ICC ceiling plus, optionally, where a predictor sits relative to it.

    Returns information gap (rho - R2_f) and irreducible noise (1 - rho).
    """
    res = icc_anova(y, groups) if method == "anova" else icc_reml(y, groups)
    out = res.to_dict()
    out["irreducible_noise"] = 1.0 - res.rho
    if predictions is not None:
        y = np.asarray(y, float)
        f = np.asarray(predictions, float)
        r2 = 1.0 - np.mean((y - f) ** 2) / np.var(y)
        out["r2_predictor"] = float(r2)
        out["corr_predictor"] = float(np.corrcoef(y, f)[0, 1])
        out["information_gap"] = float(res.rho - r2)
    return out
