"""Linear quantile regression (Koenker & Bassett, 1978).

Exact solution by linear programming (HiGHS) for moderate n; statsmodels' IRLS
QuantReg for large n. Also a gradient-boosted quantile model for non-linear signal.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog


def check_loss(u: np.ndarray, tau: float) -> np.ndarray:
    """rho_tau(u) = u (tau - 1{u < 0})."""
    u = np.asarray(u, float)
    return u * (tau - (u < 0))


def pinball(y, q, tau: float) -> float:
    return float(np.mean(check_loss(np.asarray(y) - np.asarray(q), tau)))


def _qr_lp(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    n, p = X.shape
    # variables: beta+ (p), beta- (p), u+ (n), u- (n)
    c = np.concatenate([np.zeros(2 * p), np.full(n, tau), np.full(n, 1.0 - tau)])
    Xs = sp.csr_matrix(X)
    eye = sp.identity(n, format="csr")
    A = sp.hstack([Xs, -Xs, eye, -eye], format="csr")
    res = linprog(c, A_eq=A, b_eq=y, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"quantile LP failed: {res.message}")
    return res.x[:p] - res.x[p:2 * p]


def _qr_irls(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    from statsmodels.regression.quantile_regression import QuantReg

    return np.asarray(QuantReg(y, X).fit(q=tau, max_iter=5000, p_tol=1e-8).params)


def fit_quantile(X: np.ndarray, y: np.ndarray, tau: float, method: str = "auto") -> np.ndarray:
    """Return beta_tau minimising sum rho_tau(y - X beta). X must include an intercept column."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if method == "auto":
        method = "lp" if len(y) <= 25_000 else "irls"
    return _qr_lp(X, y, tau) if method == "lp" else _qr_irls(X, y, tau)


class LinearQuantile:
    """Linear conditional quantile Q_tau(y|x) = x'beta."""

    def __init__(self, tau: float, method: str = "auto"):
        self.tau = tau
        self.method = method
        self.beta_: np.ndarray | None = None

    def fit(self, X, y):
        self.beta_ = fit_quantile(X, y, self.tau, self.method)
        return self

    def predict(self, X):
        return np.asarray(X, float) @ self.beta_


class GBMQuantile:
    """Gradient-boosted conditional quantile (non-linear upgrade path)."""

    def __init__(self, tau: float, **kw):
        from sklearn.ensemble import HistGradientBoostingRegressor

        self.tau = tau
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=tau,
                                                   max_iter=kw.pop("max_iter", 300),
                                                   learning_rate=kw.pop("learning_rate", 0.05), **kw)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def ols(X, y):
    beta, *_ = np.linalg.lstsq(np.asarray(X, float), np.asarray(y, float), rcond=None)
    return beta
