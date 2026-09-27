"""Censored quantile regression (Proposition 5, hypothesis H5).

Once ceilings are enforced the operator observes C~ = min(C, B). Naive quantile
regression on log C~ is biased toward the cap. Powell (1986):

    beta_P = argmin_b sum rho_tau( log C~_i - min(x_i'b, log B_i) )

computed by Buchinsky's (1994) iterative linear programming: alternate between the
active set {i : x_i'b < log B_i} and ordinary QR on that set until it stabilises.
Chernozhukov & Hong (2002) three-step is provided as a more stable alternative.
"""

from __future__ import annotations

import numpy as np

from .quantreg import check_loss, fit_quantile


def powell_objective(X, y, log_cap, beta, tau) -> float:
    return float(np.sum(check_loss(y - np.minimum(X @ beta, log_cap), tau)))


def powell_cqr(X: np.ndarray, y: np.ndarray, log_cap: np.ndarray | float, tau: float,
               max_iter: int = 50, beta0: np.ndarray | None = None) -> dict:
    """Powell censored QR via iterative LP. y = log of capped outcome; log_cap = log B (per row ok)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    log_cap = np.broadcast_to(np.asarray(log_cap, float), y.shape)
    beta = fit_quantile(X, y, tau) if beta0 is None else np.asarray(beta0, float)
    best_beta, best_obj = beta, powell_objective(X, y, log_cap, beta, tau)
    prev_active = None
    it = 0
    for it in range(1, max_iter + 1):
        active = X @ beta < log_cap
        if active.sum() <= X.shape[1]:
            break
        if prev_active is not None and np.array_equal(active, prev_active):
            break
        beta = fit_quantile(X[active], y[active], tau)
        obj = powell_objective(X, y, log_cap, beta, tau)
        if obj < best_obj:
            best_beta, best_obj = beta, obj
        prev_active = active
    return {"beta": best_beta, "objective": best_obj, "iterations": it}


def chernozhukov_hong(X: np.ndarray, y: np.ndarray, censored: np.ndarray, log_cap: np.ndarray | float,
                      tau: float, c: float = 0.05) -> dict:
    """Three-step censored QR (Chernozhukov & Hong, 2002)."""
    from sklearn.linear_model import LogisticRegression

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    log_cap = np.broadcast_to(np.asarray(log_cap, float), y.shape)
    uncens = ~np.asarray(censored, bool)
    # step 1: probability of being uncensored
    clf = LogisticRegression(max_iter=1000).fit(X[:, 1:], uncens)
    p = clf.predict_proba(X[:, 1:])[:, 1]
    J0 = p > 1.0 - tau + c
    # step 2: QR on J0
    b1 = fit_quantile(X[J0], y[J0], tau)
    # step 3: QR on {x'b1 > log B ... } i.e. rows whose predicted quantile is below cap
    J1 = X @ b1 < log_cap
    b2 = fit_quantile(X[J1], y[J1], tau)
    return {"beta": b2, "beta_step2": b1, "n_step2": int(J0.sum()), "n_step3": int(J1.sum())}


def naive_vs_powell(X, y_capped, log_cap, tau) -> dict:
    naive = fit_quantile(X, y_capped, tau)
    powell = powell_cqr(X, y_capped, log_cap, tau, beta0=naive)
    return {"naive": naive, "powell": powell["beta"], "iterations": powell["iterations"]}
