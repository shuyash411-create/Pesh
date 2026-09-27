"""Split conformalised quantile regression for one-sided cost quotes (Proposition 4).

Scores s_j = log C_j - Q^_{1-alpha}(X_j) on a calibration set; with
k = ceil((n+1)(1-alpha)), the quote q_alpha(x) = exp(Q^(x) + s_(k)) satisfies
1 - alpha <= Pr{C <= q} <= 1 - alpha + 1/(n+1) under exchangeability.
Mondrian: the same construction within each observable group g.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_k(n: int, alpha: float) -> int:
    return int(math.ceil((n + 1) * (1 - alpha)))


def conformal_offset(scores: np.ndarray, alpha: float) -> float:
    """s_(k); +inf if the calibration set is too small for level 1-alpha."""
    s = np.sort(np.asarray(scores, float))
    k = conformal_k(len(s), alpha)
    return float(s[k - 1]) if k <= len(s) else float("inf")


def theoretical_coverage(n: int, alpha: float) -> float:
    """Exact expected coverage ceil((n+1)(1-alpha))/(n+1) for continuous scores."""
    return min(1.0, conformal_k(n, alpha) / (n + 1))


class SplitConformal:
    """One-sided split conformal wrapper around any fitted quantile predictor on log cost."""

    def __init__(self, alpha: float = 0.10):
        self.alpha = alpha
        self.scores_: dict = {}
        self.offsets_: dict = {}
        self.global_offset_: float | None = None
        self.n_cal_: dict = {}

    def calibrate(self, pred_log: np.ndarray, y_log: np.ndarray, groups: np.ndarray | None = None,
                  min_group_n: int = 1):
        scores = np.asarray(y_log, float) - np.asarray(pred_log, float)
        self.scores_ = {"__all__": scores}
        self.global_offset_ = conformal_offset(scores, self.alpha)
        self.n_cal_ = {"__all__": len(scores)}
        self.offsets_ = {}
        if groups is not None:
            groups = np.asarray(groups).astype(str)
            for g in np.unique(groups):
                s = scores[groups == g]
                self.scores_[g] = s
                self.n_cal_[g] = len(s)
                if len(s) >= min_group_n:
                    self.offsets_[g] = conformal_offset(s, self.alpha)
        return self

    def offset(self, groups: np.ndarray | None, n: int, alpha: float | None = None) -> np.ndarray:
        alpha = self.alpha if alpha is None else alpha
        if alpha == self.alpha:
            glob, per = self.global_offset_, self.offsets_
        else:
            glob = conformal_offset(self.scores_["__all__"], alpha)
            per = {g: conformal_offset(s, alpha) for g, s in self.scores_.items()
                   if g != "__all__" and g in self.offsets_}
        if groups is None or not per:
            return np.full(n, glob)
        groups = np.asarray(groups).astype(str)
        return np.array([per.get(g, glob) for g in groups])

    def group_n(self, groups: np.ndarray | None, n: int) -> np.ndarray:
        if groups is None:
            return np.full(n, self.n_cal_["__all__"])
        return np.array([self.n_cal_.get(str(g), 0) for g in groups])
