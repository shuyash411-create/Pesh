"""Adaptive conformal inference (Gibbs & Candes, 2021), Eq. (8).

alpha_{t+1} = alpha_t + gamma (alpha - 1{C_t > q_{alpha_t}(X_t)})

The offset used at time t is the (1 - alpha_t) conformal quantile of a rolling window of
recent scores, so both the level and the score distribution track drift. Long-run average
miscoverage converges to alpha for any outcome sequence.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .conformal import conformal_offset


class ACI:
    def __init__(self, alpha: float = 0.10, gamma: float = 0.01, window: int = 1000,
                 init_scores: np.ndarray | None = None):
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_t = alpha
        self.scores: deque = deque(maxlen=window)
        if init_scores is not None:
            self.scores.extend(np.asarray(init_scores, float)[-window:])
        self.n_updates = 0
        self.n_miss = 0

    def offset(self) -> float:
        if not self.scores:
            return float("inf")
        s = np.fromiter(self.scores, float)
        if self.alpha_t <= 0:
            return float(s.max())          # most conservative finite quote
        if self.alpha_t >= 1:
            return float(s.min())
        off = conformal_offset(s, self.alpha_t)
        return float(s.max()) if not np.isfinite(off) else off

    def update(self, score: float, offset_used: float | None = None) -> bool:
        """Record a realised score (log C - Q^(x)); returns whether the quote was missed."""
        off = self.offset() if offset_used is None else offset_used
        miss = bool(score > off)
        self.alpha_t += self.gamma * (self.alpha - float(miss))
        self.scores.append(float(score))
        self.n_updates += 1
        self.n_miss += int(miss)
        return miss

    def reset(self, alpha: float | None = None):
        self.alpha_t = self.alpha if alpha is None else alpha
        self.scores.clear()
        self.n_updates = self.n_miss = 0

    @property
    def realised_coverage(self) -> float:
        return 1.0 - self.n_miss / self.n_updates if self.n_updates else float("nan")

    def state(self) -> dict:
        return {"alpha": self.alpha, "gamma": self.gamma, "alpha_t": self.alpha_t,
                "n_scores": len(self.scores), "n_updates": self.n_updates,
                "realised_coverage": self.realised_coverage if self.n_updates else None}
