"""Pre-execution feature design.

Only information available *before* a run may enter a quote (Proposition 3: anything
generated before the run carries no information about its own future randomness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd


@dataclass
class DesignBuilder:
    numeric: list[str]
    categorical: list[str] = field(default_factory=list)
    quadratic: bool = False
    levels_: dict[str, list] = field(default_factory=dict)
    ranges_: dict[str, tuple[float, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "DesignBuilder":
        for c in self.categorical:
            self.levels_[c] = sorted(pd.unique(df[c].astype(str)))
        for c in self.numeric:
            v = df[c].astype(float)
            self.ranges_[c] = (float(v.quantile(0.005)), float(v.quantile(0.995)))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = [np.ones(len(df))]
        num = [df[c].astype(float).to_numpy() for c in self.numeric]
        cols += num
        if self.quadratic:
            cols += [v**2 for v in num]
            cols += [a * b for a, b in combinations(num, 2)]
        for c in self.categorical:
            vals = df[c].astype(str).to_numpy()
            for lvl in self.levels_[c][1:]:
                cols.append((vals == lvl).astype(float))
        return np.column_stack(cols)

    def names(self) -> list[str]:
        out = ["const"] + list(self.numeric)
        if self.quadratic:
            out += [f"{c}^2" for c in self.numeric]
            out += [f"{a}*{b}" for a, b in combinations(self.numeric, 2)]
        for c in self.categorical:
            out += [f"{c}={lvl}" for lvl in self.levels_[c][1:]]
        return out

    def out_of_range(self, df: pd.DataFrame) -> np.ndarray:
        """Rows whose numeric features fall outside the central 99% of training data."""
        flag = np.zeros(len(df), bool)
        for c, (lo, hi) in self.ranges_.items():
            v = df[c].astype(float).to_numpy()
            flag |= (v < lo) | (v > hi)
        return flag


def add_log_features(df: pd.DataFrame, cols: list[str], prefix: str = "log_") -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[prefix + c] = np.log1p(out[c].astype(float).clip(lower=0))
    return out


def add_task_history(df: pd.DataFrame, target: str = "cost", key: str = "task_id") -> pd.DataFrame:
    """Leave-one-out mean log cost of *other* runs of the same task (a 'history' feature).

    Legitimate pre-execution information when a task (or template) has been run before.
    Rows with no other runs get the global mean and hist_n = 0.
    """
    out = df.copy()
    y = np.log(out[target].astype(float))
    g = y.groupby(out[key])
    s, n = g.transform("sum"), g.transform("count")
    loo = (s - y) / (n - 1).replace(0, np.nan)
    out["hist_log_cost"] = loo.fillna(y.mean())
    out["hist_n"] = (n - 1).astype(int)
    return out
