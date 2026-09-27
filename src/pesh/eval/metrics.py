"""Evaluation metrics for quotes and controllers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coverage(cost, quote) -> float:
    return float(np.mean(np.asarray(cost, float) <= np.asarray(quote, float)))


def coverage_by(cost, quote, strata, labels=None) -> dict:
    cost = np.asarray(cost, float)
    quote = np.asarray(quote, float)
    strata = np.asarray(strata)
    keys = labels if labels is not None else sorted(pd.unique(strata))
    return {str(k): coverage(cost[strata == k], quote[strata == k]) for k in keys}


def terciles(x, names=("easy", "mid", "hard")) -> np.ndarray:
    cuts = np.quantile(np.asarray(x, float), [1 / 3, 2 / 3])
    return np.asarray(names, dtype=object)[np.digitize(x, cuts)]


def r2(y, f) -> float:
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    return float(1.0 - np.mean((y - f) ** 2) / np.var(y))


def quote_summary(name: str, cost, quote, difficulty, labels=None, loop=None) -> dict:
    out = {"method": name, "marginal": coverage(cost, quote)}
    out.update(coverage_by(cost, quote, terciles(difficulty), ["easy", "mid", "hard"]))
    if labels is not None:
        for k, v in coverage_by(cost, quote, labels).items():
            out[f"label_{k}"] = v
    if loop is not None:
        loop = np.asarray(loop, bool)
        out["loop_runs"] = coverage(np.asarray(cost)[loop], np.asarray(quote)[loop])
        out["clean_runs"] = coverage(np.asarray(cost)[~loop], np.asarray(quote)[~loop])
    out["quote_over_mean_cost"] = float(np.mean(quote) / np.mean(cost))
    out["median_quote_over_median_cost"] = float(np.median(quote) / np.median(cost))
    return out
