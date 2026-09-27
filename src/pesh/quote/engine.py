"""QuoteEngine: pre-execution distributional quotes q_alpha(X) (assurance level L1).

Pipeline (paper Figure 1):
  1. base conditional quantile Q^_{1-alpha}(log C | x) on a training split
     - linear QR, or Powell censored QR when the logs contain capped runs (Prop. 5)
     - optional gradient boosting (uncensored data only)
  2. split-conformal offset on a calibration split, optionally Mondrian by group (Prop. 4)
  3. online ACI state per model version for drift (Eq. 8)
  4. a quantile grid for E[min(C, B) | x], used by L3 pricing
"""

from __future__ import annotations

import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import QuoteConfig
from ..econometrics.censored import powell_cqr
from ..econometrics.quantreg import GBMQuantile, LinearQuantile, fit_quantile, pinball
from ..features import DesignBuilder
from .aci import ACI
from .conformal import SplitConformal

GRID = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])
_BIG_LOG = np.log(1e9)


def _log_caps(df: pd.DataFrame) -> np.ndarray:
    cap = df["cap"].to_numpy(float) if "cap" in df else np.full(len(df), np.inf)
    return np.where(np.isfinite(cap), np.log(np.where(np.isfinite(cap), cap, 1.0)), _BIG_LOG)


class QuoteEngine:
    def __init__(self, config: QuoteConfig | None = None, model: str = "linear"):
        self.config = config or QuoteConfig()
        self.model = model
        self.design: DesignBuilder | None = None
        self.base = None
        self.grid_betas: dict[float, np.ndarray] = {}
        self.conformal: SplitConformal | None = None
        self.aci: dict[str, ACI] = {}
        self.card: dict = {}

    # ------------------------------------------------------------------ fitting
    def _groups(self, df: pd.DataFrame):
        g = self.config.mondrian_by
        return None if g is None else df[g].astype(str).to_numpy()

    def fit(self, train: pd.DataFrame, calib: pd.DataFrame) -> "QuoteEngine":
        cfg = self.config
        tau = 1.0 - cfg.alpha
        self.design = DesignBuilder(cfg.numeric_features, cfg.categorical_features, cfg.quadratic).fit(train)
        X = self.design.transform(train)
        y = np.log(train["cost"].to_numpy(float))
        capped = train["capped"].to_numpy(bool) if "capped" in train else np.zeros(len(train), bool)
        censored = bool(capped.any())
        log_cap = _log_caps(train)

        if censored:
            # Powell censored QR: only consistent estimator of the upper quantile on capped logs.
            beta = powell_cqr(X, y, log_cap, tau)["beta"]
            self.base = LinearQuantile(tau)
            self.base.beta_ = beta
            method = "powell-cqr"
            if self.model != "linear":
                method += " (gbm disabled: censored data)"
        elif self.model == "gbm":
            self.base = GBMQuantile(tau).fit(X, y)
            method = "gbm-quantile"
        else:
            self.base = LinearQuantile(tau).fit(X, y)
            method = "linear-qr"

        # quantile grid for E[min(C,B)|x]
        self.grid_betas = {}
        for t in GRID:
            if censored:
                self.grid_betas[float(t)] = powell_cqr(X, y, log_cap, t)["beta"]
            else:
                self.grid_betas[float(t)] = fit_quantile(X, y, t)

        # conformal calibration; capped calibration rows are conservative misses (+inf score)
        Xc = self.design.transform(calib)
        yc = np.log(calib["cost"].to_numpy(float))
        pred = self.base.predict(Xc)
        c_capped = calib["capped"].to_numpy(bool) if "capped" in calib else np.zeros(len(calib), bool)
        yc_eff = np.where(c_capped, np.inf, yc)
        self.conformal = SplitConformal(cfg.alpha).calibrate(pred, yc_eff, self._groups(calib), cfg.min_group_n)
        scores = yc_eff - pred
        self.aci = {"default": ACI(cfg.alpha, cfg.aci_gamma, cfg.aci_window, scores[np.isfinite(scores)])}

        # Prop. 5: where the fitted quantile reaches the cap it is not identified from capped logs
        unident = float(np.mean(self.base.predict(X) >= log_cap)) if censored else 0.0
        self.card = {
            "method": method,
            "unidentified_share": unident,
            "alpha": cfg.alpha,
            "n_train": int(len(train)),
            "n_calib": int(len(calib)),
            "censored_train_share": float(capped.mean()),
            "censored_calib_share": float(c_capped.mean()),
            "features": self.design.names(),
            "conformal_offset": self.conformal.global_offset_,
            "group_offsets": dict(self.conformal.offsets_),
            "config": asdict(cfg),
        }
        return self

    # ------------------------------------------------------------------ quoting
    def base_log(self, df: pd.DataFrame) -> np.ndarray:
        return self.base.predict(self.design.transform(df))

    def quantile_log(self, df: pd.DataFrame, tau: float) -> np.ndarray:
        """Interpolated conditional quantile of log cost from the grid (not conformalised)."""
        X = self.design.transform(df)
        taus = np.array(sorted(self.grid_betas))
        preds = np.column_stack([X @ self.grid_betas[t] for t in taus])
        preds = np.maximum.accumulate(preds, axis=1)   # rearrangement: monotone in tau
        return np.array([np.interp(tau, taus, row) for row in preds])

    def quote(self, df: pd.DataFrame, alpha: float | None = None, online: bool = False,
              model_version: str = "default") -> pd.DataFrame:
        """q_alpha(x) for each row: the (1-alpha) conformal cost quote in USD."""
        base = self.base_log(df)
        n = len(df)
        groups = self._groups(df) if self.config.mondrian_by in df.columns else None
        if online:
            off = np.full(n, self.aci_for(model_version).offset())
            method = "aci"
        else:
            off = self.conformal.offset(groups, n, alpha)
            method = "mondrian-cqr" if groups is not None and self.conformal.offsets_ else "split-cqr"
        p50 = np.exp(self.quantile_log(df, 0.5))
        q = np.exp(base + off)
        calib_n = self.conformal.group_n(groups, n)
        warn = np.full(n, "", dtype=object)
        ood = self.design.out_of_range(df)
        warn[ood] = "features outside training range; conditional coverage unreliable"
        small = calib_n < self.config.min_group_n
        warn[small & ~ood] = "small calibration set; coverage only loosely guaranteed"
        return pd.DataFrame({
            "p50": p50,
            "quote": q,
            "log_quote": base + off,
            "coverage_level": 1.0 - (self.config.alpha if alpha is None else alpha),
            "method": method,
            "calib_n": calib_n,
            "warning": warn,
        }, index=df.index)

    def expected_capped_cost(self, df: pd.DataFrame, B) -> np.ndarray:
        """E[min(C, B) | x] by integrating the conditional quantile function over the grid."""
        X = self.design.transform(df)
        taus = np.array(sorted(self.grid_betas))
        preds = np.maximum.accumulate(np.column_stack([X @ self.grid_betas[t] for t in taus]), axis=1)
        B = np.broadcast_to(np.asarray(B, float), (len(df),))[:, None]
        return np.minimum(np.exp(preds), B).mean(axis=1)

    # ------------------------------------------------------------------ online
    def aci_for(self, model_version: str) -> ACI:
        if model_version not in self.aci:
            d = self.aci["default"]
            self.aci[model_version] = ACI(d.alpha, d.gamma, d.scores.maxlen, np.fromiter(d.scores, float))
        return self.aci[model_version]

    def observe(self, df: pd.DataFrame, cost, model_version: str = "default") -> np.ndarray:
        """Feed realised costs into ACI (sequentially, in row order). Returns miss flags."""
        aci = self.aci_for(model_version)
        scores = np.log(np.asarray(cost, float)) - self.base_log(df)
        return np.array([aci.update(s) for s in scores])

    def reset_model_version(self, model_version: str):
        """A provider-side model change: restart ACI from the target level (keeps scores)."""
        aci = self.aci_for(model_version)
        aci.alpha_t = aci.alpha

    # ------------------------------------------------------------------ evaluation
    def evaluate(self, test: pd.DataFrame, strata: list[str] | None = None) -> dict:
        q = self.quote(test)
        c = test["cost"].to_numpy(float)
        cov = c <= q["quote"].to_numpy()
        out = {
            "coverage": float(cov.mean()),
            "target": 1.0 - self.config.alpha,
            "quote_over_mean_cost": float(q["quote"].mean() / c.mean()),
            "median_quote_over_median_cost": float(q["quote"].median() / np.median(c)),
            "pinball_log": pinball(np.log(c), q["log_quote"], 1 - self.config.alpha),
            "n_test": int(len(test)),
        }
        for s in strata or []:
            if s in test.columns:
                vals = test[s]
                if vals.dtype.kind == "f" and vals.nunique() > 10:
                    vals = pd.qcut(vals, 3, labels=["low", "mid", "high"])
                out[f"coverage_by_{s}"] = {str(k): float(cov[(vals == k).to_numpy()].mean())
                                           for k in pd.unique(vals)}
        self.card["evaluation"] = out
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return path

    @staticmethod
    def load(path: str | Path) -> "QuoteEngine":
        with open(path, "rb") as f:
            return pickle.load(f)
