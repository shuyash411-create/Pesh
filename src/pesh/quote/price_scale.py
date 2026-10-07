"""Price-scaling layer: move quotes between the engine's training price and a model's own price.

The engine learns cost in the dollars of the logs it was trained on, i.e. at one per-token
price (the simulator's default ``PricingParams``: $3 / $15 per Mtok). A run on another model
uses the same tokens at a different price, so its cost is the training cost re-priced:

    ratio(m) = (p_in_m * I + p_out_m * O) / (p_in_ref * I + p_out_ref * O)

where I and O are the training runs' total input-price units (input, cache-write and
cache-read tokens weighted as they were billed, so ``p_in_ref * I`` is everything not billed
at the output price) and output tokens. Every quantile scales by the same positive factor,
so the conformal guarantee carries over unchanged: quote in training dollars, multiply by
``ratio`` to show the model's dollars, and divide an observed cost by ``ratio`` before it is
fed back to ACI.

The engine itself is untouched; this only wraps what goes in and out of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from ..config import PricingParams


@dataclass(frozen=True)
class PriceScaler:
    reference: PricingParams        # per-token price the engine's training costs were charged at
    input_units: float              # training input tokens, in input-price units (sum over runs)
    output_tokens: float            # training output tokens (sum over runs)

    def ratio(self, price_in: float, price_out: float) -> float:
        """Model cost / training cost for the training runs' token mix."""
        if not (np.isfinite(price_in) and np.isfinite(price_out)) or price_in < 0 or price_out < 0:
            raise ValueError(f"invalid per-token price ({price_in}, {price_out})")
        ref = self.reference.p_in * self.input_units + self.reference.p_out * self.output_tokens
        return (price_in * self.input_units + price_out * self.output_tokens) / ref

    def ratio_for(self, model) -> float:
        """``ratio`` for anything with per-token ``price_in`` / ``price_out`` (an observe ModelSpec)."""
        return self.ratio(model.price_in, model.price_out)

    @classmethod
    def from_runs(cls, runs: pd.DataFrame, reference: PricingParams | None = None) -> "PriceScaler":
        """Token mix from run logs with ``cost`` and ``output_tokens`` (the training logs).

        Input-price units are backed out of the cost, so cache writes and discounted cache reads
        count as they were billed. Runs that switched to a cheaper fallback (``degraded``) were
        not billed at the reference price and are left out.
        """
        ref = reference or PricingParams()
        r = runs
        if "degraded" in r.columns:
            r = r[~r["degraded"].astype(bool)]
        cost = pd.to_numeric(r["cost"], errors="coerce")
        out = pd.to_numeric(r["output_tokens"], errors="coerce")
        ok = cost.notna() & out.notna()
        if not ok.any():
            raise ValueError("no runs with cost and output_tokens to estimate the token mix from")
        out_sum = float(out[ok].sum())
        in_units = float(np.maximum(cost[ok] - ref.p_out * out[ok], 0.0).sum() / ref.p_in)
        if in_units + out_sum <= 0:
            raise ValueError("training runs have no tokens")
        return cls(ref, in_units, out_sum)

    @classmethod
    def for_simulator(cls) -> "PriceScaler":
        """Token mix of the simulator the default engines are trained on (cached per process)."""
        return _simulator_scaler()


@lru_cache(maxsize=1)
def _simulator_scaler() -> PriceScaler:
    from ..sim.dgp import simulate
    return PriceScaler.from_runs(simulate(400, 1, seed=0), PricingParams())
