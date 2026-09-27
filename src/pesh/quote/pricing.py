"""Assurance level L3: fixed / guaranteed-maximum prices (Section 4.7, Proposition 6).

price = E[min(C,B) | x]                         expected capped cost
      + z_{1-eps} * sigma_B / sqrt(N)            idiosyncratic pooling loading (Prop. 6 i, iii)
      + systematic loading                        common provider shock does not diversify (Prop. 6 iv)
      + margin

The ceiling B (policy limit) is what makes the guarantee insurable: for costs in [0,B],
sigma_B <= B/2. An index-linked ceiling scales B with a published model price index,
passing the systematic shock back to the buyer, like a fuel surcharge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class PricingPolicy:
    portfolio_n: int = 1000          # tasks pooled by the guarantor
    eps: float = 0.01                # ruin probability tolerated per period
    shock_mult: float = 1.4          # provider-side common shock multiplier
    shock_prob: float = 0.2          # its probability per period
    index_linked: bool = False       # if True the buyer bears the systematic shock
    margin: float = 0.05             # proportional margin over loaded cost


def systematic_loading(expected: np.ndarray, shock_mult: float, shock_prob: float, eps: float) -> np.ndarray:
    """q_{1-eps}(Z) E C0 - E[Z] E C0 for a two-point shock Z in {1, shock_mult}."""
    q = shock_mult if shock_prob > eps else 1.0
    ez = 1.0 + shock_prob * (shock_mult - 1.0)
    return np.asarray(expected) * (q - ez)


def fixed_price(expected_capped: np.ndarray, B: np.ndarray, policy: PricingPolicy | None = None,
                sigma_capped: np.ndarray | None = None) -> dict:
    """Per-task fixed price for a portfolio guarantee with ceiling B."""
    pol = policy or PricingPolicy()
    e = np.asarray(expected_capped, float)
    B = np.broadcast_to(np.asarray(B, float), e.shape)
    sigma = B / 2.0 if sigma_capped is None else np.asarray(sigma_capped, float)
    idio = norm.ppf(1 - pol.eps) * sigma / np.sqrt(pol.portfolio_n)
    sys = np.zeros_like(e) if pol.index_linked else systematic_loading(e, pol.shock_mult, pol.shock_prob, pol.eps)
    ez = 1.0 + pol.shock_prob * (pol.shock_mult - 1.0)
    base = e * (1.0 if pol.index_linked else ez)
    price = (base + idio + sys) * (1 + pol.margin)
    return {"price": price, "expected_capped": e, "idiosyncratic_loading": idio,
            "systematic_loading": sys, "ceiling": B}


def index_linked_ceiling(B: float, index_now: float, index_base: float) -> float:
    """Ceiling adjusted to a published cost index for the underlying model."""
    return float(B * index_now / index_base)


def gmp_ceiling(quote: np.ndarray, headroom: float = 1.0) -> np.ndarray:
    """Guaranteed-maximum-price ceiling B = headroom * q_alpha(x) (L2 contract)."""
    return np.asarray(quote, float) * headroom
