"""Cost accounting identity, Eq. (3)-(4), and Proposition 1.

I_t = min(W, c0 + sum_{s<t} a_s)                           (3)
C   = sum_t [ w p_in n_t + delta p_in (I_t - n_t) + p_out o_t ]   (4)

with n_1 = c0 and n_t = a_{t-1} the newly appended (cache-written) tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import PricingParams


@dataclass
class StepCost:
    input_tokens: float
    new_tokens: float
    cached_tokens: float
    output_tokens: float
    cost: float


def step_cost(context: float, new_tokens: float, output_tokens: float, pricing: PricingParams) -> StepCost:
    """Cost of one model call whose full context is `context`, of which `new_tokens` are uncached."""
    context = min(context, pricing.window)
    new = min(new_tokens, context)
    cached = context - new
    cost = pricing.w * pricing.p_in * new + pricing.delta * pricing.p_in * cached + pricing.p_out * output_tokens
    return StepCost(context, new, cached, output_tokens, cost)


def run_cost(c0: float, appended: np.ndarray, outputs: np.ndarray, pricing: PricingParams) -> float:
    """Exact cost of a run given appended tokens a_1..a_T and outputs o_1..o_T (Eq. 4)."""
    appended = np.asarray(appended, float)
    outputs = np.asarray(outputs, float)
    T = len(outputs)
    if T == 0:
        return 0.0
    prior = np.concatenate([[0.0], np.cumsum(appended[:-1])])  # sum_{s<t} a_s
    context = np.minimum(pricing.window, c0 + prior)
    new = np.concatenate([[c0], appended[:-1]])
    new = np.minimum(new, context)
    cached = context - new
    return float(np.sum(pricing.w * pricing.p_in * new + pricing.delta * pricing.p_in * cached
                        + pricing.p_out * outputs))


def quadratic_coefficients(c0: float, mu_a: float, mu_o: float, pricing: PricingParams) -> tuple[float, float, float]:
    """(kappa2, kappa1, kappa0) of E[C|T] = kappa2 T^2 + kappa1 T + kappa0 with W = infinity.

    Derived exactly from Eq. (4) (Appendix A):
      E[C|T] = w p c0 + (T-1) w p mu + delta p [(T-1) c0 + mu (T-1)(T-2)/2] + p_out mu_o T
    """
    p, w, d = pricing.p_in, pricing.w, pricing.delta
    k2 = 0.5 * d * p * mu_a
    k1 = w * p * mu_a + d * p * c0 - 1.5 * d * p * mu_a + pricing.p_out * mu_o
    k0 = w * p * c0 - w * p * mu_a - d * p * c0 + d * p * mu_a
    return k2, k1, k0


def expected_cost_given_steps(T, c0: float, mu_a: float, mu_o: float, pricing: PricingParams):
    k2, k1, k0 = quadratic_coefficients(c0, mu_a, mu_o, pricing)
    T = np.asarray(T, float)
    return k2 * T**2 + k1 * T + k0


def window_bind_step(c0: float, mu_a: float, pricing: PricingParams) -> float:
    """t* ~ (W - c0)/mu after which cost becomes asymptotically linear in T."""
    return max(0.0, (pricing.window - c0) / mu_a)


def linear_slope_after_window(mu_a: float, mu_o: float, pricing: PricingParams) -> float:
    p = pricing.p_in
    return pricing.delta * p * pricing.window + (pricing.w - pricing.delta) * p * mu_a + pricing.p_out * mu_o
