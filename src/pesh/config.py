"""Parameter containers.

Symbols follow the paper (Tiwari, 2026, "Pricing the Unpredictable"):
  p_in, p_out   per-token list prices (USD)
  w             cache-write multiplier on p_in (w > 1)
  delta         cache-read discount on p_in (delta < 1)
  window        context window W in tokens
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class PricingParams:
    p_in: float = 3.0e-6
    p_out: float = 15.0e-6
    w: float = 1.25
    delta: float = 0.10
    window: float = 1_000_000.0

    def scaled(self, mult: float) -> "PricingParams":
        """Same structure, prices multiplied (e.g. a cheaper fallback model)."""
        return PricingParams(self.p_in * mult, self.p_out * mult, self.w, self.delta, self.window)


@dataclass
class DGPParams:
    """Data-generating process of Section 5.1.

    Defaults are the SMM-calibrated values from `pesh calibrate` (see sim/calibrate.py).

    Latent difficulty d = X'theta + u, u ~ N(0, sigma_u^2).
    Steps T = 1 + Poisson(m * G) + L, m = exp(mu_T + lam * d), G log-normal,
    loop L ~ floor(Lomax(alpha_L, scale_L)) with prob logit^-1(q0 + q1 d).
    """

    theta: tuple[float, float] = (0.60, 0.50)
    sigma_u: float = 0.789
    label_noise: float = 1.732         # sd of eps in the noisy human-difficulty label
    mu_T: float = 3.320                # log base step intensity
    lam: float = 0.709                 # elasticity of steps to difficulty
    sigma_G: float = 0.203             # run-level log-normal step shock
    q0: float = -1.618                 # loop hazard intercept
    q1: float = 0.55                   # loop hazard slope on difficulty
    alpha_L: float = 1.9               # Lomax tail index of loop length
    scale_L: float = 22.28             # Lomax scale of loop length
    c0_log_mean: float = 9.0           # log initial context tokens
    c0_repo_slope: float = 0.25
    a_log_mean: float = 7.610          # log tool/context tokens appended per step
    a_log_sd: float = 0.9
    o_log_mean: float = 6.409          # log model output tokens per step
    o_log_sd: float = 0.6
    tok_difficulty_slope: float = 0.10  # harder tasks append slightly more per step
    s0: float = 0.975                  # success logit intercept
    s1: float = -0.85                  # success logit slope on difficulty
    s2: float = -1.6                   # success penalty for having looped
    t_max: int = 250                   # iteration cap
    loop_repeat_rate: float = 1.6      # mean repeated-file-view signal per step while looping
    base_repeat_rate: float = 0.15     # ... while working productively

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QuoteConfig:
    alpha: float = 0.10
    numeric_features: list[str] = field(default_factory=lambda: ["x_repo", "x_complex"])
    categorical_features: list[str] = field(default_factory=lambda: ["human_label"])
    quadratic: bool = True
    mondrian_by: str | None = None
    min_group_n: int = 100
    aci_gamma: float = 0.01
    aci_window: int = 1000


@dataclass
class ControllerConfig:
    kappa: float = 0.5                 # degrade once spend passes kappa * B
    degrade_price_mult: float = 0.3    # cheaper fallback model price ratio
    compact_to: float = 12_000.0       # context size after compaction (tokens)
    degrade_extra_fail: float = 0.12   # extra failure probability induced by degradation
    feasibility_threshold: float = 0.15  # stop when pi_hat falls below this
    value_of_success: float | None = None  # if set, use rule (10): v*pi >= E[remaining cost]
    min_steps_before_stop: int = 8
