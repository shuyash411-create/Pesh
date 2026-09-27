import numpy as np

from pesh.config import PricingParams
from pesh.cost_model import (expected_cost_given_steps, linear_slope_after_window, quadratic_coefficients,
                             run_cost, window_bind_step)


def test_quadratic_identity_exact_for_constant_tokens():
    pr = PricingParams(window=1e15)
    c0, mu, mo = 8000.0, 3000.0, 600.0
    for T in (1, 2, 5, 40, 137):
        exact = run_cost(c0, np.full(T, mu), np.full(T, mo), pr)
        assert np.isclose(exact, expected_cost_given_steps(T, c0, mu, mo, pr), rtol=1e-12)


def test_kappa2_matches_paper_formula():
    pr = PricingParams()
    k2, _, _ = quadratic_coefficients(8000, 3640, 600, pr)
    assert np.isclose(k2, 0.5 * pr.delta * pr.p_in * 3640)
    assert 5.0e-4 < k2 < 6.0e-4          # paper: ~5.5e-4 $/step^2


def test_monte_carlo_mean_matches_closed_form(rng):
    pr = PricingParams(window=1e15)
    c0, T = 8000.0, 60
    costs = [run_cost(c0, rng.lognormal(7.6, 0.9, T), rng.lognormal(6.2, 0.6, T), pr) for _ in range(4000)]
    mu, mo = np.exp(7.6 + 0.9**2 / 2), np.exp(6.2 + 0.6**2 / 2)
    assert abs(np.mean(costs) / expected_cost_given_steps(T, c0, mu, mo, pr) - 1) < 0.01


def test_cost_becomes_linear_once_window_binds():
    pr = PricingParams(window=100_000)
    c0, mu, mo = 8000.0, 3000.0, 600.0
    t_star = window_bind_step(c0, mu, pr)
    T1, T2 = int(3 * t_star), int(6 * t_star)
    c1 = run_cost(c0, np.full(T1, mu), np.full(T1, mo), pr)
    c2 = run_cost(c0, np.full(T2, mu), np.full(T2, mo), pr)
    slope = (c2 - c1) / (T2 - T1)
    assert np.isclose(slope, linear_slope_after_window(mu, mo, pr), rtol=0.05)
