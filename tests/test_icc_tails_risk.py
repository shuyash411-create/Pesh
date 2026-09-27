import numpy as np

from pesh.econometrics.icc import icc_anova, icc_reml, predictability_report
from pesh.econometrics.risk import cvar, loglog_slope, pooling_loading, willingness_to_pay
from pesh.econometrics.tails import hill, powerlaw_fit


def test_icc_recovers_variance_components(rng):
    k, r = 2000, 6
    u = rng.normal(0, 1.0, k)
    y = np.repeat(u, r) + rng.normal(0, 0.5, k * r)
    g = np.repeat(np.arange(k), r)
    res = icc_anova(y, g)
    assert abs(res.rho - 0.8) < 0.02
    assert abs(icc_reml(y[:3000], g[:3000]).rho - 0.8) < 0.05
    rep = predictability_report(y, g, predictions=np.repeat(u, r))
    assert rep["r2_predictor"] <= res.rho + 0.02


def test_hill_on_pareto(rng):
    x = (1 - rng.random(200_000)) ** (-1 / 2.2)
    assert abs(hill(x, 2000) - 2.2) < 0.15
    assert abs(powerlaw_fit(x)["alpha"] - 2.2) < 0.2


def test_cvar():
    x = np.arange(1, 101, dtype=float)
    assert cvar(x, 0.95) == np.mean(x[-5:])


def test_pooling_needs_truncation(rng):
    heavy = (1 - rng.random(100_000)) ** (-1 / 1.2)
    sizes = [10, 100, 1000]
    capped = pooling_loading(np.minimum(heavy, np.quantile(heavy, 0.9)), sizes, reps=1500, seed=1)
    assert -0.6 < loglog_slope(capped) < -0.4
    shock = pooling_loading(np.minimum(heavy, np.quantile(heavy, 0.9)), sizes, reps=1500, seed=1,
                            shock_mult=1.4, shock_prob=0.2)
    assert shock[1000] > 0.15                               # systematic shock does not diversify


def test_wtp_decomposition():
    s0 = np.array([100.0, 100, 100, 400])
    s1 = np.array([90.0, 90, 90, 150])
    w = willingness_to_pay(s0, s1, 0.6, 0.58, 100, eta=1.0, value_per_success=10.0, level=0.75)
    assert np.isclose(w["wtp"], w["expected_saving"] + w["risk_saving"] - w["quality_cost"])
    assert np.isclose(w["quality_cost"], 10 * 100 * 0.02)
