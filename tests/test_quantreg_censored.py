import numpy as np

from pesh.econometrics.censored import chernozhukov_hong, powell_cqr
from pesh.econometrics.quantreg import check_loss, fit_quantile


def _data(rng, n=4000):
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    y = X @ np.array([0.5, 0.8, -0.4]) + rng.standard_normal(n)
    return X, y


def test_check_loss():
    assert np.allclose(check_loss(np.array([2.0, -2.0]), 0.9), [1.8, 0.2])


def test_lp_quantile_regression_recovers_slope(rng):
    X, y = _data(rng)
    b = fit_quantile(X, y, 0.75)
    assert np.allclose(b[1:], [0.8, -0.4], atol=0.07)
    assert np.isclose(b[0], 0.5 + 0.674, atol=0.07)
    assert np.allclose(fit_quantile(X, y, 0.75, method="irls"), b, atol=0.02)


def test_powell_corrects_censoring_bias(rng):
    X, y = _data(rng, 6000)
    cap = np.quantile(y, 0.8)
    yc = np.minimum(y, cap)
    naive = fit_quantile(X, yc, 0.75)
    powell = powell_cqr(X, yc, cap, 0.75)["beta"]
    assert abs(naive[1] / 0.8 - 1) > 0.2               # naive is badly biased toward the cap
    assert abs(powell[1] / 0.8 - 1) < 0.1
    ch = chernozhukov_hong(X, yc, y >= cap, cap, 0.75)["beta"]
    assert abs(ch[1] / 0.8 - 1) < 0.25
