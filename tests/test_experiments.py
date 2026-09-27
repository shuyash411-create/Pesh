"""Statistical reproduction of the paper's Section 5 (run with: pytest -m slow)."""

import pytest

from pesh.eval import experiments as E

pytestmark = pytest.mark.slow


def test_e1_predictability_ceiling():
    r = E.e1_predictability(scale=0.5)
    assert 0.74 <= r["rho_tau"] <= 0.88                    # paper 0.804
    assert 0.35 <= r["feature_r2_log"] <= 0.55             # paper 0.422
    assert r["feature_r2_log"] < r["oracle_r2_log"] <= r["rho_tau"] + 0.02


def test_e2_quote_coverage():
    t = E.e2_quotes(scale=0.5, finite_sample=False)["table"]
    for m in ["CQR", "Mondrian CQR (by human label)"]:
        assert 0.88 <= t.loc[m, "marginal"] <= 0.92
        assert t.loc[m, "hard"] < 0.84 < 0.95 < t.loc[m, "easy"]      # conditional under-coverage
    assert t.loc["Point x1.5", "marginal"] < 0.87
    assert t.loc["CQR", "loop_runs"] < t.loc["CQR", "clean_runs"]


def test_e2b_aci_restores_coverage():
    r = E.e2b_drift(scale=0.6, periods=50, shift_at=25)
    assert r["static_post"] < 0.84
    assert abs(r["aci_post"] - 0.90) < 0.025


def test_e3_powell_removes_censoring_bias():
    t = E.e3_censoring(scale=1.0)["table"]                  # paper's n = 8,000
    for b in ("bias_repo", "bias_complex"):
        assert t.loc["naive QR on capped logs", b] < -0.25     # paper -40%
        assert abs(t.loc["Powell censored QR", b]) < 0.10     # paper -6% / 0%
    assert t.loc["naive QR on capped logs", "coverage_hard_tercile"] < t.loc["Powell censored QR",
                                                                             "coverage_hard_tercile"]


def test_e4_controllers():
    t = E.e4_controllers(scale=0.5)["table"]
    p4 = t.loc["P4 cap + feasibility + degrade"]
    assert -0.50 <= p4["mean_cost_change"] <= -0.33
    assert -0.88 <= p4["cvar95_change"] <= -0.70
    assert -6.0 <= p4["success_change_pp"] <= 0
    assert (t.iloc[1:]["p_exceed_B"] == 0).all()


def test_e5_tails_and_pooling():
    r = E.e5_tails_pooling(scale=0.5)
    lo, hi = r["mechanism"]["ratio"]
    assert 0.45 <= lo and hi <= 0.65                        # tail-index halving
    assert -0.6 < r["pooling"]["slope_capped"] < -0.4
    assert r["pooling"]["slope_capped_shock"] > -0.2 and r["pooling"]["floor_capped_shock"] > 0.15
    w = r["wtp"]
    assert w["eta=1, v=$0"]["wtp"] > w["eta=1, v=$20"]["wtp"] > w["eta=0, v=$20"]["wtp"]
