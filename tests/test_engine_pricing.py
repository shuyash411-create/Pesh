import numpy as np

from pesh.config import QuoteConfig
from pesh.quote.engine import QuoteEngine
from pesh.quote.pricing import PricingPolicy, fixed_price, index_linked_ceiling


def test_engine_marginal_coverage_and_roundtrip(sim_small, tmp_path):
    tr, ca, te = sim_small
    eng = QuoteEngine(QuoteConfig(mondrian_by="human_label")).fit(tr, ca)
    ev = eng.evaluate(te, strata=["difficulty"])
    assert 0.86 <= ev["coverage"] <= 0.94
    assert ev["coverage_by_difficulty"]["high"] < ev["coverage_by_difficulty"]["low"]   # conditional gap
    q = eng.quote(te.head(5))
    assert (q["quote"] > q["p50"]).all()
    eng2 = QuoteEngine.load(eng.save(tmp_path / "e.pkl"))
    assert np.allclose(eng2.quote(te.head(5))["quote"], q["quote"])


def test_engine_uses_powell_on_capped_logs(sim_small):
    tr, ca, te = sim_small
    # cap high enough that Q_0.9(C|x) < B for most x: identified region of Proposition 5
    B = float(np.quantile(tr["cost"], 0.97))
    capped = tr.assign(capped=tr["cost"] >= B, cost=np.minimum(tr["cost"], B), cap=B)
    eng = QuoteEngine().fit(capped, ca)
    assert eng.card["method"].startswith("powell")
    assert 0 <= eng.card["unidentified_share"] < 0.2
    naive = QuoteEngine().fit(capped.assign(capped=False, cap=np.inf), ca)
    truth = QuoteEngine().fit(tr, ca)
    ref = truth.base_log(te)
    err_powell = np.mean(np.abs(eng.base_log(te) - ref))
    err_naive = np.mean(np.abs(naive.base_log(te) - ref))
    assert err_powell < err_naive


def test_expected_capped_cost_and_price(sim_small):
    tr, ca, te = sim_small
    eng = QuoteEngine().fit(tr, ca)
    x = te.head(200)
    e = eng.expected_capped_cost(x, 3.0)
    assert (e <= 3.0).all()
    assert abs(e.mean() / np.minimum(x["cost"], 3.0).mean() - 1) < 0.2
    p = fixed_price(e, 3.0, PricingPolicy())
    assert (p["price"] > e).all()
    p_idx = fixed_price(e, 3.0, PricingPolicy(index_linked=True))
    assert (p_idx["price"] < p["price"]).all()
    assert np.isclose(index_linked_ceiling(3.0, 1.2, 1.0), 3.6)


def test_online_aci_tracks_shift(sim_small):
    from pesh.sim.dgp import simulate

    tr, ca, _ = sim_small
    eng = QuoteEngine(QuoteConfig(aci_window=300)).fit(tr, ca)
    shifted = simulate(3000, 1, seed=77, verbosity=1.35, step_mult=1.25)
    before = (shifted["cost"] <= eng.quote(shifted)["quote"]).mean()
    hits = []
    for i in range(len(shifted)):
        row = shifted.iloc[[i]]
        hits.append(row["cost"].iloc[0] <= eng.quote(row, online=True)["quote"].iloc[0])
        eng.observe(row, row["cost"])
    assert before < 0.85
    assert abs(np.mean(hits[1000:]) - 0.9) < 0.03
