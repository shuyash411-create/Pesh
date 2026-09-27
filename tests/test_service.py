import pytest
from fastapi.testclient import TestClient

from pesh.config import QuoteConfig
from pesh.control.feasibility import train_from_simulation
from pesh.quote.engine import QuoteEngine
from pesh.service.api import create_app
from pesh.service.gateway import BudgetExceeded, BudgetSession, RunStopped

FEATS = {"x_repo": 0.3, "x_complex": -0.2, "human_label": 1}


@pytest.fixture(scope="module")
def client(sim_small):
    tr, ca, _ = sim_small
    eng = QuoteEngine(QuoteConfig()).fit(tr, ca)
    fm = train_from_simulation(n_tasks=600, seed=4)
    return TestClient(create_app(eng, fm, exploration_rate=0.0, drift_min_n=50))


def test_quote_endpoint(client):
    r = client.post("/quote", json={"features": FEATS}).json()
    assert r["quote"] > r["p50"] > 0 and r["ceiling"] == pytest.approx(r["quote"])
    assert r["fixed_price"] > r["expected_capped_cost"]
    assert client.post("/quote", json={"features": {"x_repo": 1}}).status_code == 422


def test_run_lifecycle_is_bounded(client):
    run = client.post("/runs", json={"task_id": "t1", "features": FEATS, "budget": 0.5}).json()
    rid, spent, decision = run["run_id"], 0.0, "continue"
    ctx = 10_000
    for _ in range(200):
        auth = client.post(f"/runs/{rid}/authorize", json={"input_tokens": ctx, "cache_read_tokens": ctx - 3000,
                                                           "max_output_tokens": 1000}).json()
        if not auth["allowed"]:
            break
        s = client.post(f"/runs/{rid}/step", json={"input_tokens": ctx, "cache_read_tokens": ctx - 3000,
                                                   "cache_write_tokens": 3000, "output_tokens": 600}).json()
        spent, decision = s["spent"], s["decision"]
        ctx += 3500
        if decision == "stop":
            break
    assert spent <= 0.5
    assert "degrade" in client.get(f"/runs/{rid}").json()["decisions"]
    fin = client.post(f"/runs/{rid}/finish", json={"success": False}).json()
    assert fin["status"] in ("finished", "stopped", "capped")
    assert client.post(f"/runs/{rid}/finish", json={"success": True}).status_code == 409


def test_drift_alarm_after_cost_shift(client):
    for i in range(60):
        run = client.post("/runs", json={"task_id": f"d{i}", "features": FEATS, "model_version": "v2"}).json()
        q = run["quote"]
        client.post(f"/runs/{run['run_id']}/step", json={"input_tokens": 1, "output_tokens": 0,
                                                         "cost": q * 1.5 if i % 2 else q * 0.5})
        client.post(f"/runs/{run['run_id']}/finish", json={"success": True})
    mon = client.get("/monitor/coverage", params={"model_version": "v2"}).json()
    assert mon["n"] == 60 and mon["drift_alarm"]
    assert mon["aci"]["alpha_t"] < 0.1


def test_budget_session_blocks_before_paying():
    sess = BudgetSession(budget=0.05, degrade=False)

    def fake_llm(**kw):
        return "ok", {"input_tokens": 5000, "output_tokens": 500}

    n = 0
    with pytest.raises(BudgetExceeded):
        for _ in range(100):
            sess.call(fake_llm, input_tokens=5000, max_output_tokens=500)
            n += 1
    assert sess.spent <= 0.05 and n > 0
    with pytest.raises(RunStopped):
        sess.call(fake_llm, input_tokens=10, max_output_tokens=10)
