import io

import pytest
from fastapi.testclient import TestClient

from pesh.cli import build_parser
from pesh.config import ControllerConfig, QuoteConfig
from pesh.control.feasibility import train_from_simulation
from pesh.dashboard import create_dashboard_app
from pesh.quote.engine import QuoteEngine
from pesh.sim.dgp import simulate


@pytest.fixture(scope="module")
def client(sim_small):
    tr, ca, _ = sim_small
    eng = QuoteEngine(QuoteConfig()).fit(tr, ca)
    fm = train_from_simulation(n_tasks=600, seed=4)
    app = create_dashboard_app(eng, fm, exploration_rate=0.0, controller_config=ControllerConfig())
    return TestClient(app)


def test_index_page_has_three_screens(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    for name in ("Analyze logs", "Quote a task", "Live monitor"):
        assert name in r.text


def test_analyze_sample_data(client):
    r = client.post("/dashboard/analyze?sample=true&runs_per_month=1000").json()
    assert 0.6 < r["predictability"]["value"] < 0.95
    assert r["tail"]["value"] > 0 and isinstance(r["tail"]["infinite_variance"], bool)
    f = r["forecast"]
    assert f["runs_per_month"] == 1000 and 0 < f["expected"] < f["p90"]
    assert "90% of months below" in f["text"]
    h = r["histogram"]
    assert sum(h["counts"]) == r["n_runs"] and h["median"] < h["p90"] < h["p99"]
    assert 0.8 < r["quote_check"]["coverage"] < 1.0


def test_analyze_uploaded_csv_and_bad_file(client):
    df = simulate(300, 2, seed=5)[["task_id", "run_id", "cost", "success"]]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    r = client.post("/dashboard/analyze?filename=runs.csv", content=buf.getvalue().encode())
    assert r.status_code == 200
    body = r.json()
    assert body["n_runs"] == 600 and body["source"] == "runs.csv"
    assert body["tail"]["value"] is not None            # 600 runs >= 500
    assert body["quote_check"] is None                  # no feature columns in this file
    bad = client.post("/dashboard/analyze?filename=x.csv", content=b"a,b\n1,2\n")
    assert bad.status_code == 422 and "run log" in bad.json()["detail"]
    assert client.post("/dashboard/analyze?filename=x.csv", content=b"").status_code == 422


def test_features_drive_the_quote_form(client):
    cfg = client.get("/dashboard/features").json()
    names = [f["name"] for f in cfg["features"]]
    assert names == ["x_repo", "x_complex", "human_label"]
    choice = cfg["features"][2]
    assert [o["label"] for o in choice["options"]][0] == "Under 15 minutes"
    feats = {f["name"]: f["default"] for f in cfg["features"]}
    q = client.post("/quote", json={"features": feats}).json()
    assert 0 < q["p50"] < q["ceiling"] and q["fixed_price"] > 0


def _drive(client, traj, compact_to):
    """Python replay of the page's 'Simulate a run' loop through the real endpoints."""
    run = client.post("/runs", json={"task_id": "demo", "features": traj["features"]}).json()
    rid, budget = run["run_id"], run["budget"]
    base = traj["initial_context"]
    prior, new, ended = 0.0, base, None
    for s in traj["steps"]:
        ctx = min(1e6, base + prior)
        w = min(new, ctx)
        tok = {"input_tokens": ctx, "cache_read_tokens": ctx - w, "cache_write_tokens": w}
        if not client.post(f"/runs/{rid}/authorize", json={**tok, "max_output_tokens": max(1500, 2 * s["output"])}
                           ).json()["allowed"]:
            ended = "capped"
            break
        st = client.post(f"/runs/{rid}/step", json={**tok, "output_tokens": s["output"],
                                                    "repeated_views": s["repeated_views"]}).json()
        prior, new = prior + s["appended"], s["appended"]
        if st["decision"] == "degrade":
            base, prior, new = min(compact_to, base + prior), 0.0, min(compact_to, base + prior)
        if st["decision"] == "stop":
            ended = st["status"]
            break
    client.post(f"/runs/{rid}/finish", json={"success": False if ended else traj["success"]})
    return client.get(f"/runs/{rid}").json(), budget


def test_simulated_runs_stay_under_ceiling_and_show_in_monitor(client):
    compact_to = client.get("/dashboard/features").json()["compact_to"]
    for seed in range(20):
        traj = client.get(f"/dashboard/sample-task?seed={seed}&drift={'true' if seed % 2 else 'false'}").json()
        assert traj["steps"] and set(traj["features"]) == {"x_repo", "x_complex", "human_label"}
        rec, budget = _drive(client, traj, compact_to)
        assert rec["spent"] <= budget + 1e-9 and rec["closed"]
    runs = client.get("/dashboard/runs?limit=50").json()
    assert len(runs) >= 20 and runs[0]["closed"]
    mon = client.get("/monitor/coverage").json()
    assert mon["n"] > 0 and 0 <= mon["rolling_coverage"] <= 1


def test_authorize_refusal_marks_run_capped(client):
    feats = {"x_repo": 0.0, "x_complex": 0.0, "human_label": "1"}
    run = client.post("/runs", json={"task_id": "big", "features": feats}).json()   # ceiling = quote
    assert run["budget"] == pytest.approx(run["quote"])
    rid = run["run_id"]
    a = client.post(f"/runs/{rid}/authorize", json={"input_tokens": 5_000_000, "max_output_tokens": 2000}).json()
    assert not a["allowed"]
    assert client.get(f"/runs/{rid}").json()["status"] == "capped"
    fin = client.post(f"/runs/{rid}/finish", json={"success": True}).json()
    # true cost would have exceeded the ceiling, which equals the quote: a miss, not a hit
    assert fin["status"] == "capped" and fin["within_quote"] is False and fin["aci_missed"]


def test_dashboard_cli_parses():
    a = build_parser().parse_args(["dashboard", "--no-browser", "--port", "9001"])
    assert a.cmd == "dashboard" and a.port == 9001 and a.no_browser and a.engine is None
