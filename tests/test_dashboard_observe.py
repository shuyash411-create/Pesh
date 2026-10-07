import os
import time

import pytest
from fastapi.testclient import TestClient

from pesh.config import QuoteConfig
from pesh.dashboard import create_dashboard_app
from pesh.dashboard import observe_routes
from pesh.observe import Provider, RunResult
from pesh.quote.engine import QuoteEngine

SECRET = "sk-very-secret-key-9876"


@pytest.fixture
def make_client(sim_small, tmp_path, monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    tr, ca, _ = sim_small

    def make(sim_pause=0.0):
        eng = QuoteEngine(QuoteConfig()).fit(tr, ca)
        app = create_dashboard_app(eng, None, db_path=str(tmp_path / "db.sqlite"), env_path=tmp_path / ".env",
                                   observe_report_dir=str(tmp_path / "rep"), sim_pause=sim_pause,
                                   exploration_rate=0.0)
        return TestClient(app)
    return make


def _wait(c, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get("/dashboard/observe/status").json()
        if not st["running"]:
            return st
        time.sleep(0.05)
    raise AssertionError("observation did not finish")


def test_keys_are_saved_locally_and_never_returned(make_client, tmp_path):
    c = make_client()
    k = c.get("/dashboard/observe/keys").json()
    assert not k["any_connected"] and {p["env"] for p in k["providers"]} == {
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "TOGETHER_API_KEY", "FIREWORKS_API_KEY"}
    r = c.post("/dashboard/observe/keys", json={"keys": {"OPENAI_API_KEY": SECRET}})
    assert SECRET not in r.text
    row = next(p for p in r.json()["providers"] if p["provider"] == "openai")
    assert row["connected"] and row["hint"] == "…9876"
    env = tmp_path / ".env"
    assert f"OPENAI_API_KEY={SECRET}" in env.read_text()
    if os.name == "posix":
        assert oct(env.stat().st_mode)[-3:] == "600"
    for path in ("/dashboard/observe/keys", "/dashboard/observe/setup", "/dashboard/observe/status"):
        assert SECRET not in c.get(path).text
    # a new app reads the saved key; an empty string removes it
    c2 = make_client()
    assert c2.get("/dashboard/observe/keys").json()["any_connected"]
    c2.post("/dashboard/observe/keys", json={"keys": {"OPENAI_API_KEY": ""}})
    assert "OPENAI_API_KEY" not in env.read_text()
    assert c2.post("/dashboard/observe/keys", json={"keys": {"OPENAI_API_KEY": "short"}}).status_code == 422
    assert c2.post("/dashboard/observe/keys", json={"keys": {"PATH": "x" * 20}}).status_code == 422


def test_parse_tasks_plain_and_jsonl():
    tasks = observe_routes.parse_tasks("Write a haiku\n\n{\"task_id\": \"t2\", \"text\": \"Fix the bug in `a.py`\"}\n")
    assert [t.task_id for t in tasks][1] == "t2" and tasks[0].task_id.startswith("task-")
    assert {"x_repo", "x_complex", "human_label"} <= set(tasks[0].features)
    assert observe_routes.parse_tasks("Write a haiku")[0].task_id == tasks[0].task_id   # stable id -> resumable
    with pytest.raises(Exception):
        observe_routes.parse_tasks("   \n  ")


def test_simulated_observation_fills_shared_store_and_analyze(make_client):
    c = make_client()
    s = c.get("/dashboard/observe/setup").json()
    r = c.post("/dashboard/observe/start", json={"tasks": "\n".join(s["sample_tasks"]), "models": ["claude", "gpt"],
                                                 "variance_runs": 3, "cross_runs": 2}).json()
    assert r["mode"] == "simulated" and r["planned"] == 3 * (2 * 2 + 3)
    st = _wait(c)
    assert st["done_this_run"] == st["planned"] == len(st["rows"])
    row = st["rows"][0]
    assert row["provider"] == "sim" and row["predicted"] > 0 and row["actual"] > 0 and row["task_text"]
    assert all(r["within"] == (r["actual"] <= r["ceiling"]) for r in st["rows"])
    summ = st["summary"]
    assert 0 <= summ["rho_tau"] <= 1 and summ["mean_pct_error"] > 0 and 0 <= summ["coverage"] <= 1
    assert set(summ["cheapest"]) and set(summ["cheapest"].values()) <= {"claude", "gpt"}
    # the Analyze screen reads the same SQLite file
    a = c.post("/dashboard/analyze?source=observations").json()
    assert a["n_runs"] == st["planned"] and "observation" in a["source"]
    assert c.get("/dashboard/observation-count").json()["finished"] == st["planned"]
    # starting again with the same tasks resumes: nothing is repeated
    c.post("/dashboard/observe/start", json={"tasks": "\n".join(s["sample_tasks"]), "models": ["claude", "gpt"],
                                             "variance_runs": 3, "cross_runs": 2})
    assert _wait(c)["total_observations"] == st["planned"]


class FakeOpenAI(Provider):
    name = "openai"

    def available(self):
        return True

    def run(self, model, task, run_idx=0, kind="cross"):
        time.sleep(0.01)
        return RunResult(1000, 200, 0.4, 5.0, 1, True)


def test_live_mode_skips_models_without_keys_and_respects_spend_limit(make_client, monkeypatch):
    monkeypatch.setattr(observe_routes, "build_providers", lambda env=None, **kw: {"openai": FakeOpenAI()})
    c = make_client()
    r = c.post("/dashboard/observe/start", json={"tasks": "a\nb\nc", "models": ["claude", "gpt"],
                                                 "variance_runs": 5, "cross_runs": 2, "max_spend": 1.0}).json()
    assert r["mode"] == "live" and r["models"] == ["gpt"] and r["skipped"] == ["claude"]
    st = _wait(c)
    assert st["budget_hit"] and st["spent"] == pytest.approx(1.2)        # stops once $1 is passed
    assert {row["model"] for row in st["rows"]} == {"gpt"}
    # forcing the simulator ignores saved keys
    r2 = c.post("/dashboard/observe/start", json={"tasks": "d", "models": ["claude"], "variance_runs": 1,
                                                  "cross_runs": 1, "simulate": True}).json()
    assert r2["mode"] == "simulated"
    _wait(c)


def test_stop_and_validation(make_client):
    c = make_client(sim_pause=0.05)
    assert c.post("/dashboard/observe/start", json={"tasks": "", "models": ["claude"]}).status_code == 422
    assert c.post("/dashboard/observe/start", json={"tasks": "x", "models": ["nope"]}).status_code == 422
    c.post("/dashboard/observe/start", json={"tasks": "one\ntwo", "models": ["claude"], "variance_runs": 20,
                                             "cross_runs": 2})
    assert c.post("/dashboard/observe/start", json={"tasks": "x", "models": ["claude"]}).status_code == 409
    c.post("/dashboard/observe/stop")
    st = _wait(c)
    assert not st["running"] and st["done_this_run"] < st["planned"]


def test_page_has_observation_tab(make_client):
    html = make_client().get("/").text
    for text in ("Observation", "Start observation", "API keys", "Stop"):
        assert text in html


def test_summary_cards_only_use_the_current_batch(make_client):
    c = make_client()
    tasks = "first task\nsecond task"
    c.post("/dashboard/observe/start", json={"tasks": tasks, "models": ["claude", "gpt"], "variance_runs": 2,
                                             "cross_runs": 2})
    st = _wait(c)
    assert set(st["summary"]["cheapest"].values()) <= {"claude", "gpt"} and st["summary"]["cheapest"]
    # second batch: same tasks, claude only; old gpt rows stay in the store but must not drive the cards
    c.post("/dashboard/observe/start", json={"tasks": tasks, "models": ["claude"], "variance_runs": 3,
                                             "cross_runs": 2})
    st = _wait(c)
    assert st["summary"]["cheapest"] == {}                 # one model: nothing to compare
    assert "gpt" in {r["model"] for r in st["rows"]}       # history still visible in the table
    n_claude = sum(1 for r in st["rows"] if r["model"] == "claude" and r["status"] == "ok")
    assert st["summary"]["n_ok"] == n_claude
