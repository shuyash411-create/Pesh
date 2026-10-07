import json

import httpx
import pytest

from pesh.cli import main
from pesh.config import QuoteConfig
from pesh.observe import (ObservationLoop, ObservationStore, ProviderError, SimProvider, Task, build_providers,
                          build_report, load_tasks, plan_jobs, resolve_models)
from pesh.observe.providers import AnthropicProvider, OpenAIProvider
from pesh.quote.engine import QuoteEngine

KEY = "sk-secret-123"


@pytest.fixture(scope="module")
def engine_bytes(sim_small):
    import pickle
    train, calib, _ = sim_small
    return pickle.dumps(QuoteEngine(QuoteConfig()).fit(train, calib))


@pytest.fixture
def engine(engine_bytes):
    import pickle
    return pickle.loads(engine_bytes)


def _tasks(n=3):
    return [Task(f"t{i}", f"task {i}", {"x_repo": 0.1 * i, "x_complex": 0.2 * i, "human_label": i % 3})
            for i in range(n)]


def test_plan_counts():
    models = resolve_models(["claude", "gpt", "together-open"])
    jobs = plan_jobs(_tasks(1)[0], models, r_variance=10, r_cross=3)
    assert len(jobs) == 3 * 3 + 10
    assert {j.model.alias for j in jobs if j.kind == "variance"} == {"claude"}
    assert len({j.key for j in jobs}) == len(jobs)


def test_loop_offline_resume_and_report(engine, tmp_path):
    models = resolve_models(["claude", "gpt"])
    store = ObservationStore(tmp_path / "o.sqlite")
    loop = ObservationLoop(engine, _tasks(3), models, {}, store, r_variance=4, r_cross=2, batch_size=2,
                           report_dir=str(tmp_path / "out"))
    rep = loop.run()
    assert store.count() == 3 * (2 * 2 + 4)
    assert rep["rho_tau"] and 0 <= rep["rho_tau"]["rho"] <= 1
    assert 0 <= rep["coverage"]["realised"] <= 1
    assert (tmp_path / "out" / "report.md").exists() and (tmp_path / "out" / "observations.parquet").exists()
    df = store.frame()
    assert (df["within_ceiling"] == (df["actual_cost"] <= df["predicted_ceiling"])).all()
    assert set(df["provider"]) == {"sim"}

    # restart: nothing re-run, ACI state rebuilt from the store
    n_updates = engine.aci_for("claude").n_updates
    loop2 = ObservationLoop(engine, _tasks(3), models, {}, store, r_variance=4, r_cross=2)
    loop2.run()
    assert store.count() == 3 * 8
    assert engine.aci_for("claude").n_updates > n_updates      # replayed, not re-run


def test_stop_then_resume(engine, tmp_path):
    store = ObservationStore(tmp_path / "o.sqlite")
    models = resolve_models(["claude"])
    seen = []

    def on_event(kind, info):
        if kind == "run":
            seen.append(1)
            if len(seen) == 5:
                loop.stop(wait=False)

    loop = ObservationLoop(engine, _tasks(2), models, {}, store, r_variance=3, r_cross=1, on_event=on_event)
    loop.run()
    assert store.count() == 5
    loop._stop.clear()
    loop.run()
    assert store.count() == 2 * 4


def test_errors_are_logged_and_retried(engine, tmp_path):
    class Flaky(SimProvider):
        calls = 0

        def run(self, model, task, run_idx=0, kind="cross"):
            Flaky.calls += 1
            if Flaky.calls == 2:
                raise ProviderError("boom")
            return super().run(model, task, run_idx, kind)

    store = ObservationStore(tmp_path / "o.sqlite")
    models = resolve_models(["claude"])
    loop = ObservationLoop(engine, _tasks(1), models, {"anthropic": Flaky()}, store, r_variance=0, r_cross=3)
    loop.run()
    df = store.frame()
    assert list(df["status"]) == ["ok", "error", "ok"] and df.loc[1, "error"] == "boom"
    assert build_report(df)["n_errors"] == 1
    loop.run()                                  # the errored run is retried, ok ones are not
    assert list(store.frame().sort_values("run_idx")["status"]) == ["ok"] * 3


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_anthropic_adapter_cost_and_key_safety():
    seen = {}

    def handler(req):
        seen["key"] = req.headers["x-api-key"]
        return httpx.Response(200, json={"content": [{"type": "text", "text": "hi"}],
                                         "usage": {"input_tokens": 1000, "output_tokens": 200}})

    p = AnthropicProvider(env={"ANTHROPIC_API_KEY": KEY}, client=_client(handler))
    m = resolve_models(["claude"])[0]
    r = p.run(m, Task("a", "hello", {}))
    assert seen["key"] == KEY and KEY not in repr(p)
    assert r.input_tokens == 1000 and r.output_tokens == 200 and r.success and r.steps == 1
    assert r.cost == pytest.approx(1000 * 1e-6 + 200 * 5e-6)


def test_http_error_scrubs_key():
    p = OpenAIProvider(env={"OPENAI_API_KEY": KEY},
                       client=_client(lambda req: httpx.Response(401, text=f"bad key {KEY}")), retries=0)
    with pytest.raises(ProviderError) as e:
        p.run(resolve_models(["gpt"])[0], Task("a", "x", {}))
    assert KEY not in str(e.value) and "401" in str(e.value)


def test_missing_key_provider_skipped():
    assert build_providers(env={}) == {}
    assert set(build_providers(env={"OPENAI_API_KEY": KEY, "TOGETHER_API_KEY": ""})) == {"openai"}


def test_load_tasks_features():
    ts = load_tasks("tasks.jsonl")
    assert len(ts) >= 3 and all({"x_repo", "x_complex", "human_label"} <= set(t.features) for t in ts)
    assert ts[-1].features["human_label"] == 2           # explicit features override the heuristic


def test_cli_observe_simulated(engine, tmp_path, capsys):
    path = engine.save(tmp_path / "engine.pkl")
    main(["observe", "--tasks", "tasks.jsonl", "--models", "claude,gpt", "--variance", "3", "--cross", "2",
          "--engine", str(path), "--db", str(tmp_path / "o.sqlite"), "--out", str(tmp_path / "rep"), "--quiet"])
    assert "rho_tau" in capsys.readouterr().out
    rows = ObservationStore(tmp_path / "o.sqlite").count()
    assert rows == len(load_tasks("tasks.jsonl")) * (2 * 2 + 3)
    assert json.loads((tmp_path / "rep" / "report.json").read_text())["n_ok"] == rows
