"""M1 pipeline: OpenHands eval outputs -> run log -> H1/H2 with confidence intervals."""

import json

import numpy as np
import pytest

from pesh.cli import main
from pesh.config import PricingParams
from pesh.cost_model import run_cost
from pesh.data.adapters.openhands_eval import load_eval_outputs
from pesh.econometrics.icc import icc_anova
from pesh.econometrics.inference import h1h2_report, icc_ci, tail_ci
from pesh.eval.h1h2 import verdicts, write_h1h2_report
from pesh.sim.dgp import plan_runs, simulate_tasks

MODELS = ("anthropic/model-a", "openai/model-b")


def write_openhands(root, n_tasks=80, runs=4):
    """Simulated runs written as OpenHands SWE-bench output.jsonl, one file per model x run.

    model-a carries per-call usage in metrics.token_usages with resolved flags in each record;
    model-b carries cumulative llm_metrics on history events with a per-run report.json.
    Returns the true per-run costs keyed by (model, instance, run).
    """
    pr = PricingParams()
    tasks = simulate_tasks(n_tasks, seed=1)
    truth = {}
    for m_i, model in enumerate(MODELS):
        plan = plan_runs(tasks, runs, seed=10 + m_i)
        rng = np.random.default_rng(m_i)
        per_run = {r: [] for r in range(runs)}
        resolved = {r: [] for r in range(runs)}
        for j in range(plan.n):
            T = int(plan.t_total[j])
            c0 = float(plan.c0[j])
            a = np.exp(rng.normal(7.2, 0.9, T))
            o = np.exp(rng.normal(6.2, 0.6, T))
            ctx = c0 + np.concatenate([[0.0], np.cumsum(a[:-1])])
            new = np.concatenate([[c0], a[:-1]])
            cost = run_cost(c0, a, o, pr)
            inst = f"repo{plan.task_idx[j] % 5}__pkg-{plan.task_idx[j]}"
            run = int(plan.run_id[j])
            ok = bool(plan.u_success[j] < plan.p_success[j])
            truth[(model, inst, run)] = cost
            usages = [{"prompt_tokens": float(c), "completion_tokens": float(oo), "cache_read_tokens": float(c - n),
                       "cache_write_tokens": float(n)} for c, oo, n in zip(ctx, o, new)]
            rec = {"instance_id": inst, "metadata": {"llm_config": {"model": model}},
                   "instance": {"repo": f"org/repo{plan.task_idx[j] % 5}",
                                "problem_statement": "x" * (200 + 10 * int(plan.task_idx[j])),
                                "patch": "--- a\n+++ b\n-old\n+new\n+more\n", "FAIL_TO_PASS": '["t1", "t2"]'}}
            if m_i == 0:
                rec["metrics"] = {"accumulated_cost": cost, "token_usages": usages}
                rec["report"] = {"resolved": ok}
            else:
                # no accumulated_cost here, so the loader must recompute cost from tokens
                hist = []
                acc = {"prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
                for u in usages:
                    hist.append({"action": "read", "args": {"path": "setup.py"}})
                    acc = {k: acc[k] + u[k] for k in acc}
                    hist.append({"action": "message", "llm_metrics": {"accumulated_cost": 0.0,
                                                                      "accumulated_token_usage": dict(acc)}})
                rec["history"] = hist
                if ok:
                    resolved[run].append(inst)
            per_run[run].append(rec)
        for r, recs in per_run.items():
            d = root / model.split("/")[1] / f"run_{r + 1}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "output.jsonl").write_text("\n".join(json.dumps(x) for x in recs) + "\n")
            if m_i == 1:
                (d / "report.json").write_text(json.dumps({"resolved_ids": resolved[r]}))
    return truth


@pytest.fixture(scope="module")
def oh(tmp_path_factory):
    root = tmp_path_factory.mktemp("openhands")
    truth = write_openhands(root)
    runs, steps = load_eval_outputs([root])
    return root, truth, runs, steps


def test_ingest_both_formats_recovers_costs_and_outcomes(oh):
    _, truth, runs, steps = oh
    assert set(runs["model_version"]) == set(MODELS)
    assert len(runs) == len(truth) == 2 * 80 * 4
    assert runs.groupby("model_version")["instance_id"].nunique().eq(80).all()
    got = {(r.model_version, r.instance_id, r.run_id - 1): r.cost for r in runs.itertuples()}
    err = [abs(got[k] / v - 1) for k, v in truth.items()]
    assert max(err) < 1e-6                                   # reported or recomputed, cost is exact
    assert set(runs.loc[runs.model_version == MODELS[0], "cost_source"]) == {"reported"}
    assert set(runs.loc[runs.model_version == MODELS[1], "cost_source"]) == {"tokens"}
    assert 0.3 < runs["success"].mean() < 0.9
    assert (runs["proxy_gold_patch_lines"] == 3).all() and (runs["proxy_fail_to_pass"] == 2).all()
    assert steps["task_id"].str.contains("::").all() and len(steps) == runs["steps"].sum()


def test_h1h2_matches_point_estimates_and_has_intervals(oh):
    _, _, runs, _ = oh
    rep = h1h2_report(runs, n_boot=300)
    assert list(rep.index) == [*sorted(MODELS), "all"]
    for name, row in rep.iterrows():
        g = runs if name == "all" else runs[runs.model_version == name]
        assert row["rho"] == pytest.approx(icc_anova(np.log(g["cost"]), g["task_id"]).rho)
        assert row["rho_lo"] <= row["rho"] <= row["rho_hi"]
        assert row["alpha_cost_lo"] <= row["alpha_cost"] <= row["alpha_cost_hi"]
        assert row["variance_verdict"] in ("infinite", "finite", "inconclusive")
        assert verdicts(row)


def test_icc_interval_has_nominal_coverage():
    rng = np.random.default_rng(3)
    hits = 0
    for r in range(60):
        y = np.repeat(rng.normal(0, 1, 300), 4) + rng.normal(0, 0.5, 1200)
        c = icc_ci(y, np.repeat(np.arange(300), 4), n_boot=300, seed=r)
        hits += c["rho_lo"] <= 0.8 <= c["rho_hi"]
    assert hits / 60 >= 0.88


def test_tail_interval_covers_pareto_index():
    rng = np.random.default_rng(4)
    x = (1 - rng.random(20_000)) ** (-1 / 1.5)
    t = tail_ci(x, x**2, n_boot=200)
    assert t["alpha_cost_lo"] <= 1.5 <= t["alpha_cost_hi"]
    assert t["variance_verdict"] == "infinite"
    assert t["ratio"] == pytest.approx(2.0, rel=0.05)       # steps = cost^2 halves the index


def test_cli_ingest_and_report(oh, tmp_path, capsys):
    root = oh[0]
    out = tmp_path / "runs.parquet"
    main(["ingest-openhands", str(root), "--out", str(out), "--steps-out", str(tmp_path / "steps.parquet")])
    main(["h1h2", "--logs", str(out), "--boot", "200", "--out", str(tmp_path / "m1")])
    text = capsys.readouterr().out
    assert "wrote 640 runs" in text and "H1:" in text and "H2:" in text
    md = (tmp_path / "m1" / "h1h2.md").read_text()
    assert "| all |" in md and "Verdicts" in md


def test_write_report_handles_missing_sections(tmp_path):
    import pandas as pd

    rep = pd.DataFrame([{"n_runs": 30, "n_tasks": 30, "mean_cost": 1.0, "median_cost": 1.0, "success": 0.5}],
                       index=pd.Index(["tiny"], name="group"))
    path = write_h1h2_report(rep, tmp_path)
    assert "not enough data" in path.read_text()


def test_step_cap_ties_are_excluded_not_divided_by_zero():
    rng = np.random.default_rng(5)
    # max-iterations cap at 15 binds for ~6% of runs: more than the 2% tail the Hill estimator uses
    steps = np.minimum(np.floor(5 * (1 - rng.random(8000)) ** (-1 / 2.5)), 15)
    cost = steps**2 * rng.lognormal(0, 0.1, 8000)
    with np.errstate(all="raise"):
        t = tail_ci(cost, steps, n_boot=100)
    assert 0.04 < t["steps_at_cap_share"] < 0.09
    assert np.isfinite(t["alpha_steps"]) and np.isfinite(t["alpha_steps_hi"])
