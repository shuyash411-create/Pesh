"""Command-line interface: pesh simulate | calibrate | train | evaluate | ingest-openhands | h1h2 |
experiments | quote | serve | dashboard | observe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ControllerConfig, QuoteConfig


def _csv(s: str | None) -> list[str]:
    return [x for x in (s or "").split(",") if x]


def cmd_simulate(a):
    from .data.io import save_runs
    from .sim.dgp import simulate

    df = simulate(a.tasks, a.runs, seed=a.seed, verbosity=a.verbosity, step_mult=a.step_mult)
    df["model_version"] = a.model_version
    path = save_runs(df, a.out)
    print(f"wrote {len(df)} runs of {a.tasks} tasks to {path}  (mean cost ${df['cost'].mean():.2f}, "
          f"success {df['success'].mean():.3f})")


def cmd_calibrate(a):
    from .sim.calibrate import FREE, calibrate

    best, final, res = calibrate(n_tasks=a.tasks, maxiter=a.maxiter, verbose=a.verbose)
    out = {"distance": float(res.fun), "params": {k: float(getattr(best, k)) for k in FREE},
           "moments": {k: float(v) for k, v in final.items()}}
    print(json.dumps(out, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps({**asdict(best), "theta": list(best.theta)}, indent=2))


def _hypothesis_report(runs: pd.DataFrame) -> dict:
    """H1 (ICC ceiling) and H2 (tail indices) on any run log."""
    from .econometrics.icc import icc_anova
    from .econometrics.tails import hill_range, powerlaw_fit

    out = {}
    counts = runs.groupby("task_id").size()
    if (counts > 1).sum() >= 10:
        multi = runs[runs["task_id"].isin(counts[counts > 1].index)]
        r = icc_anova(np.log(multi["cost"]), multi["task_id"])
        out["H1_predictability"] = r.to_dict()
    if len(runs) >= 500:
        c = runs["cost"].to_numpy(float)
        out["H2_tails"] = {"alpha_cost_hill_1-2pct": hill_range(c), "powerlaw_cost": powerlaw_fit(c)}
        if runs["steps"].notna().all():
            out["H2_tails"]["alpha_steps_hill_1-2pct"] = hill_range(runs["steps"].to_numpy(float))
    return out


def cmd_train(a):
    from .control.feasibility import FeasibilityModel, train_from_simulation
    from .data.adapters.generic import feasibility_frame, load_steps
    from .data.io import load_runs, one_run_per_task, split_by_task
    from .quote.engine import QuoteEngine

    runs = load_runs(a.logs)
    if a.model_version:
        runs = runs[runs["model_version"] == a.model_version]
    cfg = QuoteConfig(alpha=a.alpha, numeric_features=_csv(a.numeric), categorical_features=_csv(a.categorical),
                      quadratic=not a.linear_only, mondrian_by=a.mondrian_by)
    train, calib, test = split_by_task(runs, (0.5, 0.25, 0.25), seed=a.seed)
    # one run per task keeps calibration scores exchangeable (runs of one task are correlated)
    engine = QuoteEngine(cfg, model=a.model).fit(train, one_run_per_task(calib, seed=a.seed))
    strata = [s for s in ["human_label", "difficulty", "loop", a.mondrian_by] if s and s in test]
    ev = engine.evaluate(test, strata=strata)
    engine.card["hypotheses"] = _hypothesis_report(runs)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) / (a.model_version or "all") / stamp
    out.mkdir(parents=True, exist_ok=True)
    engine.save(out / "engine.pkl")

    if a.steps:
        fm = FeasibilityModel().fit(feasibility_frame(load_steps(a.steps), runs, seed=a.seed))
    else:
        fm = train_from_simulation(seed=a.seed + 101)
    fm.save(out / "feasibility.pkl")
    engine.card["feasibility"] = fm.metrics
    (out / "card.json").write_text(json.dumps(engine.card, indent=2, default=float))
    print(json.dumps({"artifact": str(out), "evaluation": ev, "feasibility": fm.metrics,
                      "hypotheses": engine.card["hypotheses"]}, indent=2, default=float))


def cmd_evaluate(a):
    from .data.io import load_runs
    from .quote.engine import QuoteEngine

    runs = load_runs(a.logs)
    out = {}
    if a.engine:
        engine = QuoteEngine.load(a.engine)
        strata = [s for s in ["human_label", "difficulty", "loop"] if s in runs]
        out["quotes"] = engine.evaluate(runs, strata=strata)
    out.update(_hypothesis_report(runs))
    print(json.dumps(out, indent=2, default=float))


def cmd_experiments(a):
    from .eval.experiments import run_all
    from .eval.report import write_report

    res = run_all(scale=a.scale)
    path = write_report(res, a.out, figures=not a.no_figures)
    print(f"report written to {path}")


def cmd_quote(a):
    from .quote.engine import QuoteEngine

    engine = QuoteEngine.load(a.engine)
    feats = json.loads(a.features)
    print(engine.quote(pd.DataFrame([feats]), alpha=a.alpha).iloc[0].to_json(indent=2))


def cmd_serve(a):
    import uvicorn

    from .control.feasibility import FeasibilityModel
    from .quote.engine import QuoteEngine
    from .service.api import create_app

    engine = QuoteEngine.load(a.engine)
    fm = FeasibilityModel.load(a.feasibility) if a.feasibility else None
    app = create_app(engine, fm, ControllerConfig(), headroom=a.headroom, exploration_rate=a.exploration,
                     db_path=a.db)
    uvicorn.run(app, host=a.host, port=a.port)


def cmd_ingest_openhands(a):
    from .data.adapters.openhands_eval import load_eval_outputs
    from .data.io import save_runs

    runs, steps = load_eval_outputs(a.inputs, a.report)
    save_runs(runs, a.out)
    if a.steps_out:
        save_runs(steps, a.steps_out)
    by_model = runs.groupby("model_version").agg(runs=("cost", "size"), tasks=("instance_id", "nunique"),
                                                 mean_cost=("cost", "mean"), resolved=("success", "mean"))
    print(f"wrote {len(runs)} runs to {a.out} ({runs.attrs.get('dropped', 0)} records without token usage "
          f"dropped; cost source: {runs['cost_source'].value_counts().to_dict()})")
    print(by_model.round(3).to_string())


def cmd_h1h2(a):
    from .data.io import load_runs
    from .econometrics.inference import h1h2_report
    from .eval.h1h2 import verdicts, write_h1h2_report

    runs = load_runs(a.logs)
    rep = h1h2_report(runs, by=a.by, n_boot=a.boot, fraction=a.fraction)
    path = write_h1h2_report(rep, a.out, source=str(a.logs))
    for name, row in rep.iterrows():
        print(f"[{name}]")
        for v in verdicts(row):
            print("  " + v)
    print(f"report written to {path}")


def _sample_model(folder: Path):
    """Quote engine + feasibility model trained offline on simulated runs; cached in `folder`."""
    from .control.feasibility import FeasibilityModel, train_from_simulation
    from .data.io import split_by_task
    from .quote.engine import QuoteEngine
    from .sim.dgp import simulate

    eng_path, fm_path = folder / "engine.pkl", folder / "feasibility.pkl"
    if eng_path.exists() and fm_path.exists():
        return QuoteEngine.load(eng_path), FeasibilityModel.load(fm_path)
    print("Preparing a sample model from simulated runs (first launch only, about 20 seconds)...", flush=True)
    train, calib = split_by_task(simulate(3000, 1, seed=0), (0.67, 0.33), seed=0)
    engine = QuoteEngine(QuoteConfig()).fit(train, calib)
    fm = train_from_simulation(n_tasks=800, seed=101)
    folder.mkdir(parents=True, exist_ok=True)
    engine.save(eng_path)
    fm.save(fm_path)
    return engine, fm


def cmd_dashboard(a):
    import threading
    import webbrowser

    import uvicorn

    from .control.feasibility import FeasibilityModel
    from .dashboard import create_dashboard_app
    from .quote.engine import QuoteEngine

    if a.engine:
        engine = QuoteEngine.load(a.engine)
        fm = FeasibilityModel.load(a.feasibility) if a.feasibility else None
    else:
        engine, fm = _sample_model(Path(a.sample_dir))
    app = create_dashboard_app(engine, fm, db_path=a.db, drift_min_n=a.drift_min_n, headroom=a.headroom,
                               controller_config=ControllerConfig())
    url = f"http://{a.host}:{a.port}/"
    print(f"Pesh dashboard running at {url}  (press Ctrl+C to stop)", flush=True)
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def cmd_observe(a):
    from .observe.cmd import run

    code = run(a, lambda: _sample_model(Path(a.sample_dir))[0])
    if code:
        sys.exit(code)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pesh", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate run logs from the calibrated DGP")
    s.add_argument("--tasks", type=int, default=3000)
    s.add_argument("--runs", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--verbosity", type=float, default=1.0)
    s.add_argument("--step-mult", type=float, default=1.0)
    s.add_argument("--model-version", default="sim-v1")
    s.add_argument("--out", default="data/sim.parquet")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("calibrate", help="SMM-calibrate the DGP to published moments")
    s.add_argument("--tasks", type=int, default=1000)
    s.add_argument("--maxiter", type=int, default=300)
    s.add_argument("--verbose", action="store_true")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("train", help="fit quote engine + feasibility model on run logs")
    s.add_argument("--logs", required=True)
    s.add_argument("--steps", help="optional step-level log for the feasibility model")
    s.add_argument("--out", default="artifacts")
    s.add_argument("--model-version")
    s.add_argument("--alpha", type=float, default=0.10)
    s.add_argument("--numeric", default="x_repo,x_complex")
    s.add_argument("--categorical", default="human_label")
    s.add_argument("--mondrian-by")
    s.add_argument("--model", choices=["linear", "gbm"], default="linear")
    s.add_argument("--linear-only", action="store_true", help="no quadratic/interaction terms")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("evaluate", help="coverage of a trained engine and H1/H2 diagnostics on logs")
    s.add_argument("--logs", required=True)
    s.add_argument("--engine")
    s.set_defaults(fn=cmd_evaluate)

    s = sub.add_parser("ingest-openhands", help="convert OpenHands SWE-bench output.jsonl files into a run log")
    s.add_argument("inputs", nargs="+", help="output.jsonl files, or folders searched for output.jsonl")
    s.add_argument("--report", nargs="*", help="SWE-bench harness report.json file(s) with resolved_ids")
    s.add_argument("--out", default="data/openhands_runs.parquet")
    s.add_argument("--steps-out", help="also write the per-call step log here")
    s.set_defaults(fn=cmd_ingest_openhands)

    s = sub.add_parser("h1h2", help="M1: predictability ceiling and tail indices with confidence intervals")
    s.add_argument("--logs", required=True)
    s.add_argument("--by", default="model_version", help="column that identifies the agent configuration")
    s.add_argument("--boot", type=int, default=1000, help="bootstrap replicates")
    s.add_argument("--fraction", type=float, default=0.02, help="tail share used by the Hill estimator")
    s.add_argument("--out", default="reports/m1")
    s.set_defaults(fn=cmd_h1h2)

    s = sub.add_parser("experiments", help="reproduce the paper's E0-E5 and write a report")
    s.add_argument("--out", default="reports")
    s.add_argument("--scale", type=float, default=1.0)
    s.add_argument("--no-figures", action="store_true")
    s.set_defaults(fn=cmd_experiments)

    s = sub.add_parser("quote", help="quote one task from a JSON feature dict")
    s.add_argument("--engine", required=True)
    s.add_argument("--features", required=True)
    s.add_argument("--alpha", type=float)
    s.set_defaults(fn=cmd_quote)

    s = sub.add_parser("serve", help="run the HTTP quote / enforcement service")
    s.add_argument("--engine", required=True)
    s.add_argument("--feasibility")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--headroom", type=float, default=1.0)
    s.add_argument("--exploration", type=float, default=0.05)
    s.add_argument("--db")
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("dashboard", help="open the local web dashboard (analyze logs, quote, live monitor)")
    s.add_argument("--engine", help="trained engine.pkl; default: a sample model trained on simulated runs")
    s.add_argument("--feasibility", help="feasibility.pkl to go with --engine")
    s.add_argument("--sample-dir", default="artifacts/dashboard-sample", help="where the sample model is cached")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8050)
    s.add_argument("--db", default="pesh_dashboard.sqlite", help="SQLite file for finished runs")
    s.add_argument("--headroom", type=float, default=1.0, help="ceiling = headroom x quote")
    s.add_argument("--drift-min-n", type=int, default=20, help="finished runs before the drift alarm can fire")
    s.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    s.set_defaults(fn=cmd_dashboard)

    s = sub.add_parser("observe", help="observation loop: run tasks through model APIs, compare PESH's "
                                       "predicted cost with the actual cost (cost only)")
    s.add_argument("--tasks", required=True, help="JSONL rows {task_id, text, [features]}")
    s.add_argument("--models", default="claude,gpt,together-open,fireworks-open",
                   help="comma list of aliases (claude, gpt, together-open, fireworks-open) or provider:model-id; "
                        "the first is the primary model for variance runs")
    s.add_argument("--variance", type=int, default=10, help="same task, primary model, this many runs")
    s.add_argument("--cross", type=int, default=3, help="same task, every model, this many runs each")
    s.add_argument("--engine", help="trained engine.pkl; default: sample model trained on simulated runs")
    s.add_argument("--sample-dir", default="artifacts/dashboard-sample")
    s.add_argument("--db", default="pesh_dashboard.sqlite", help="SQLite file holding the observations table")
    s.add_argument("--out", default="reports/observe", help="where report.md/json and CSV/Parquet exports go")
    s.add_argument("--batch-size", type=int, default=10, help="tasks per batch; a report is printed after each")
    s.add_argument("--max-tokens", type=int, default=512, help="max output tokens per API call")
    s.add_argument("--max-spend", type=float, help="stop once this much USD has been spent in this session")
    s.add_argument("--prices", help="JSON file of per-million-token price overrides")
    s.add_argument("--simulate", action="store_true", help="force the offline simulator even if keys are set")
    s.add_argument("--quiet", action="store_true", help="don't print every run")
    s.set_defaults(fn=cmd_observe)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
