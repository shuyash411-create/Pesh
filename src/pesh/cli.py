"""Command-line interface: pesh simulate | calibrate | train | evaluate | experiments | quote | serve."""

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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
