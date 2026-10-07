"""Local dashboard: a thin presentation layer over the existing service and engine.

`create_dashboard_app` builds the normal service app (`service.api.create_app`) and adds
a single HTML page plus a few read-mostly `/dashboard/*` helpers. All numbers come from
existing code paths: the CLI's evaluate report, `QuoteEngine.quote`, the pooling
calculation in `econometrics.risk`, and the calibrated simulator for demo runs.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from ..config import ControllerConfig, DGPParams
from ..control.feasibility import FeasibilityModel
from ..econometrics.risk import pooling_loading
from ..quote.engine import QuoteEngine
from ..schema import SchemaError, validate_runs
from ..service.api import create_app
from .observe_routes import add_observation_routes, observations_as_runs

STATIC = Path(__file__).parent / "static"
MAX_RUNS_PER_MONTH = 20_000

# Plain-language labels for the simulator's features; anything else shows its column name.
FEATURE_LABELS = {
    "x_repo": {"label": "Repository size",
               "help": "How big the codebase is compared with a typical task (0 = typical, +2 = very large)."},
    "x_complex": {"label": "Issue complexity",
                  "help": "How involved the requested change is (0 = typical, +2 = very complex)."},
    "human_label": {"label": "How long a person would take",
                    "help": "A rough human estimate of the task's size.",
                    "options": {"0": "Under 15 minutes", "1": "15 minutes to 1 hour", "2": "More than 1 hour"}},
}


def _money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def _read_upload(raw: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    try:
        if name.endswith((".parquet", ".pq")):
            df = pd.read_parquet(io.BytesIO(raw))
        elif name.endswith(".jsonl"):
            df = pd.read_json(io.BytesIO(raw), lines=True)
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - any parse failure becomes a readable message
        raise HTTPException(422, f"Couldn't read this file as CSV or Parquet ({type(exc).__name__}).") from exc
    try:
        return validate_runs(df)
    except SchemaError as exc:
        raise HTTPException(422, f"This file doesn't look like a run log: {exc}.") from exc


def analyze_runs(runs: pd.DataFrame, engine: QuoteEngine, runs_per_month: int, source: str) -> dict:
    """The existing evaluate path (H1/H2 report + quote check) plus a monthly forecast."""
    from ..cli import _hypothesis_report

    rep = _hypothesis_report(runs)
    cost = runs["cost"].to_numpy(float)
    out: dict = {"source": source, "n_runs": int(len(runs)), "n_tasks": int(runs["task_id"].nunique()),
                 "mean_cost": float(cost.mean())}

    h1 = rep.get("H1_predictability")
    if h1:
        rho = float(h1["rho"])
        verdict = "high" if rho >= 0.7 else "some" if rho >= 0.4 else "low"
        lead = {"high": "Cost is mostly decided by the task itself, so a pre-run quote can be accurate.",
                "some": "The task explains part of the cost; quotes will be useful but wide.",
                "low": "Repeat runs of the same task differ a lot; rely on ceilings more than quotes."}[verdict]
        out["predictability"] = {
            "value": rho, "verdict": verdict,
            "text": f"{lead} The other {1 - rho:.0%} is run-to-run luck that no model can remove."}
    else:
        out["predictability"] = {"value": None, "reason": "Needs at least 10 tasks that were run more than once."}

    h2 = rep.get("H2_tails")
    if h2:
        lo, hi = h2["alpha_cost_hill_1-2pct"]
        alpha = (lo + hi) / 2
        if alpha < 1:
            text = ("The most expensive runs are so extreme that even the average cost is unreliable. "
                    "A hard ceiling is essential.")
        elif alpha < 2:
            text = ("Infinite variance: rare runs are expensive enough that averages swing wildly. "
                    "Use a hard ceiling rather than trusting the average.")
        else:
            text = "Expensive runs happen, but they are not extreme enough to make averages unreliable."
        out["tail"] = {"value": alpha, "range": [lo, hi], "infinite_variance": alpha < 2,
                       "infinite_mean": alpha < 1, "text": text}
    else:
        out["tail"] = {"value": None, "reason": "Needs at least 500 runs to judge the expensive tail."}

    n = int(min(max(runs_per_month, 1), MAX_RUNS_PER_MONTH))
    loading = pooling_loading(cost, [n], eps=0.10, reps=2000, seed=0)[n]
    expected = n * cost.mean()
    p90 = expected * (1 + loading)
    out["forecast"] = {"runs_per_month": n, "expected": float(expected), "p90": float(p90),
                       "text": f"Expected monthly spend ~{_money(expected)}, 90% of months below {_money(p90)}."}

    needed = engine.config.numeric_features + engine.config.categorical_features
    uncapped = runs[~runs["capped"]]
    if all(c in runs for c in needed) and len(uncapped) >= 50:
        q = engine.quote(uncapped)["quote"].to_numpy()
        cov = float(np.mean(uncapped["cost"].to_numpy(float) <= q))
        target = 1.0 - engine.config.alpha
        out["quote_check"] = {"coverage": cov, "target": target,
                              "text": f"{cov:.0%} of these runs came in under our quote (target {target:.0%})."}
    else:
        out["quote_check"] = None

    pos = cost[cost > 0]
    edges = np.logspace(np.log10(pos.min()), np.log10(pos.max()), 31)
    edges[0], edges[-1] = pos.min(), pos.max()      # exact ends so rounding can't drop a run
    counts, _ = np.histogram(pos, bins=edges)
    out["histogram"] = {"edges": edges.tolist(), "counts": counts.tolist(),
                        "median": float(np.median(pos)), "p90": float(np.quantile(pos, 0.9)),
                        "p99": float(np.quantile(pos, 0.99))}
    return out


def demo_trajectory(params: DGPParams, seed: int, drift: bool, features: list[str]) -> dict:
    """One simulated run's per-step token draws, mirroring `sim.dgp.execute`."""
    from ..sim.dgp import plan_runs, simulate_tasks

    tasks = simulate_tasks(1, params, seed)
    plan = plan_runs(tasks, 1, params, seed, verbosity=1.35 if drift else 1.0, step_mult=1.25 if drift else 1.0)
    rng = np.random.default_rng([seed, 99])
    T = int(plan.t_total[0])
    a_mu = params.a_log_mean - 0.5 * params.a_log_sd**2
    o_mu = params.o_log_mean - 0.5 * params.o_log_sd**2
    scale = float(plan.tok_scale[0])
    looping = np.arange(1, T + 1) > plan.t_base[0]
    steps = [{"appended": float(np.exp(rng.normal(a_mu, params.a_log_sd)) * scale),
              "output": float(np.exp(rng.normal(o_mu, params.o_log_sd)) * scale),
              "repeated_views": float(rng.poisson(params.loop_repeat_rate if lp else params.base_repeat_rate))}
             for lp in looping]
    row = tasks.iloc[0]
    feats = {c: (int(row[c]) if c == "human_label" else float(row[c])) for c in features}
    success = bool(not plan.truncated[0] and plan.u_success[0] < plan.p_success[0])
    return {"seed": seed, "drift": drift, "features": feats, "initial_context": float(plan.c0[0]),
            "steps": steps, "success": success}


def create_dashboard_app(engine: QuoteEngine, feasibility: FeasibilityModel | None = None, *,
                         db_path: str | None = None, drift_min_n: int = 20,
                         params: DGPParams | None = None, env_path: str | Path = ".env",
                         observe_report_dir: str = "reports/observe", sim_pause: float = 0.25,
                         **service_kw) -> FastAPI:
    app = create_app(engine, feasibility, db_path=db_path, drift_min_n=drift_min_n, **service_kw)
    # the Observation tab writes to the same SQLite file the service uses (a temp file if none was given)
    obs_db = db_path or str(Path(tempfile.mkdtemp(prefix="pesh-")) / "pesh_dashboard.sqlite")
    add_observation_routes(app, engine, obs_db, env_path, observe_report_dir, sim_pause)
    params = params or DGPParams()
    controller = service_kw.get("controller_config") or ControllerConfig()
    cache: dict = {}
    counter = {"seed": 10_000}

    def sample_runs() -> pd.DataFrame:
        if "sample" not in cache:
            from ..sim.dgp import simulate

            cache["sample"] = validate_runs(simulate(1500, 4, params, seed=7))
        return cache["sample"]

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html", media_type="text/html")

    @app.post("/dashboard/analyze")
    async def analyze(request: Request, filename: str = "upload.csv", sample: bool = False,
                      runs_per_month: int = 2000, source: str = ""):
        if source == "observations":
            obs = app.state.observation_store.frame()
            runs = observations_as_runs(obs)
            if runs.empty:
                raise HTTPException(422, "No finished observation runs yet. Start one on the Observation tab.")
            return analyze_runs(validate_runs(runs), engine, runs_per_month,
                                f"Your observation runs ({len(runs)} finished)")
        if sample:
            runs, source = sample_runs(), "Sample data (simulated agent runs)"
        else:
            raw = await request.body()
            if not raw:
                raise HTTPException(422, "The file is empty.")
            runs, source = _read_upload(raw, filename), filename
        return analyze_runs(runs, engine, runs_per_month, source)

    @app.get("/dashboard/features")
    def features():
        out = []
        for c in engine.config.numeric_features:
            lo, hi = engine.design.ranges_.get(c, (-3.0, 3.0))
            meta = FEATURE_LABELS.get(c, {})
            out.append({"name": c, "kind": "number", "label": meta.get("label", c), "help": meta.get("help", ""),
                        "min": lo, "max": hi, "default": 0.0 if lo <= 0 <= hi else (lo + hi) / 2})
        for c in engine.config.categorical_features:
            meta = FEATURE_LABELS.get(c, {})
            names = meta.get("options", {})
            levels = engine.design.levels_.get(c, [])
            out.append({"name": c, "kind": "choice", "label": meta.get("label", c), "help": meta.get("help", ""),
                        "options": [{"value": v, "label": names.get(v, v)} for v in levels],
                        "default": levels[len(levels) // 2] if levels else ""})
        return {"features": out, "coverage_level": 1.0 - engine.config.alpha, "drift_min_n": drift_min_n,
                "compact_to": controller.compact_to, "max_runs_per_month": MAX_RUNS_PER_MONTH}

    @app.get("/dashboard/sample-task")
    def sample_task(drift: bool = False, seed: int | None = None):
        needed = engine.config.numeric_features + engine.config.categorical_features
        if any(c not in ("x_repo", "x_complex", "human_label") for c in needed):
            raise HTTPException(400, "Demo runs need the sample model; this engine uses other task features.")
        if seed is None:
            counter["seed"] += 1
            seed = counter["seed"]
        return demo_trajectory(params, seed, drift, needed)

    @app.get("/dashboard/observation-count")
    def observation_count():
        return {"finished": int(app.state.observation_store.count("ok"))}

    @app.get("/dashboard/runs")
    def recent_runs(limit: int = 12):
        recs = list(app.state.store.runs.values())[-limit:][::-1]
        return [{"run_id": r.run_id, "task_id": r.task_id, "spent": r.spent, "budget": r.budget, "quote": r.quote,
                 "status": r.status, "closed": r.closed, "success": r.success, "degraded": r.degraded,
                 "exploration": r.exploration, "steps": r.steps} for r in recs]

    return app
