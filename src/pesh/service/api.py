"""FastAPI service: pre-run quotes (L1), enforced ceilings (L2) and fixed prices (L3).

Lifecycle of a governed run:
  POST /runs                       quote + ceiling B (exploration slice gets a wider B)
  POST /runs/{id}/authorize        before each model call: may this call be paid for? (pathwise C <= B)
  POST /runs/{id}/step             after each call: meter spend, controller says continue/degrade/stop
  POST /runs/{id}/finish           outcome; realised cost updates ACI (quote re-calibration)
  GET  /monitor/coverage           realised vs nominal coverage, drift alarm
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..config import ControllerConfig, PricingParams
from ..control.controller import CostController
from ..control.feasibility import FeasibilityModel
from ..quote.engine import QuoteEngine
from ..quote.pricing import PricingPolicy, fixed_price
from .state import RunRecord, RunStore


class QuoteRequest(BaseModel):
    features: dict
    alpha: float | None = Field(None, gt=0, lt=1)
    model_version: str = "default"


class RunRequest(QuoteRequest):
    task_id: str
    budget: float | None = Field(None, gt=0, description="explicit ceiling; default headroom * quote")


class AuthorizeRequest(BaseModel):
    input_tokens: float = Field(ge=0)
    cache_read_tokens: float = Field(0, ge=0)
    cache_write_tokens: float = Field(0, ge=0)
    max_output_tokens: float = Field(ge=0)


class StepRequest(BaseModel):
    input_tokens: float = Field(ge=0)
    output_tokens: float = Field(ge=0)
    cache_read_tokens: float = Field(0, ge=0)
    cache_write_tokens: float = Field(0, ge=0)
    cost: float | None = Field(None, ge=0, description="provider-reported cost; else computed")
    repeated_views: float = Field(0, ge=0)


class FinishRequest(BaseModel):
    success: bool


def call_cost(pr: PricingParams, input_tokens, output_tokens, cache_read=0.0, cache_write=0.0,
              price_mult: float = 1.0) -> float:
    uncached = max(input_tokens - cache_read - cache_write, 0.0)
    return price_mult * (pr.p_in * uncached + pr.w * pr.p_in * cache_write + pr.delta * pr.p_in * cache_read
                         + pr.p_out * output_tokens)


def create_app(engine: QuoteEngine, feasibility: FeasibilityModel | None = None,
               controller_config: ControllerConfig | None = None, pricing: PricingParams | None = None,
               pricing_policy: PricingPolicy | None = None, headroom: float = 1.0,
               exploration_rate: float = 0.05, exploration_mult: float = 3.0, db_path: str | None = None,
               seed: int = 0, drift_tolerance: float = 0.03, drift_min_n: int = 100) -> FastAPI:
    app = FastAPI(title="Pesh cost-bounded execution", version="0.1.0")
    cfg = controller_config or ControllerConfig()
    pr = pricing or PricingParams()
    store = RunStore(db_path)
    rng = np.random.default_rng(seed)
    app.state.store = store
    app.state.engine = engine

    def _quote(features: dict, alpha, model_version: str) -> dict:
        df = pd.DataFrame([features])
        missing = [c for c in engine.config.numeric_features + engine.config.categorical_features if c not in df]
        if missing:
            raise HTTPException(422, f"missing features: {missing}")
        online = model_version in engine.aci and alpha is None
        q = engine.quote(df, alpha=alpha, online=online, model_version=model_version).iloc[0]
        base = float(engine.base_log(df)[0])
        B = float(q["quote"]) * headroom
        price = fixed_price(engine.expected_capped_cost(df, B), B, pricing_policy)
        return {"p50": float(q["p50"]), "quote": float(q["quote"]), "coverage_level": float(q["coverage_level"]),
                "method": q["method"], "calib_n": int(q["calib_n"]), "warning": q["warning"] or None,
                "ceiling": B, "fixed_price": float(price["price"][0]),
                "expected_capped_cost": float(price["expected_capped"][0]), "base_log": base}

    def _run(run_id: str) -> RunRecord:
        rec = store.get(run_id)
        if rec is None:
            raise HTTPException(404, "unknown run")
        return rec

    @app.get("/health")
    def health():
        return {"status": "ok", "engine": engine.card.get("method"), "alpha": engine.config.alpha,
                "feasibility": feasibility is not None}

    @app.get("/model")
    def model_card():
        return {k: v for k, v in engine.card.items() if k != "config"}

    @app.post("/quote")
    def quote(req: QuoteRequest):
        out = _quote(req.features, req.alpha, req.model_version)
        out.pop("base_log")
        return out

    @app.post("/runs")
    def create_run(req: RunRequest):
        q = _quote(req.features, req.alpha, req.model_version)
        exploration = bool(rng.random() < exploration_rate)
        B = req.budget if req.budget is not None else q["ceiling"]
        if exploration:
            B *= exploration_mult     # buy information above the usual cap (censoring externality)
        rec = RunRecord(store.new_id(), str(req.task_id), req.model_version, req.features, q["quote"],
                        q["base_log"], float(B), exploration)
        store.add(rec)
        q.pop("base_log")
        return {"run_id": rec.run_id, "budget": rec.budget, "exploration": exploration, **q}

    @app.get("/runs/{run_id}")
    def get_run(run_id: str):
        return _run(run_id).to_dict()

    @app.post("/runs/{run_id}/authorize")
    def authorize(run_id: str, req: AuthorizeRequest):
        rec = _run(run_id)
        mult = cfg.degrade_price_mult if rec.degraded else 1.0
        worst = call_cost(pr, req.input_tokens, req.max_output_tokens, req.cache_read_tokens,
                          req.cache_write_tokens, mult)
        allowed = rec.status == "running" and rec.spent + worst <= rec.budget
        return {"allowed": bool(allowed), "worst_case_cost": worst, "spent": rec.spent,
                "remaining": rec.budget - rec.spent, "degraded": rec.degraded, "status": rec.status}

    @app.post("/runs/{run_id}/step")
    def step(run_id: str, req: StepRequest):
        rec = _run(run_id)
        if rec.status != "running":
            raise HTTPException(409, f"run is {rec.status}")
        mult = cfg.degrade_price_mult if rec.degraded else 1.0
        c = req.cost if req.cost is not None else call_cost(pr, req.input_tokens, req.output_tokens,
                                                            req.cache_read_tokens, req.cache_write_tokens, mult)
        rec.spent += c
        rec.steps += 1
        rec.context = req.input_tokens
        rec.last_step_cost = c
        rec.repeat_ewma = 0.7 * rec.repeat_ewma + 0.3 * req.repeated_views
        if rec.spent >= rec.budget:
            decision = "stop"
            rec.status = "capped"
        else:
            ctl = CostController(rec.budget, feasibility, degrade=True, config=cfg)
            decision = ctl.decide_one(rec.steps, rec.spent, rec.repeat_ewma, rec.context, c, rec.degraded,
                                      rec.budget)
            if decision == "degrade":
                rec.degraded = True
            elif decision == "stop":
                rec.status = "stopped"
        rec.decisions.append(decision)
        return {"decision": decision, "spent": rec.spent, "remaining": rec.budget - rec.spent,
                "step_cost": c, "degraded": rec.degraded, "status": rec.status}

    @app.post("/runs/{run_id}/finish")
    def finish(run_id: str, req: FinishRequest):
        rec = _run(run_id)
        if rec.closed:
            raise HTTPException(409, "outcome already reported")
        rec.closed = True
        capped = rec.status == "capped"
        stopped = rec.status == "stopped"
        rec.success = bool(req.success) and rec.status == "running"
        if rec.status == "running":
            rec.status = "finished"
        aci = engine.aci_for(rec.model_version)
        missed, covered = False, None
        if not stopped:
            # A capped run's true cost is >= B; if B >= quote it is a certain miss. Runs stopped by the
            # feasibility controller are censored at an uninformative point and do not update ACI.
            cost = max(max(rec.budget, rec.spent) if capped else rec.spent, 1e-9)
            missed = aci.update(math.log(cost) - rec.base_log)
            covered = bool(cost <= rec.quote)
            store.record_outcome(rec, covered)
        return {"run_id": run_id, "status": rec.status, "success": rec.success, "cost": rec.spent,
                "within_quote": covered, "aci_missed": bool(missed), "aci": aci.state()}

    @app.get("/monitor/coverage")
    def monitor(model_version: str = "default"):
        cov, n = store.rolling_coverage(model_version)
        target = 1.0 - engine.config.alpha
        alarm = bool(n >= drift_min_n and cov < target - drift_tolerance)
        aci = engine.aci_for(model_version).state()
        return {"model_version": model_version, "target": target, "rolling_coverage": None if n == 0 else cov,
                "n": n, "drift_alarm": alarm, "aci": aci,
                "action": "recalibrate / retrain on recent runs" if alarm else None}

    @app.post("/model_version/{model_version}/reset")
    def reset(model_version: str):
        engine.reset_model_version(model_version)
        return engine.aci_for(model_version).state()

    @app.get("/runs")
    def export(limit: int = 1000):
        df = store.export_runs()
        return df.tail(limit).to_dict(orient="records")

    return app
