"""BudgetSession: a pathwise cost ceiling around any LLM call function (assurance level L2).

    session = BudgetSession(budget=2.50)
    result = session.call(my_llm_fn, input_tokens=12_000, cached_tokens=10_000,
                          max_output_tokens=1_024, messages=msgs)

`my_llm_fn(**kwargs)` must return (result, usage) where usage has input_tokens,
output_tokens and optionally cache_read_tokens / cache_write_tokens / cost.
The call is refused *before* it is paid for if its worst case (max_output_tokens) could
breach the ceiling — the enforcement half that gateways implement, plus the controller.

PeshClient does the same against a running `pesh serve` instance over HTTP.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable

from ..config import ControllerConfig, PricingParams
from ..control.controller import CostController
from ..control.feasibility import FeasibilityModel


class BudgetExceeded(RuntimeError):
    pass


class RunStopped(RuntimeError):
    pass


class BudgetSession:
    def __init__(self, budget: float, pricing: PricingParams | None = None,
                 feasibility: FeasibilityModel | None = None, config: ControllerConfig | None = None,
                 degrade: bool = True):
        self.budget = float(budget)
        self.pricing = pricing or PricingParams()
        self.cfg = config or ControllerConfig()
        self.controller = CostController(self.budget, feasibility, degrade, self.cfg)
        self.spent = 0.0
        self.steps = 0
        self.repeat_ewma = 0.0
        self.degraded = False
        self.stopped = False
        self.decision = "continue"
        self.log: list[dict] = []

    @property
    def price_mult(self) -> float:
        return self.cfg.degrade_price_mult if self.degraded else 1.0

    def estimate(self, input_tokens: float, output_tokens: float, cached_tokens: float = 0.0,
                 cache_write_tokens: float = 0.0) -> float:
        p = self.pricing
        uncached = max(input_tokens - cached_tokens - cache_write_tokens, 0.0)
        return self.price_mult * (p.p_in * uncached + p.w * p.p_in * cache_write_tokens
                                  + p.delta * p.p_in * cached_tokens + p.p_out * output_tokens)

    def can_afford(self, input_tokens: float, max_output_tokens: float, cached_tokens: float = 0.0) -> bool:
        return self.spent + self.estimate(input_tokens, max_output_tokens, cached_tokens) <= self.budget

    def call(self, fn: Callable[..., tuple[Any, dict]], *, input_tokens: float, max_output_tokens: float,
             cached_tokens: float = 0.0, repeated_views: float = 0.0, **kwargs) -> Any:
        if self.stopped:
            raise RunStopped(f"run stopped by controller after {self.steps} steps (spent ${self.spent:.4f})")
        if not self.can_afford(input_tokens, max_output_tokens, cached_tokens):
            self.stopped = True
            raise BudgetExceeded(f"next call could exceed ceiling ${self.budget:.4f} (spent ${self.spent:.4f})")
        result, usage = fn(**kwargs)
        cost = usage.get("cost")
        if cost is None:
            cost = self.estimate(usage.get("input_tokens", input_tokens), usage.get("output_tokens", 0.0),
                                 usage.get("cache_read_tokens", cached_tokens), usage.get("cache_write_tokens", 0.0))
        self.spent += float(cost)
        self.steps += 1
        self.repeat_ewma = 0.7 * self.repeat_ewma + 0.3 * repeated_views
        self.decision = self.controller.decide_one(self.steps, self.spent, self.repeat_ewma,
                                                   usage.get("input_tokens", input_tokens), float(cost),
                                                   self.degraded, self.budget)
        if self.decision == "degrade":
            self.degraded = True
        elif self.decision == "stop":
            self.stopped = True
        self.log.append({"step": self.steps, "cost": float(cost), "spent": self.spent, "decision": self.decision})
        return result


class PeshClient:
    """Minimal HTTP client for a running Pesh service (stdlib only)."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 10.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read())

    def quote(self, features: dict, **kw) -> dict:
        return self._post("/quote", {"features": features, **kw})

    def start_run(self, task_id: str, features: dict, **kw) -> dict:
        return self._post("/runs", {"task_id": task_id, "features": features, **kw})

    def authorize(self, run_id: str, **usage) -> dict:
        return self._post(f"/runs/{run_id}/authorize", usage)

    def step(self, run_id: str, **usage) -> dict:
        return self._post(f"/runs/{run_id}/step", usage)

    def finish(self, run_id: str, success: bool) -> dict:
        return self._post(f"/runs/{run_id}/finish", {"success": success})

    def coverage(self, model_version: str = "default") -> dict:
        return self._get(f"/monitor/coverage?model_version={model_version}")
