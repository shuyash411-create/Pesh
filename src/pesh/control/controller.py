"""Runtime controller (assurance level L2), Section 4.6 and experiment E4.

Policies:
  P0 no control
  P1 hard cap: refuse any call that would take spend past B (pathwise C <= B)
  P2 P1 + feasibility stop: stop when pi^(sigma_t) is too low
  P3 P1 + degrade: once spend passes kappa*B, compact context and switch to a cheaper model
  P4 P1 + P2 + P3

Stopping rule (10):  continue  <=>  v * pi^(sigma_t) >= E[C - c_t | sigma_t, continue, C <= B].
Without a value v the rule reduces to a threshold on pi^.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ControllerConfig
from ..sim.dgp import NoControl, RunState
from .feasibility import FeasibilityModel


class CostController:
    def __init__(self, B, feasibility: FeasibilityModel | None = None, degrade: bool = False,
                 config: ControllerConfig | None = None, name: str | None = None,
                 expected_remaining_steps: float = 10.0):
        self.B = B
        self.feasibility = feasibility
        self.degrade = degrade
        self.cfg = config or ControllerConfig()
        self.expected_remaining_steps = expected_remaining_steps
        if name is None:
            parts = ["cap"] + (["feasibility"] if feasibility is not None else []) + (["degrade"] if degrade else [])
            name = " + ".join(parts)
        self.name = name

    def budget(self, plan) -> np.ndarray:
        B = np.asarray(self.B, float)
        return np.full(plan.n, float(B)) if B.ndim == 0 else B

    def decide(self, state: RunState) -> tuple[np.ndarray, np.ndarray]:
        n = len(state.t)
        stop = np.zeros(n, bool)
        if self.feasibility is not None:
            pi = self.feasibility.predict_state(state)
            eligible = state.t >= self.cfg.min_steps_before_stop
            if self.cfg.value_of_success is not None:
                remaining = np.minimum(state.budget - state.spent,
                                       state.last_step_cost * self.expected_remaining_steps)
                stop = eligible & (self.cfg.value_of_success * pi < remaining)
            else:
                stop = eligible & (pi < self.cfg.feasibility_threshold)
        degrade = np.zeros(n, bool)
        if self.degrade:
            degrade = (state.spent >= self.cfg.kappa * state.budget) & ~state.degraded
        return stop, degrade

    # scalar interface for the service ---------------------------------------------------
    def decide_one(self, step: int, spent: float, repeat_ewma: float, context: float,
                   last_step_cost: float, degraded: bool, budget: float) -> str:
        st = RunState(t=np.array([step]), spent=np.array([spent]), context=np.array([context]),
                      repeat_ewma=np.array([repeat_ewma]), degraded=np.array([degraded]),
                      active=np.array([True]), budget=np.array([budget]),
                      last_step_cost=np.array([last_step_cost]))
        stop, deg = self.decide(st)
        if stop[0]:
            return "stop"
        if deg[0]:
            return "degrade"
        return "continue"


@dataclass
class PolicySet:
    """The five policies of Table 7 at a common ceiling B."""

    B: float
    feasibility: FeasibilityModel
    config: ControllerConfig

    def all(self) -> dict:
        c = self.config
        return {
            "P0 no control": NoControl(),
            "P1 hard cap": CostController(self.B, None, False, c, "P1 hard cap"),
            "P2 cap + feasibility stop": CostController(self.B, self.feasibility, False, c,
                                                        "P2 cap + feasibility stop"),
            "P3 cap + degrade": CostController(self.B, None, True, c, "P3 cap + degrade"),
            "P4 cap + feasibility + degrade": CostController(self.B, self.feasibility, True, c,
                                                             "P4 cap + feasibility + degrade"),
        }


def can_afford(spent: float, next_call_cost: float, budget: float) -> bool:
    """Pathwise ceiling check performed *before* a call is paid for."""
    return spent + next_call_cost <= budget
