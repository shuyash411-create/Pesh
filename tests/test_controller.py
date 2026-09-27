import numpy as np
import pytest

from pesh.config import ControllerConfig
from pesh.control.controller import CostController, PolicySet
from pesh.control.feasibility import train_from_simulation
from pesh.control.replay import compare_policies, replay_logged_runs
from pesh.sim.dgp import execute, plan_runs, simulate_tasks


@pytest.fixture(scope="module")
def fm():
    return train_from_simulation(n_tasks=800, seed=3)


def test_pathwise_ceiling_never_breached(fm, rng):
    tasks = simulate_tasks(2500, seed=21)
    plan = plan_runs(tasks, 4, seed=21)
    B = rng.uniform(0.2, 6.0, plan.n)                 # per-run ceilings (10k random trajectories)
    for degrade in (False, True):
        runs = execute(plan, CostController(B, fm, degrade)).runs
        assert (runs["cost"].to_numpy() <= B + 1e-12).all()


def test_policy_ordering(fm):
    tasks = simulate_tasks(2000, seed=22)
    plan = plan_runs(tasks, 2, seed=22)
    B = float(np.quantile(execute(plan).runs["cost"], 0.9))
    cfg = ControllerConfig()
    t = compare_policies(plan, PolicySet(B, fm, cfg).all(), config=cfg, B=B)
    assert t.loc["P0 no control", "p_exceed_B"] > 0.05
    assert (t.iloc[1:]["p_exceed_B"] == 0).all()
    assert t.loc["P4 cap + feasibility + degrade", "mean_cost"] < t.loc["P1 hard cap", "mean_cost"]
    assert t.loc["P1 hard cap", "cvar95"] < 0.5 * t.loc["P0 no control", "cvar95"]


def test_decide_one_and_replay(fm):
    ctl = CostController(2.0, fm, degrade=True)
    assert ctl.decide_one(3, 0.1, 0.0, 20_000, 0.02, False, 2.0) == "continue"
    assert CostController(2.0, None, degrade=True).decide_one(20, 1.2, 0.0, 200_000, 0.05, False, 2.0) == "degrade"
    assert ctl.decide_one(60, 1.9, 2.5, 400_000, 0.1, True, 2.0) == "stop"
    # pi^ falls with the loop signal and with the share of budget already spent
    pi = fm.predict([30, 30, 30], [0.5, 0.5, 1.5], [0.0, 1.5, 0.0], [150_000] * 3, [0.03] * 3, [2.0] * 3)
    assert pi[0] > pi[1] and pi[0] > pi[2]
    import pandas as pd
    steps = pd.DataFrame({"task_id": 1, "run_id": 0, "step": range(1, 21), "step_cost": 0.2,
                          "repeat_ewma": 0.0, "context": 50_000})
    out = replay_logged_runs(steps, CostController(1.0), 1.0)
    assert out.loc[0, "capped"] and out.loc[0, "cost"] <= 1.0
