import numpy as np

from pesh.sim.calibrate import TARGETS, moments
from pesh.sim.dgp import execute, plan_runs, simulate, simulate_tasks


def test_calibrated_moments_near_targets():
    m = moments(simulate(1500, 4, seed=3))
    assert abs(m["mean_tokens_m"] / TARGETS["mean_tokens_m"] - 1) < 0.2
    assert abs(m["mean_cost"] / TARGETS["mean_cost"] - 1) < 0.2
    assert abs(m["in_out_ratio"] / TARGETS["in_out_ratio"] - 1) < 0.15
    assert abs(m["success"] - TARGETS["success"]) < 0.04
    assert abs(m["kendall_tau"] - TARGETS["kendall_tau"]) < 0.08
    assert abs(m["icc"] - TARGETS["icc"]) < 0.06


def test_common_random_numbers_are_deterministic():
    tasks = simulate_tasks(200, seed=5)
    plan = plan_runs(tasks, 2, seed=5)
    a, b = execute(plan).runs, execute(plan).runs
    assert np.array_equal(a["cost"], b["cost"])


def test_shift_raises_cost():
    base = simulate(1500, 1, seed=9)["cost"].mean()
    shifted = simulate(1500, 1, seed=9, verbosity=1.35, step_mult=1.25)["cost"].mean()
    assert shifted > 1.4 * base
