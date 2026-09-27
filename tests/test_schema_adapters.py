from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pesh.data.adapters.generic import feasibility_frame, load_steps, runs_from_steps
from pesh.data.adapters.openhands import load_openhands_dir
from pesh.data.io import split_by_task
from pesh.schema import SchemaError, validate_runs

FIX = Path(__file__).parent / "fixtures"


def test_validate_runs_defaults_and_errors():
    df = validate_runs(pd.DataFrame({"task_id": [1], "run_id": [0], "cost": [1.0], "success": [True]}))
    assert not df["capped"].iloc[0] and np.isinf(df["cap"].iloc[0])
    with pytest.raises(SchemaError):
        validate_runs(pd.DataFrame({"task_id": [1], "run_id": [0], "cost": [0.0], "success": [True]}))
    with pytest.raises(SchemaError):
        validate_runs(pd.DataFrame({"task_id": [1], "run_id": [0], "cost": [1.0], "success": [True],
                                    "capped": [True]}))
    with pytest.raises(SchemaError):
        validate_runs(pd.DataFrame({"task_id": [1], "cost": [1.0], "success": [True]}))


def test_generic_steps_to_runs():
    steps = load_steps(FIX / "steps.jsonl")
    runs = runs_from_steps(steps, feature_cols=["repo_files"])
    assert len(runs) == 4
    assert set(runs["steps"]) == {4, 6}
    assert (runs["cost"] > 0).all() and runs["success"].sum() == 2
    ff = feasibility_frame(steps, runs)
    assert {"budget", "final_cost", "spent", "repeat_ewma"} <= set(ff.columns)
    assert (ff["spent"] <= ff["budget"]).all()


def test_openhands_adapter():
    steps, outs = load_openhands_dir(FIX / "openhands", outcomes={"django__django-1": True})
    assert len(steps) == 5
    assert (steps["input_tokens"] == 9000).all() and np.allclose(steps["cost"], 0.02)
    assert steps["repeated_views"].sum() >= 4
    runs = runs_from_steps(steps, outcomes=outs)
    assert runs["success"].iloc[0] and np.isclose(runs["cost"].iloc[0], 0.10)


def test_split_by_task_has_no_leakage(sim_small):
    from pesh.sim.dgp import simulate

    df = simulate(300, 3, seed=1)
    a, b, c = split_by_task(df, (0.5, 0.25, 0.25))
    assert not (set(a.task_id) & set(b.task_id)) and not (set(b.task_id) & set(c.task_id))
    assert len(a) + len(b) + len(c) == len(df)
