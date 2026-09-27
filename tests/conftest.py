import numpy as np
import pytest

from pesh.sim.dgp import simulate


@pytest.fixture(scope="session")
def sim_small():
    """(train, calib, test) simulated run logs, one run per task."""
    return (simulate(1500, 1, seed=11), simulate(800, 1, seed=12), simulate(3000, 1, seed=13))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
