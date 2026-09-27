import numpy as np

from pesh.quote.aci import ACI
from pesh.quote.conformal import SplitConformal, conformal_k, conformal_offset, theoretical_coverage


def test_conformal_k_and_small_n():
    assert conformal_k(99, 0.1) == 90
    assert np.isinf(conformal_offset(np.arange(5.0), 0.1))     # n too small for 90%
    assert np.isclose(theoretical_coverage(30, 0.1), 28 / 31)


def test_split_conformal_marginal_validity(rng):
    covs = []
    for _ in range(300):
        cal, test = rng.standard_normal(200), rng.standard_normal(2000)
        sc = SplitConformal(0.1).calibrate(np.zeros(200), cal)
        covs.append(np.mean(test <= sc.global_offset_))
    assert 0.9 <= np.mean(covs) <= 0.9 + 1 / 201 + 0.005


def test_mondrian_equalises_group_coverage(rng):
    g = rng.integers(0, 2, 20000)
    y = rng.standard_normal(20000) * np.where(g == 1, 3.0, 1.0)
    sc = SplitConformal(0.1).calibrate(np.zeros(10000), y[:10000], g[:10000], min_group_n=100)
    off = sc.offset(g[10000:], 10000)
    cov = [np.mean(y[10000:][g[10000:] == k] <= off[g[10000:] == k]) for k in (0, 1)]
    assert all(abs(c - 0.9) < 0.02 for c in cov)
    glob = SplitConformal(0.1).calibrate(np.zeros(10000), y[:10000]).offset(None, 10000)
    assert np.mean(y[10000:][g[10000:] == 1] <= glob[g[10000:] == 1]) < 0.85


def test_aci_restores_coverage_after_shift(rng):
    aci = ACI(0.1, gamma=0.01, window=500, init_scores=rng.standard_normal(500))
    hits = []
    for t in range(12000):
        s = rng.standard_normal() + (1.0 if t >= 4000 else 0.0)
        off = aci.offset()
        hits.append(s <= off)
        aci.update(s, off)
    post = np.array(hits[5000:])
    assert abs(post.mean() - 0.9) < 0.02
