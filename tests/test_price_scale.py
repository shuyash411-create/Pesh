import numpy as np
import pandas as pd
import pytest

from pesh.config import PricingParams, QuoteConfig
from pesh.observe import ObservationLoop, ObservationStore, Task, resolve_models
from pesh.observe.providers import ModelSpec
from pesh.observe.tasks import feature_frame
from pesh.quote.engine import QuoteEngine
from pesh.quote.price_scale import PriceScaler
from pesh.sim.dgp import simulate_tasks


def test_ratio_reprices_the_training_token_mix():
    ref = PricingParams()
    # 900 input units and 100 output tokens per run, billed at the reference price
    runs = pd.DataFrame({"cost": [900 * ref.p_in + 100 * ref.p_out] * 2 + [1.0],
                         "output_tokens": [100, 100, 1], "degraded": [False, False, True]})
    s = PriceScaler.from_runs(runs, ref)
    assert s.input_units == pytest.approx(1800) and s.output_tokens == pytest.approx(200)   # degraded run left out
    assert s.ratio(ref.p_in, ref.p_out) == pytest.approx(1.0)
    assert s.ratio(ref.p_in / 3, ref.p_out / 3) == pytest.approx(1 / 3)        # same shape -> exact multiple
    # a flat price is weighted by the mix, not by the sum of the two prices
    flat = 0.88e-6
    expected = flat * 1000 / (900 * ref.p_in + 100 * ref.p_out)
    assert s.ratio(flat, flat) == pytest.approx(expected)
    assert s.ratio(flat, flat) != pytest.approx(2 * flat / (ref.p_in + ref.p_out))
    with pytest.raises(ValueError):
        s.ratio(float("nan"), 1e-6)


def test_every_model_gets_its_own_scale_from_the_price_table():
    s = PriceScaler.for_simulator()
    specs = resolve_models(["claude", "gpt", "together-open", "fireworks-open"])
    ratios = {m.alias: s.ratio_for(m) for m in specs}
    assert ratios["claude"] == pytest.approx(1 / 3)      # Haiku is $1/$5 vs the $3/$15 training price
    assert len(set(np.round(list(ratios.values()), 6))) == 4 and all(0 < r < 1 for r in ratios.values())
    future = ModelSpec("future", "openai", "x", 6e-6, 30e-6)   # any new model just needs a price
    assert s.ratio_for(future) == pytest.approx(2.0)


@pytest.fixture(scope="module")
def engine_bytes(sim_small):
    import pickle
    train, calib, _ = sim_small
    return pickle.dumps(QuoteEngine(QuoteConfig()).fit(train, calib))


def test_observation_quotes_are_in_each_models_dollars(engine_bytes, tmp_path):
    import pickle
    engine = pickle.loads(engine_bytes)
    t = simulate_tasks(40, seed=5)
    tasks = [Task(f"t{i}", "", {"x_repo": r.x_repo, "x_complex": r.x_complex, "human_label": int(r.human_label)})
             for i, r in enumerate(t.itertuples())]
    models = resolve_models(["claude", "gpt"])
    store = ObservationStore(tmp_path / "o.sqlite")
    loop = ObservationLoop(engine, tasks, models, {}, store, r_variance=0, r_cross=4, batch_size=100)
    loop.run()
    df = store.frame().query("status == 'ok'")

    # the point estimate is the engine's p50 re-priced, nothing else
    cols = list(engine.design.numeric) + list(engine.design.categorical)
    p50 = float(engine.quote(feature_frame(tasks[0], cols))["p50"].iloc[0])
    row = df[(df.task_id == "t0") & (df.model == "gpt")].iloc[0]
    assert row.predicted_cost == pytest.approx(p50 * loop.price_scale(models[1]))

    for m, g in df.groupby("model"):
        ratio = (g.actual_cost / g.predicted_cost).median()
        assert 0.6 < ratio < 1.7, (m, ratio)                  # unscaled this was ~0.4 (claude), ~0.05 (gpt)
        assert 0.8 <= (g.actual_cost <= g.predicted_ceiling).mean() <= 0.98, m
