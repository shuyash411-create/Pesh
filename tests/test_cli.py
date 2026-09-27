import json

from pesh.cli import main


def test_simulate_train_evaluate(tmp_path, capsys):
    logs = tmp_path / "sim.parquet"
    main(["simulate", "--tasks", "600", "--runs", "3", "--out", str(logs)])
    main(["train", "--logs", str(logs), "--out", str(tmp_path / "art"), "--model-version", "sim-v1"])
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"):])
    assert 0.8 < payload["evaluation"]["coverage"] < 0.97
    assert 0.6 < payload["hypotheses"]["H1_predictability"]["rho"] < 0.95
    engine = next((tmp_path / "art").rglob("engine.pkl"))
    main(["evaluate", "--logs", str(logs), "--engine", str(engine)])
    main(["quote", "--engine", str(engine), "--features", '{"x_repo":0,"x_complex":0,"human_label":1}'])
    assert "quote" in capsys.readouterr().out
