# Pesh — cost-bounded AI agent execution

Tell a buyer **before an agent runs** what the run will cost, enforce a **ceiling** while it
runs, and price a **guarantee**. This is an MVP implementation of Tiwari (2026), *Pricing the
Unpredictable*: quantile/conformal econometrics for the quote, a feasibility-aware controller
for the ceiling, and pooling-aware pricing for the guarantee.

- **Plan:** [`docs/PLAN.md`](docs/PLAN.md) covers the research grounding, architecture, the
  train/test/improve loop, the go-to-market ladder and milestones.
- **Evidence:** [`reports/results.md`](reports/results.md) shows the MVP reproducing the
  paper's simulation study (E0–E5).

## Quickstart

```bash
pip install -e ".[dev]"

pytest                     # unit, API and CLI tests (~1–2 min)
pytest -m slow             # statistical reproduction of E1–E5 (~1 min)

pesh simulate --tasks 3000 --runs 4 --out data/sim.parquet
pesh train --logs data/sim.parquet --model-version sim-v1 --out artifacts
pesh evaluate --logs data/sim.parquet --engine artifacts/sim-v1/<stamp>/engine.pkl
pesh experiments --out reports         # regenerate reports/results.md + figures
pesh serve --engine artifacts/sim-v1/<stamp>/engine.pkl \
           --feasibility artifacts/sim-v1/<stamp>/feasibility.pkl
```

Quote one task:

```bash
curl -s localhost:8000/quote -H 'content-type: application/json' \
  -d '{"features": {"x_repo": 1.2, "x_complex": 0.8, "human_label": 2}}'
```

Govern a run (the pathwise ceiling comes from `authorize` running before every model call):

```python
from pesh.service.gateway import PeshClient
c = PeshClient("http://127.0.0.1:8000")
run = c.start_run("task-42", {"x_repo": 1.2, "x_complex": 0.8, "human_label": 2})
while c.authorize(run["run_id"], input_tokens=ctx, max_output_tokens=1500)["allowed"]:
    ... call the model ...
    d = c.step(run["run_id"], input_tokens=ctx, output_tokens=out, cache_read_tokens=cached)
    if d["decision"] == "degrade": ...   # compact context, switch to the cheaper model
    if d["decision"] == "stop": break
c.finish(run["run_id"], success=ok)
```

Or in-process, around any LLM call: `BudgetSession(budget=2.5).call(fn, input_tokens=..., max_output_tokens=...)`.

## Dashboard

```bash
pesh dashboard            # opens http://127.0.0.1:8050 in your browser
```

A local web page for people who don't use the terminal. It has three screens:

- **Analyze logs.** Load a CSV or Parquet file of past runs, or press **Use sample data**.
  The page shows how predictable the cost is, how extreme the expensive runs get (with an
  "infinite variance" warning when the tail index is below 2), a monthly forecast ("expected
  ~$X, 90% of months below $Y"), and a chart of cost per run with the median, p90 and p99
  marked.
- **Quote a task.** Describe the task and get "This run will cost about $X. Guaranteed
  ceiling $Y. Fixed price $Z."
- **Live monitor.** Quote accuracy against the 90% target, the drift alarm, spend against
  the ceiling for the current run, how often the safety controls stepped in, and recent
  runs. **Simulate a run** sends demo runs through the real service endpoints (`/runs`,
  `/authorize`, `/step`, `/finish`). Tick "Pretend the AI provider just updated its model"
  to make runs more expensive and watch the safety controls work harder.

Everything runs offline on simulated data, with no API keys. On first launch, without
`--engine`, it trains a sample model (about 20 s) and caches it in
`artifacts/dashboard-sample/`. To use your own model, pass
`pesh dashboard --engine <engine.pkl> --feasibility <feasibility.pkl>`. Finished runs are
stored in `pesh_dashboard.sqlite` (`--db`). The cost chart loads Chart.js from a CDN; without
internet the page still works and shows the same figures as text.

## Paper → code map

| Paper | Code |
|---|---|
| Eq. (3)–(4) cost identity, Prop. 1 quadratic accumulation | `pesh/cost_model.py` |
| §5.1 calibrated DGP, Appendix B calibration (here: SMM) | `pesh/sim/dgp.py`, `pesh/sim/calibrate.py` |
| Prop. 2 tail-index halving, H2 | `pesh/econometrics/tails.py` |
| Prop. 3 predictability ceiling (ICC), H1 | `pesh/econometrics/icc.py` |
| Prop. 4 conformal quotes, Mondrian | `pesh/quote/conformal.py`, `pesh/quote/engine.py` |
| Eq. (8) adaptive conformal inference, §5.4 | `pesh/quote/aci.py` |
| Prop. 5 censoring, Powell / Chernozhukov–Hong, H5 | `pesh/econometrics/censored.py` |
| §4.6 rule (10), controllers P0–P4, H6 | `pesh/control/` |
| Prop. 6 pooling, Eq. (12) WTP, L3 pricing | `pesh/econometrics/risk.py`, `pesh/quote/pricing.py` |
| §5 experiments E1–E5 | `pesh/eval/experiments.py`, `reports/` |

## Bring your own logs

Run-level schema (`pesh/schema.py`): `task_id, run_id, cost, success` are required. Optional:
`capped, cap, steps, input_tokens, output_tokens, model_version, exploration`, plus any
**pre-execution** feature columns. Pass them with `pesh train --numeric a,b --categorical c`.

Step-level logs (JSONL/CSV/Parquet: `task_id, run_id, step, input_tokens, output_tokens` and
optionally `cache_read_tokens, cache_write_tokens, cost, repeated_views`) go through
`pesh.data.adapters.generic.runs_from_steps`. OpenHands trajectories go through
`pesh.data.adapters.openhands.load_openhands_dir`. The step logs also train the feasibility
model (`pesh train --steps`).

## Caveats

- Every number in `reports/` is **simulated**, from a DGP calibrated to published moments. It
  is not an empirical estimate.
- Conformal guarantees are **marginal**. Coverage on the hardest tasks is lower, and the model
  card reports it.
- Capped logs identify quotes only below the cap. Keep the exploration slice on
  (`--exploration`) and retrain with the automatic Powell path.
- The OpenHands adapter assumes `llm_metrics` / `accumulated_token_usage` fields. Adjust
  `FIELD_MAP` for other versions.
