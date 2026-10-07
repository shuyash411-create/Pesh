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
- **Observation.** Runs the observation loop (`pesh observe`) from the page. Paste API keys
  for Anthropic, OpenAI, Together or Fireworks. They are saved only in a local `.env` file
  (`--env-file`), and the page only ever shows the last four characters. Paste tasks, one
  prompt per line or JSONL, or upload a file; three samples are filled in. Pick models and the
  number of repeat and per-model runs, then press **Start observation**. A table fills row by
  row with predicted cost, actual cost, whether the run stayed under the ceiling, and any
  error. Cards show run-to-run predictability (ρτ), average prediction error, ceiling coverage
  and the cheapest model per task. Without keys it runs on the built-in simulator, free and
  offline. Runs go into the same SQLite file, and **Use my observation runs** on Analyze logs
  analyses them.

Everything runs offline on simulated data, with no API keys. On first launch, without
`--engine`, it trains a sample model (about 20 s) and caches it in
`artifacts/dashboard-sample/`. To use your own model, pass
`pesh dashboard --engine <engine.pkl> --feasibility <feasibility.pkl>`. Finished runs are
stored in `pesh_dashboard.sqlite` (`--db`). The cost chart loads Chart.js from a CDN; without
internet the page still works and shows the same figures as text.

## Observation loop

`pesh observe` is PESH's internal ground-truth collector. It runs real tasks through real model
APIs, asks the existing quote engine for a predicted cost and ceiling **before** each run,
records the **actual** cost the API returned, and feeds the pairs into the existing
econometrics to show how good the predictions were. It sits on top of the engine
(`src/pesh/observe/`) and reuses `QuoteEngine.quote`/`observe`, `econometrics/icc.py`,
`econometrics/tails.py` and `quote/aci.py` rather than reimplementing them.

```bash
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... TOGETHER_API_KEY=... FIREWORKS_API_KEY=...   # any subset
pesh observe --tasks tasks.jsonl --models claude,gpt,together-open --variance 10 --cross 3
```

- **Models** (API only, nothing runs locally): `claude` (Anthropic), `gpt` (OpenAI),
  `together-open`, `fireworks-open`, or `provider:model-id`. Keys are read from the environment,
  sent only in request headers, never logged or stored, and scrubbed from error messages. A model
  whose key is missing is skipped. With no usable keys the loop falls back to the offline
  simulator (`provider = sim`), so it always runs; `--simulate` forces that.
- **Tasks:** JSONL rows `{task_id, text, [features]}` (`tasks.jsonl` is a sample). Missing
  `features` are filled by a crude text heuristic. Supply real pre-execution features for
  meaningful predictions.
- **Two multi-run types:** *variance* (same task, first model in `--models`, `--variance` runs,
  default 10) and *cross-model* (same task on every model, `--cross` runs each, default 3). Runs per
  task = (models x `--cross`) + `--variance`. Runs are sequential; a failed call is stored as
  `status=error` and the loop continues.
- **Resumable:** every run is stored in the `observations` table of the SQLite file (`--db`,
  default `pesh_dashboard.sqlite`, shared with the dashboard) keyed by
  `(task_id, model, kind, run_idx)`. Stop with Ctrl+C and rerun the same command: completed runs
  are skipped, errored runs are retried, and the ACI state is rebuilt from the stored rows.
- **Per row:** predicted cost (p50) and ceiling (online ACI quote), actual cost, tokens, wall
  time, steps, `within_ceiling`, `abs_error`, `pct_error`. Actual cost is API-reported token usage
  times a per-token price (`DEFAULT_MODELS` in `observe/providers.py`; list prices change, so
  override with `--prices prices.json`, `{"claude": {"in_per_mtok": 1, "out_per_mtok": 5}}`).
- **After each batch** (`--batch-size` tasks): rho_tau (ICC) from the variance runs, Hill and
  power-law tail index of cost, mean/median absolute and percentage error and the
  predicted-vs-actual correlation, realised ceiling coverage against the 90% target (outcomes are
  fed to ACI so the ceiling self-calibrates), and a per-model summary with the cheapest model per
  task. Output goes to `--out` (default `reports/observe/`): `report.md`, `report.json`,
  `observations.csv`, `observations.parquet`.
- **Safety:** `--max-spend USD` stops the loop once that much has been spent in the session;
  `--max-tokens` caps each call's output.
- **From Python** (for the dashboard): `ObservationLoop(...).start()` / `.stop()` run it in a
  background thread, and `ObservationStore(db).frame()` reads the table.

**Cost-only by design.** The loop never scores, judges or predicts output quality. `success`
only records that the call completed with output. The default engine is trained on simulated
agent runs, so its absolute scale will not match single API calls until you train an engine on
your own observations (`observations.parquet` fits the run-log schema).

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

## M1: real trajectories (Bai et al., 2026)

Bai et al. ran OpenHands on SWE-bench Verified with eight models, four runs per task, and
released the trajectories. OpenHands writes one `output.jsonl` per model and run. Point
Pesh at the folder that holds them:

```bash
pesh ingest-openhands path/to/trajectories --out data/bai_runs.parquet --steps-out data/bai_steps.parquet
pesh h1h2 --logs data/bai_runs.parquet --out reports/m1
```

- **Ingest** reads per-call token usage from `history[*].llm_metrics` or `metrics.token_usages`.
  It uses the provider-reported `accumulated_cost` when present and recomputes cost from tokens
  otherwise. Outcomes come from each record's `report.resolved`, or a `report.json` (SWE-bench
  harness) next to each `output.jsonl`, or `--report`. Run numbers come from `run_<n>` folder
  names. `task_id` becomes `<model>::<instance_id>`, because repeat runs are only comparable
  within one agent.
- **`h1h2`** reports, per model and pooled, ρτ and its correlation ceiling (H1) and the Hill
  tail indices of cost and steps with their ratio (H2). Each comes with a 95% cluster-bootstrap
  interval and a plain verdict. Runs that hit the agent's step cap are excluded from the step
  tail, and the report says how many there were.
- Gold-patch size and the number of failing tests are kept as `proxy_*` columns for H3. They
  are only known after the fact, so never use them as quote features.

On simulated data at the same scale (500 tasks × 4 runs per model), the pipeline recovers the
known ρτ ≈ 0.80. The cost tail index interval is wide ([2.0, 3.1] pooled over two models), so
expect "inconclusive" on the infinite-variance question unless several models are pooled.

## Caveats

- Every number in `reports/` is **simulated**, from a DGP calibrated to published moments. It
  is not an empirical estimate.
- Conformal guarantees are **marginal**. Coverage on the hardest tasks is lower, and the model
  card reports it.
- Capped logs identify quotes only below the cap. Keep the exploration slice on
  (`--exploration`) and retrain with the automatic Powell path.
- The OpenHands adapter assumes `llm_metrics` / `accumulated_token_usage` fields. Adjust
  `FIELD_MAP` for other versions.
