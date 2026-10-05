"""`pesh observe` command."""

from __future__ import annotations

import sys
from pathlib import Path

from .providers import KEY_ENV, build_providers, resolve_models
from .runner import ObservationLoop, plan_jobs
from .store import ObservationStore
from .tasks import load_tasks


def run(a, load_default_engine) -> int:
    from ..quote.engine import QuoteEngine

    tasks = load_tasks(a.tasks)
    specs = resolve_models([m for m in a.models.split(",") if m], a.prices)
    if not specs:
        sys.exit("pesh observe: --models is empty")
    providers = {} if a.simulate else build_providers(max_tokens=a.max_tokens)

    runnable = [m for m in specs if m.provider in providers]
    for m in specs:
        if m not in runnable and not a.simulate and providers:
            print(f"skipping {m.alias}: {KEY_ENV[m.provider]} is not set", file=sys.stderr)
    mode = "live"
    if not runnable:
        mode, runnable = "simulated", specs
        why = "--simulate" if a.simulate else "no API keys for the requested models"
        print(f"{why}: using the offline simulator (no API calls, no cost)", file=sys.stderr)

    engine = QuoteEngine.load(a.engine) if a.engine else load_default_engine()
    store = ObservationStore(a.db)
    out = Path(a.out)
    loop = ObservationLoop(
        engine, tasks, runnable, providers, store, r_variance=a.variance, r_cross=a.cross,
        batch_size=a.batch_size, report_dir=str(out), max_spend=a.max_spend,
        on_event=lambda kind, info: _echo(kind, info, a.quiet))

    per_task = len(plan_jobs(tasks[0], runnable, a.variance, a.cross))
    print(f"{mode}: {len(tasks)} tasks x {per_task} runs = {len(tasks) * per_task} planned "
          f"(models: {', '.join(m.alias for m in runnable)}; variance runs on {runnable[0].alias}); "
          f"db={a.db}", flush=True)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\ninterrupted; stored runs are kept, rerun the same command to resume", file=sys.stderr)
        loop._finish()
        return 130
    print(f"\nreport: {out / 'report.md'}  data: {out / 'observations.csv'}, {out / 'observations.parquet'}")
    return 0


def _echo(kind: str, info: dict, quiet: bool) -> None:
    if kind == "run" and not quiet:
        print(f"  {info['task_id']} {info['model']} {info['kind']}#{info['run_idx']}: predicted "
              f"${info['predicted_cost']:.4f} ceiling ${info['predicted_ceiling']:.4f} actual "
              f"${info['actual_cost']:.4f} {'ok' if info['within_ceiling'] else 'OVER CEILING'}", flush=True)
    elif kind == "error":
        print(f"  {info['task_id']} {info['model']} {info['kind']}#{info['run_idx']}: ERROR {info['error']}",
              file=sys.stderr, flush=True)
    elif kind == "budget":
        print(f"max spend ${info['max_spend']} reached (spent ${info['spent']:.4f}); stopping", flush=True)
    elif kind == "batch":
        print(info["markdown"].split("## Per-model")[0], flush=True)
    elif kind == "final":
        print(info["markdown"], flush=True)
