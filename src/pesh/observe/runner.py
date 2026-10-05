"""The observation loop: predict -> run -> record -> (per batch) report.

    loop = ObservationLoop(engine, tasks, models, providers, store)
    loop.start()      # background thread (what a dashboard would call)
    loop.stop()       # finishes the current API call, then returns; safe to start again later
    loop.run()        # or run in the foreground

Per run: ``engine.quote`` gives the predicted cost (p50) and ceiling (the online/ACI quote);
the provider returns the actual cost; the pair is stored; the actual cost is fed to ACI so the
ceiling self-calibrates. Runs are sequential, errors are recorded and skipped, and runs already
stored as ok are not repeated, so stopping and restarting is safe.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from .providers import ModelSpec, Provider, ProviderError, SimProvider, Task
from .report import build_report, save_report, to_markdown
from .store import ObservationStore
from .tasks import feature_frame

_MIN_COST = 1e-9


@dataclass(frozen=True)
class Job:
    task: Task
    model: ModelSpec
    kind: str       # "variance" (R_variance on the primary model) | "cross" (R_model on every model)
    run_idx: int

    @property
    def key(self) -> tuple:
        return (self.task.task_id, self.model.alias, self.kind, self.run_idx)


def plan_jobs(task: Task, models: list[ModelSpec], r_variance: int, r_cross: int) -> list[Job]:
    """Total runs per task = sum over models of r_cross + r_variance on models[0]."""
    jobs = [Job(task, m, "cross", i) for m in models for i in range(r_cross)]
    jobs += [Job(task, models[0], "variance", i) for i in range(r_variance)]
    return jobs


class ObservationLoop:
    def __init__(self, engine, tasks: list[Task], models: list[ModelSpec], providers: dict[str, Provider],
                 store: ObservationStore, r_variance: int = 10, r_cross: int = 3, batch_size: int = 10,
                 report_dir: str | None = None, max_spend: float | None = None,
                 on_event: Callable[[str, dict], None] | None = None):
        if not models:
            raise ValueError("no runnable models")
        self.engine, self.tasks, self.models = engine, tasks, models
        self.providers, self.store = providers, store
        self.r_variance, self.r_cross, self.batch_size = r_variance, r_cross, max(1, batch_size)
        self.report_dir, self.max_spend = report_dir, max_spend
        self.on_event = on_event or (lambda kind, info: None)
        self.spent = 0.0                    # actual spend in this process
        self.last_report: dict | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._columns = list(engine.design.numeric) + list(engine.design.categorical)
        self._replayed = False

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        if self.is_running():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self.run, name="pesh-observe", daemon=True)
        self._thread.start()

    def stop(self, wait: bool = True) -> None:
        self._stop.set()
        if wait and self._thread:
            self._thread.join()

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ loop
    def pending(self, task: Task, done: set) -> list[Job]:
        return [j for j in plan_jobs(task, self.models, self.r_variance, self.r_cross) if j.key not in done]

    def _replay_aci(self) -> None:
        """Rebuild the ACI state from stored runs so a restart continues where it stopped."""
        if self._replayed:
            return
        self._replayed = True
        import json

        import pandas as pd
        prior = self.store.frame()
        for r in prior[prior["status"] == "ok"].itertuples():
            try:
                feats = json.loads(r.features)
                df = pd.DataFrame([{c: feats[c] for c in self._columns}])
            except (TypeError, ValueError, KeyError):
                continue
            self.engine.observe(df, [max(float(r.actual_cost), _MIN_COST)], model_version=r.model)

    def run(self) -> dict:
        """Process every task (in batches); returns the final report. Honors stop()."""
        self._replay_aci()
        done = self.store.done_keys()
        attempted: set = set()
        for b in range(0, len(self.tasks), self.batch_size):
            batch = self.tasks[b:b + self.batch_size]
            for task in batch:
                for job in self.pending(task, done | attempted):
                    if self._stop.is_set() or self._over_budget():
                        return self._finish()
                    attempted.add(job.key)
                    self._run_one(job)
            self._finish(final=False)
        return self._finish()

    def _over_budget(self) -> bool:
        if self.max_spend is not None and self.spent >= self.max_spend:
            self.on_event("budget", {"spent": self.spent, "max_spend": self.max_spend})
            self._stop.set()
            return True
        return False

    def _run_one(self, job: Job) -> dict:
        task, model = job.task, job.model
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        base = {"task_id": task.task_id, "kind": job.kind, "run_idx": job.run_idx, "model": model.alias,
                "provider": model.provider if model.provider in self.providers else "sim",
                "features": {c: task.features[c] for c in self._columns if c in task.features},
                "timestamp": ts}
        try:
            df = feature_frame(task, self._columns)
            q = self.engine.quote(df, online=True, model_version=model.alias).iloc[0]
            predicted, ceiling = float(q["p50"]), float(q["quote"])
        except Exception as e:      # a bad row must not stop the loop
            return self._error(base, f"predict failed: {type(e).__name__}: {e}")
        base.update(predicted_cost=predicted, predicted_ceiling=ceiling)

        provider = self.providers.get(model.provider) or SimProvider()
        base["provider"] = provider.name
        try:
            res = provider.run(model, task, run_idx=job.run_idx, kind=job.kind)
        except ProviderError as e:
            return self._error(base, str(e))
        except Exception as e:      # adapters should wrap errors, but never let one escape
            return self._error(base, f"{type(e).__name__}: {e}")

        actual = float(res.cost)
        self.spent += actual
        err = abs(predicted - actual)
        obs = {**base, "status": "ok", "actual_cost": actual, "actual_input_tokens": res.input_tokens,
               "actual_output_tokens": res.output_tokens, "wall_ms": res.wall_ms, "steps": res.steps,
               "success": res.success, "within_ceiling": actual <= ceiling, "abs_error": err,
               "pct_error": err / actual if actual > 0 else None}
        self.store.record(obs)
        self.engine.observe(df, [max(actual, _MIN_COST)], model_version=model.alias)   # ACI self-calibration
        self.on_event("run", obs)
        return obs

    def _error(self, base: dict, msg: str) -> dict:
        obs = {**base, "status": "error", "error": msg[:500]}
        self.store.record(obs)
        self.on_event("error", obs)
        return obs

    # ------------------------------------------------------------------ report
    def report(self) -> tuple[dict, object]:
        """(report, observations) over everything collected so far."""
        obs = self.store.frame()
        return build_report(obs, self.engine, primary=self.models[0].alias), obs

    def _finish(self, final: bool = True) -> dict:
        rep, obs = self.report()
        self.last_report = rep
        if self.report_dir:
            save_report(rep, obs, self.report_dir)
        self.on_event("batch" if not final else "final", {"report": rep, "markdown": to_markdown(rep, obs, 10)})
        return rep
