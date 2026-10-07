"""Observation tab: drives the existing observation loop (`pesh.observe`) from the dashboard.

Thin layer only. Tasks are parsed with `observe.load_tasks`, models with `resolve_models`,
providers with `build_providers`, the loop is `ObservationLoop` (its own background thread,
`start()` / `stop()`), rows go to `ObservationStore` in the dashboard's SQLite file, and the
summary cards are `observe.build_report`.

API keys are kept in a local `.env` file (mode 0600) and in memory. They are never sent back
to the browser: the page only sees which providers have a key and its last four characters.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..observe import (DEFAULT_MODELS, ObservationLoop, ObservationStore, SimProvider, build_providers,
                       build_report, load_tasks, plan_jobs, resolve_models)
from ..observe.providers import KEY_ENV

PROVIDER_LABELS = {"anthropic": "Anthropic (Claude)", "openai": "OpenAI (GPT)",
                   "together": "Together AI (open models)", "fireworks": "Fireworks AI (open models)"}
MODEL_LABELS = {"claude": "Claude Haiku", "gpt": "GPT-4o mini", "together-open": "Llama 3.3 70B on Together",
                "fireworks-open": "Llama 3.3 70B on Fireworks"}
SAMPLE_TASKS = [
    "Summarize the following in three bullet points: PESH quotes the cost of an AI agent run before it starts, "
    "enforces a ceiling while it runs, and prices a guarantee.",
    "In `utils/pagination.py` the last page is dropped when the item count is an exact multiple of the page size. "
    "Explain the bug and give the corrected function.",
    "Refactor a Python module that mixes database access, HTTP handling and business logic across several files "
    "into a layered design. Describe the new module layout, the interfaces between layers, and a migration plan.",
]
_KEY_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)\s*=\s*(.*?)\s*$")


# ----------------------------------------------------------------------------- keys

class KeyVault:
    """Provider keys from the environment and a local .env file; writes only the four key names."""

    def __init__(self, env_path: str | Path):
        self.path = Path(env_path)
        self.lock = threading.Lock()
        self.saved: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                m = _KEY_LINE.match(line)
                if m and m.group(1) in KEY_ENV.values():
                    self.saved[m.group(1)] = m.group(2).strip().strip("'\"")

    def env(self) -> dict[str, str]:
        out = {k: v for k, v in os.environ.items() if k in KEY_ENV.values() and v.strip()}
        out.update({k: v for k, v in self.saved.items() if v})
        return out

    def status(self) -> list[dict]:
        env, rows = self.env(), []
        for prov, var in KEY_ENV.items():
            key = env.get(var, "")
            source = "saved" if self.saved.get(var) else ("environment" if key else None)
            rows.append({"provider": prov, "label": PROVIDER_LABELS[prov], "env": var, "connected": bool(key),
                         "hint": f"…{key[-4:]}" if len(key) >= 8 else ("set" if key else ""), "source": source})
        return rows

    def update(self, values: dict[str, str | None]) -> None:
        with self.lock:
            for var, v in values.items():
                if var not in KEY_ENV.values():
                    raise HTTPException(422, f"Unknown key name {var}.")
                if v is None:
                    continue
                v = v.strip()
                if v and (any(c.isspace() for c in v) or len(v) < 8):
                    raise HTTPException(422, f"That doesn't look like a valid {var}. Paste the whole key.")
                if v:
                    self.saved[var] = v
                else:
                    self.saved.pop(var, None)
            self._write()

    def _write(self) -> None:
        keep = []
        if self.path.exists():
            keep = [ln for ln in self.path.read_text().splitlines()
                    if not ((m := _KEY_LINE.match(ln)) and m.group(1) in KEY_ENV.values())]
        lines = keep + [f"{k}={v}" for k, v in self.saved.items() if v]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("\n".join(lines) + ("\n" if lines else ""))
        try:
            os.chmod(self.path, 0o600)
        except OSError:         # Windows ignores POSIX modes; the file stays in the user's own folder
            pass


# ----------------------------------------------------------------------------- tasks

def parse_tasks(text: str) -> list:
    """JSONL rows ({task_id, text, [features]}) or plain prompts, one per line; via observe.load_tasks."""
    rows = []
    for n, line in enumerate(raw.strip() for raw in text.splitlines()):
        if not line:
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                raise HTTPException(422, f"Line {n + 1} starts with {{ but isn't valid JSON.") from None
            if "text" not in obj:
                raise HTTPException(422, f"Line {n + 1} needs a \"text\" field.")
            obj.setdefault("task_id", "task-" + hashlib.sha1(obj["text"].encode()).hexdigest()[:8])
            rows.append(obj)
        else:   # a stable id from the text, so a rerun of the same prompt resumes instead of repeating
            rows.append({"task_id": "task-" + hashlib.sha1(line.encode()).hexdigest()[:8], "text": line})
    if not rows:
        raise HTTPException(422, "Add at least one task: one prompt per line, or JSONL rows.")
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write("\n".join(json.dumps(r) for r in rows))
        path = fh.name
    try:
        return load_tasks(path)
    except ValueError as e:
        raise HTTPException(422, str(e).replace(path, "tasks")) from None
    finally:
        os.unlink(path)


class PacedSimProvider(SimProvider):
    """The existing simulator, with a short pause so the table visibly fills row by row."""

    def __init__(self, pause: float = 0.25):
        super().__init__()
        self.pause = pause

    def run(self, *a, **kw):
        time.sleep(self.pause)
        return super().run(*a, **kw)


class StartRequest(BaseModel):
    tasks: str
    models: list[str] = Field(min_length=1)
    variance_runs: int = Field(5, ge=0, le=50)
    cross_runs: int = Field(2, ge=1, le=20)
    max_spend: float | None = Field(1.0, gt=0)
    simulate: bool = False


class KeysRequest(BaseModel):
    keys: dict[str, str | None]


def observations_as_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Observation rows -> the generic run-log schema the Analyze screen reads."""
    ok = df[(df["status"] == "ok") & (df["actual_cost"] > 0)].copy()
    feats = ok["features"].map(lambda s: json.loads(s) if isinstance(s, str) and s else {})
    out = pd.DataFrame({
        "task_id": ok["model"] + "::" + ok["task_id"].astype(str),
        "run_id": ok["kind"] + "-" + ok["run_idx"].astype(str),
        "cost": ok["actual_cost"].astype(float),
        "success": ok["success"].fillna(False).astype(bool),
        "steps": ok["steps"],
        "model_version": ok["model"],
    })
    for c in ("x_repo", "x_complex", "human_label"):
        vals = feats.map(lambda f, c=c: f.get(c))
        if vals.notna().all():
            out[c] = vals.values
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------- routes

def add_observation_routes(app: FastAPI, engine, db_path: str, env_path: str | Path = ".env",
                           report_dir: str = "reports/observe", sim_pause: float = 0.25) -> None:
    vault = KeyVault(env_path)
    store = ObservationStore(db_path)
    state: dict = {"loop": None, "mode": None, "planned": 0, "skipped": [], "started": None, "task_ids": [],
                   "events": 0, "budget_hit": False, "texts": {}}
    app.state.observation_store = store
    app.state.key_vault = vault

    def _event(kind: str, info: dict) -> None:
        state["events"] += 1
        if kind == "budget":
            state["budget_hit"] = True

    @app.get("/dashboard/observe/keys")
    def keys():
        rows = vault.status()
        return {"providers": rows, "any_connected": any(r["connected"] for r in rows), "file": str(vault.path)}

    @app.post("/dashboard/observe/keys")
    def save_keys(req: KeysRequest):
        vault.update(req.keys)
        return keys()

    @app.get("/dashboard/observe/setup")
    def setup():
        env = vault.env()
        models = [{"alias": a, "label": MODEL_LABELS.get(a, a), "provider": m.provider,
                   "provider_label": PROVIDER_LABELS[m.provider], "connected": bool(env.get(KEY_ENV[m.provider])),
                   "price_in_per_mtok": m.price_in * 1e6, "price_out_per_mtok": m.price_out * 1e6}
                  for a, m in DEFAULT_MODELS.items()]
        return {"models": models, "sample_tasks": SAMPLE_TASKS}

    @app.post("/dashboard/observe/start")
    def start(req: StartRequest):
        loop = state["loop"]
        if loop is not None and loop.is_running():
            raise HTTPException(409, "An observation is already running. Stop it first.")
        tasks = parse_tasks(req.tasks)
        try:
            specs = resolve_models(req.models)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        providers = {} if req.simulate else build_providers(env=vault.env())
        runnable = [m for m in specs if m.provider in providers]
        skipped = [m.alias for m in specs if m not in runnable] if runnable else []
        if runnable:
            mode = "live"
        else:
            mode, runnable = "simulated", specs
            providers = {m.provider: PacedSimProvider(sim_pause) for m in specs}
        if req.variance_runs == 0 and req.cross_runs == 0:
            raise HTTPException(422, "Set at least one run.")
        loop = ObservationLoop(engine, tasks, runnable, providers, store, r_variance=req.variance_runs,
                               r_cross=req.cross_runs, report_dir=report_dir,
                               max_spend=req.max_spend if mode == "live" else None, on_event=_event)
        planned = sum(len(plan_jobs(t, runnable, req.variance_runs, req.cross_runs)) for t in tasks)
        state.update(loop=loop, mode=mode, planned=planned, skipped=skipped, started=time.time(),
                     task_ids=[t.task_id for t in tasks], budget_hit=False)
        state["texts"].update({t.task_id: t.text for t in tasks})
        loop.start()
        return {"mode": mode, "planned": planned, "models": [m.alias for m in runnable], "skipped": skipped,
                "tasks": len(tasks)}

    @app.post("/dashboard/observe/stop")
    def stop():
        loop = state["loop"]
        if loop is not None:
            loop.stop(wait=False)       # the call in flight finishes, then the loop returns
        return {"stopping": bool(loop and loop.is_running())}

    @app.get("/dashboard/observe/status")
    def status(limit: int = 200):
        loop = state["loop"]
        df = store.frame()
        # summary cards describe the current batch only: its tasks AND the models it ran, so runs of
        # other models left in the SQLite file from earlier batches can't leak in (e.g. a stale
        # "cheapest model" for a model that wasn't ticked this time)
        models = [m.alias for m in loop.models] if loop else []
        current = df[df["task_id"].isin(state["task_ids"]) & df["model"].isin(models)] if loop else df.iloc[0:0]
        rep = build_report(current, engine, primary=models[0]) if len(current) else {}
        rows = df.tail(limit).iloc[::-1]
        table = [{"task_id": r.task_id, "task_text": state["texts"].get(r.task_id), "model": r.model, "kind": r.kind, "run": int(r.run_idx),
                  "provider": r.provider, "status": r.status,
                  "predicted": None if pd.isna(r.predicted_cost) else float(r.predicted_cost),
                  "ceiling": None if pd.isna(r.predicted_ceiling) else float(r.predicted_ceiling),
                  "actual": None if pd.isna(r.actual_cost) else float(r.actual_cost),
                  "within": None if r.within_ceiling is None else bool(r.within_ceiling),
                  "error": r.error if isinstance(r.error, str) else None} for r in rows.itertuples()]
        rho = rep.get("rho_tau")
        return {
            "running": bool(loop and loop.is_running()), "mode": state["mode"], "planned": state["planned"],
            "done_this_run": int(len(current)), "skipped": state["skipped"], "budget_hit": state["budget_hit"],
            "spent": float(loop.spent) if loop else 0.0, "total_observations": int(len(df)),
            "summary": {
                "rho_tau": rho["rho"] if rho else None,
                "mean_pct_error": (rep.get("accuracy") or {}).get("mean_pct_error"),
                "coverage": (rep.get("coverage") or {}).get("realised"),
                "coverage_target": (rep.get("coverage") or {}).get("target", 0.9),
                "cheapest": rep.get("cheapest_model_per_task", {}),
                "n_ok": rep.get("n_ok", 0), "n_errors": rep.get("n_errors", 0),
            },
            "rows": table,
        }
