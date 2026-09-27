"""OpenHands / SWE-bench trajectories -> generic step logs.

OpenHands event streams attach `llm_metrics` to agent events, containing
`accumulated_cost` and `accumulated_token_usage` (prompt_tokens, completion_tokens,
cache_read_tokens, cache_write_tokens) and, in newer versions, a per-call `token_usages`
list. Per-step usage is taken from `token_usages` when present, otherwise from successive
differences of the accumulated counters. Repeated file views (the loop signal of Bai et
al., 2026) are counted from `read` / editor `view` actions on an already-viewed path.

Field names differ across OpenHands versions; `FIELD_MAP` can be overridden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FIELD_MAP = {
    "metrics": "llm_metrics",
    "acc_cost": "accumulated_cost",
    "acc_usage": "accumulated_token_usage",
    "per_call": "token_usages",
    "prompt": "prompt_tokens",
    "completion": "completion_tokens",
    "cache_read": "cache_read_tokens",
    "cache_write": "cache_write_tokens",
}


def _viewed_path(ev: dict) -> str | None:
    args = ev.get("args") or {}
    if ev.get("action") == "read" or args.get("command") == "view":
        return args.get("path")
    return None


def parse_trajectory(events: list[dict], task_id, run_id, fm: dict = FIELD_MAP) -> pd.DataFrame:
    rows = []
    n_calls = 0
    seen: set = set()
    repeats = 0
    prev = {"cost": 0.0, "prompt": 0, "completion": 0, "cache_read": 0, "cache_write": 0}
    for ev in events:
        path = _viewed_path(ev)
        if path:
            repeats += int(path in seen)
            seen.add(path)
        m = ev.get(fm["metrics"])
        if not m:
            continue
        per_call = m.get(fm["per_call"])
        if per_call:
            calls = per_call[n_calls:]  # token_usages is cumulative: keep only new calls
            n_calls = len(per_call)
            for u in calls:
                rows.append({"input_tokens": u.get(fm["prompt"], 0), "output_tokens": u.get(fm["completion"], 0),
                             "cache_read_tokens": u.get(fm["cache_read"], 0),
                             "cache_write_tokens": u.get(fm["cache_write"], 0), "cost": None,
                             "repeated_views": repeats})
                repeats = 0
            continue
        acc = m.get(fm["acc_usage"]) or {}
        cur = {"cost": float(m.get(fm["acc_cost"], 0.0)), "prompt": acc.get(fm["prompt"], 0),
               "completion": acc.get(fm["completion"], 0), "cache_read": acc.get(fm["cache_read"], 0),
               "cache_write": acc.get(fm["cache_write"], 0)}
        if cur["prompt"] <= prev["prompt"] and cur["completion"] <= prev["completion"]:
            continue
        rows.append({"input_tokens": cur["prompt"] - prev["prompt"],
                     "output_tokens": cur["completion"] - prev["completion"],
                     "cache_read_tokens": cur["cache_read"] - prev["cache_read"],
                     "cache_write_tokens": cur["cache_write"] - prev["cache_write"],
                     "cost": (cur["cost"] - prev["cost"]) if cur["cost"] > 0 else None,
                     "repeated_views": repeats})
        repeats = 0
        prev = cur
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df.insert(0, "step", range(1, len(df) + 1))
    df.insert(0, "run_id", run_id)
    df.insert(0, "task_id", task_id)
    return df


def load_openhands_dir(root: str | Path, outcomes: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse <root>/<instance_id>/<run_id>.json trajectories (or <root>/<instance_id>.json).

    outcomes: {instance_id: resolved_bool} or {(instance_id, run_id): resolved_bool}.
    Returns (steps, outcomes_frame).
    """
    root = Path(root)
    frames, outs = [], []
    for f in sorted(root.rglob("*.json")):
        task = f.parent.name if f.parent != root else f.stem
        run = f.stem if f.parent != root else 0
        obj = json.loads(f.read_text())
        events = obj.get("history", obj) if isinstance(obj, dict) else obj
        df = parse_trajectory(events, task, run)
        if df.empty:
            continue
        frames.append(df)
        if outcomes is not None:
            ok = outcomes.get((task, run), outcomes.get(task))
            if ok is not None:
                outs.append({"task_id": task, "run_id": run, "success": bool(ok)})
    steps = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return steps, pd.DataFrame(outs)
