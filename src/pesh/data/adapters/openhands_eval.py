"""OpenHands SWE-bench evaluation output (`output.jsonl`) -> run and step logs.

This is the format OpenHands' evaluation harness writes, one JSON object per SWE-bench
instance, and the form in which Bai et al. (2026) report their trajectories (OpenHands
on SWE-bench Verified, eight models, four runs per task). One file holds one model x one
run. Fields used, all optional except `instance_id`:

    instance_id                       -> task_id
    metadata.llm_config.model         -> model_version (else the folder name)
    history[*].llm_metrics            -> per-call token usage, via `openhands.parse_trajectory`
    metrics.token_usages / costs      -> per-call usage when history carries no metrics
    metrics.accumulated_cost          -> provider-reported run cost (preferred over recomputing)
    report.resolved / test_result.report.resolved -> success; otherwise `resolved_ids` from a
        report.json next to the output.jsonl (per run), else from the `reports` passed in
    instance.repo, instance.problem_statement                -> pre-execution features
    instance.patch, instance.FAIL_TO_PASS                    -> difficulty proxies (NOT features)

Run ids come from the path: a `run_<n>` / `run-<n>` / `run<n>` folder or file name if
present, otherwise the files of one model are numbered in sorted order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import PricingParams
from .generic import runs_from_steps
from .openhands import parse_trajectory

_RUN_RE = re.compile(r"run[_-]?(\d+)", re.IGNORECASE)


def _first(d: dict, *paths, default=None):
    for path in paths:
        cur = d
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None:
            return cur
    return default


def _resolved(rec: dict, resolved_ids: set | None) -> bool | None:
    r = _first(rec, "report.resolved", "test_result.report.resolved", "resolved")
    if r is not None:
        return bool(r)
    if resolved_ids is not None:
        return rec.get("instance_id") in resolved_ids
    return None


def _steps_from_metrics(metrics: dict) -> pd.DataFrame:
    usages = metrics.get("token_usages") or []
    costs = [c.get("cost") for c in (metrics.get("costs") or [])]
    if not usages:
        return pd.DataFrame()
    rows = []
    for i, u in enumerate(usages):
        rows.append({"input_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0),
                     "cache_read_tokens": u.get("cache_read_tokens", 0),
                     "cache_write_tokens": u.get("cache_write_tokens", 0),
                     "cost": costs[i] if len(costs) == len(usages) else None, "repeated_views": 0})
    return pd.DataFrame(rows)


def _features(rec: dict) -> dict:
    inst = rec.get("instance") or {}
    statement = inst.get("problem_statement") or rec.get("instruction") or ""
    patch = inst.get("patch") or ""
    f2p = inst.get("FAIL_TO_PASS")
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except ValueError:
            f2p = [f2p]
    return {
        "repo": inst.get("repo", "unknown"),
        "statement_chars": len(statement),
        "log_statement_chars": float(np.log1p(len(statement))),
        # difficulty proxies for H3: known only after the fact, never use them as quote features
        "proxy_gold_patch_lines": sum(1 for line in patch.splitlines() if line.startswith(("+", "-"))
                                      and not line.startswith(("+++", "---"))),
        "proxy_fail_to_pass": len(f2p) if isinstance(f2p, list) else np.nan,
    }


def parse_record(rec: dict, model: str, run_id, resolved_ids: set | None = None) -> tuple[pd.DataFrame, dict]:
    """One output.jsonl line -> (step rows, run-level metadata)."""
    task = rec["instance_id"]
    steps = parse_trajectory(rec.get("history") or [], task, run_id)
    metrics = rec.get("metrics") or {}
    if steps.empty:
        steps = _steps_from_metrics(metrics)
        if not steps.empty:
            steps.insert(0, "step", range(1, len(steps) + 1))
            steps.insert(0, "run_id", run_id)
            steps.insert(0, "task_id", task)
    if not steps.empty:
        steps.insert(2, "model_version", model)
    reported = metrics.get("accumulated_cost")
    meta = {"task_id": task, "run_id": run_id, "model_version": model, "success": _resolved(rec, resolved_ids),
            "reported_cost": float(reported) if reported else np.nan, "error": bool(rec.get("error")),
            **_features(rec)}
    return steps, meta


def _model_of(rec: dict, path: Path) -> str:
    return str(_first(rec, "metadata.llm_config.model", "metadata.model_name", default=path.parent.name))


def _run_of(path: Path) -> int | None:
    for part in reversed(path.parts):
        m = _RUN_RE.search(part)
        if m:
            return int(m.group(1))
    return None


def find_output_files(inputs) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, inputs):
        if p.is_dir():
            files += sorted(p.rglob("output.jsonl"))
        else:
            files.append(p)
    return files


def load_resolved_ids(report_paths) -> set | None:
    """Union of `resolved_ids` / `resolved` lists across SWE-bench harness report.json files."""
    if not report_paths:
        return None
    ids: set = set()
    for rp in report_paths:
        obj = json.loads(Path(rp).read_text())
        ids |= set(obj.get("resolved_ids") or obj.get("resolved") or [])
    return ids


def load_eval_outputs(inputs, reports=None, pricing: PricingParams | None = None,
                      max_records: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse OpenHands eval outputs into (runs, steps).

    `task_id` in both frames is "<model>::<instance_id>", because repeated runs of one task are
    only exchangeable within one agent configuration (Prop. 3); `instance_id` keeps the raw id.

    runs uses the provider-reported `accumulated_cost` when present (`cost_source="reported"`),
    otherwise cost recomputed from tokens with `pricing` (`cost_source="tokens"`).
    Records with no token usage at all are dropped and counted in `runs.attrs["dropped"]`.
    """
    resolved_ids = load_resolved_ids(reports)
    step_frames, metas = [], []
    per_model_count: dict[str, int] = {}
    dropped = 0
    for path in find_output_files(inputs):
        with open(path) as fh:
            lines = [json.loads(line) for line in fh if line.strip()]
        if max_records:
            lines = lines[:max_records]
        if not lines:
            continue
        model = _model_of(lines[0], path)
        sibling = path.parent / "report.json"          # the harness writes one report per run
        file_ids = load_resolved_ids([sibling]) if sibling.exists() else resolved_ids
        run = _run_of(path)
        if run is None:
            run = per_model_count.get(model, 0)
        per_model_count[model] = per_model_count.get(model, 0) + 1
        for rec in lines:
            steps, meta = parse_record(rec, model, run, file_ids)
            if steps.empty:
                dropped += 1
                continue
            step_frames.append(steps)
            metas.append(meta)
    if not metas:
        raise ValueError("no OpenHands records with token usage were found in the given inputs")
    steps = pd.concat(step_frames, ignore_index=True)
    meta = pd.DataFrame(metas)
    # one task can be run by several models: make the (model, task, run) triple the run key
    key = ["task_id", "run_id", "model_version"]
    outcomes = meta[key + ["success"]].copy()
    if outcomes["success"].isna().any():
        missing = int(outcomes["success"].isna().sum())
        raise ValueError(f"{missing} runs have no resolved/unresolved outcome; pass the SWE-bench report.json")
    feature_cols = [c for c in meta.columns if c not in key + ["success"]]
    tokens = steps.assign(task_id=steps["model_version"] + "::" + steps["task_id"].astype(str))
    runs = runs_from_steps(tokens.drop(columns=["model_version"]),
                           outcomes.assign(task_id=outcomes["model_version"] + "::" + outcomes["task_id"].astype(str))
                           .drop(columns=["model_version"]), pricing)
    runs = runs.rename(columns={"cost": "token_cost"})
    runs["model_version"] = runs["task_id"].str.split("::", n=1).str[0]
    runs["instance_id"] = runs["task_id"].str.split("::", n=1).str[1]
    runs = runs.merge(meta.rename(columns={"task_id": "instance_id"})[["instance_id", "run_id", "model_version"]
                                                                     + feature_cols],
                      on=["instance_id", "run_id", "model_version"], how="left")
    runs["cost"] = runs["reported_cost"].where(runs["reported_cost"] > 0, runs["token_cost"])
    runs["cost_source"] = np.where(runs["reported_cost"] > 0, "reported", "tokens")
    runs = runs[runs["cost"] > 0].reset_index(drop=True)
    runs.attrs["dropped"] = dropped
    return runs, tokens
