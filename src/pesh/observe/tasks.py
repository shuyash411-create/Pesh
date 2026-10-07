"""Task input: JSONL rows ``{task_id, text, [features]}``.

The quote engine only sees *pre-execution* features. When a row has none, a crude
text heuristic fills the engine's standard columns (x_repo, x_complex, human_label).
It is a placeholder: supply real ``features`` (or train an engine on your own columns)
for meaningful predictions.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd

from .providers import Task

_COMPLEX = re.compile(r"\b(refactor|implement|debug|migrate|design|optimi[sz]e|concurren\w*|race|distributed|"
                      r"multi|several|across|integrat\w*|benchmark|prove|derive)\b", re.I)
_REPO = re.compile(r"(`[^`]+`|\b\w+\.(py|js|ts|go|rs|java|cpp)\b|\brepo\w*\b|\bmodule\b|\bfiles?\b|/\w+)", re.I)


def _clip(v: float, lim: float = 2.0) -> float:
    return max(-lim, min(lim, v))


def text_features(text: str) -> dict:
    """Heuristic stand-ins for the engine's standardised features (roughly mean 0, sd 1)."""
    words = len(text.split())
    x_complex = _clip(0.6 * (math.log1p(words) - math.log(40)) + 0.5 * len(_COMPLEX.findall(text)) - 0.3)
    x_repo = _clip(0.5 * len(_REPO.findall(text)) - 0.5)
    label = 0 if x_complex < -0.4 else (1 if x_complex < 0.4 else 2)
    return {"x_repo": x_repo, "x_complex": x_complex, "human_label": label}


def load_tasks(path: str | Path) -> list[Task]:
    tasks, seen = [], set()
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            tid, text = str(row["task_id"]), str(row["text"])
        except (ValueError, KeyError, TypeError):
            raise ValueError(f"{path}:{n}: each row needs JSON fields task_id and text") from None
        if tid in seen:
            raise ValueError(f"{path}:{n}: duplicate task_id {tid!r}")
        seen.add(tid)
        tasks.append(Task(tid, text, {**text_features(text), **(row.get("features") or {})}))
    if not tasks:
        raise ValueError(f"{path}: no tasks")
    return tasks


def feature_frame(task: Task, columns: list[str]) -> pd.DataFrame:
    """One-row frame with exactly the columns the engine was trained on."""
    missing = [c for c in columns if c not in task.features]
    if missing:
        raise ValueError(f"task {task.task_id!r} lacks engine feature(s) {missing}; add them to its 'features'")
    return pd.DataFrame([{c: task.features[c] for c in columns}])
