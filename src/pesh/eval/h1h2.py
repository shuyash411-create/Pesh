"""Milestone M1 report: H1 (predictability ceiling) and H2 (tail index) on real run logs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Reference values: the paper's calibrated simulation (Tiwari, 2026, §5) and Bai et al.'s
# best self-prediction correlation (≈0.39), which H1 compares against the ceiling.
PAPER_SIM = {"rho": 0.804, "corr_ceiling": 0.897, "alpha_cost": (2.42, 2.48), "ratio": 0.56}
SELF_PREDICTION_CORR = 0.39


def _ci(row, key, digits=2) -> str:
    v, lo, hi = row.get(key), row.get(f"{key}_lo"), row.get(f"{key}_hi")
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if lo is None or (isinstance(lo, float) and np.isnan(lo)):
        return f"{v:.{digits}f}"
    return f"{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def verdicts(row: pd.Series) -> list[str]:
    out = []
    if not np.isnan(row.get("rho", np.nan)):
        gap = row["corr_ceiling_lo"] - SELF_PREDICTION_CORR
        out.append(f"H1: no pre-run predictor can exceed correlation {row['corr_ceiling']:.2f} with log cost "
                   f"(95% CI {row['corr_ceiling_lo']:.2f}–{row['corr_ceiling_hi']:.2f}); "
                   + (f"self-prediction at {SELF_PREDICTION_CORR} leaves an information gap of at least {gap:.2f}."
                      if gap > 0 else "self-prediction is already within the ceiling's interval."))
    if not np.isnan(row.get("alpha_cost", np.nan)):
        v = row["variance_verdict"]
        out.append({"infinite": "H2: cost has infinite variance (whole 95% CI for the tail index is below 2).",
                    "finite": "H2: cost variance is finite (whole 95% CI for the tail index is above 2).",
                    "inconclusive": "H2: the 95% CI for the cost tail index straddles 2, so whether variance is "
                                    "finite is not settled at this sample size."}[v])
        cap = row.get("steps_at_cap_share", 0.0)
        if cap and not np.isnan(cap) and cap > 0:
            out.append(f"{cap:.1%} of runs stopped at the step cap; the step tail is estimated on the runs "
                       "below it.")
        if not np.isnan(row.get("ratio", np.nan)):
            lo, hi = row["ratio_lo"], row["ratio_hi"]
            if row["alpha_cost"] >= row["alpha_steps"]:
                out.append("Prop. 2 mechanism rejected here: the cost tail is not heavier than the step tail.")
            elif lo <= 0.5 <= hi:
                out.append(f"Prop. 2 consistent: alpha_C/alpha_T = {row['ratio']:.2f} [{lo:.2f}, {hi:.2f}] "
                           "includes the predicted 0.5.")
            else:
                out.append(f"Prop. 2 partially supported: the cost tail is heavier than the step tail, but the "
                           f"ratio {row['ratio']:.2f} [{lo:.2f}, {hi:.2f}] excludes 0.5.")
    return out


def write_h1h2_report(report: pd.DataFrame, out_dir: str | Path, source: str = "") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    L = ["# M1 — predictability ceiling (H1) and tail index (H2)", "",
         f"Source: {source}" if source else "", "",
         "Intervals are 95% cluster-bootstrap intervals (tasks resampled with all their runs). "
         "Tail indices use the Hill estimator on the top 2% of runs; the sensitivity columns show "
         "how the estimate moves with that choice.", "",
         "| configuration | runs | tasks | runs/task | ρτ | corr ceiling √ρτ | α cost | α steps | α_C/α_T | variance |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for name, row in report.iterrows():
        L.append(f"| {name} | {int(row['n_runs'])} | {int(row['n_tasks'])} | "
                 f"{row.get('runs_per_task', float('nan')):.1f} | {_ci(row, 'rho')} | {_ci(row, 'corr_ceiling')} | "
                 f"{_ci(row, 'alpha_cost')} | {_ci(row, 'alpha_steps')} | {_ci(row, 'ratio')} | "
                 f"{row.get('variance_verdict', '–')} |")
    L += ["", f"Paper's calibrated simulation for comparison: ρτ = {PAPER_SIM['rho']}, "
              f"ceiling {PAPER_SIM['corr_ceiling']}, α_cost {PAPER_SIM['alpha_cost'][0]}–{PAPER_SIM['alpha_cost'][1]}, "
              f"ratio ≈ {PAPER_SIM['ratio']}.", "", "## Verdicts", ""]
    for name, row in report.iterrows():
        L.append(f"**{name}**")
        L += [f"- {v}" for v in verdicts(row)] or ["- not enough data"]
        L.append("")
    path = out / "h1h2.md"
    path.write_text("\n".join(L))
    (out / "h1h2.json").write_text(json.dumps(report.reset_index().to_dict(orient="records"), indent=2,
                                              default=float))
    return path
