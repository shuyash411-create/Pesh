"""Batch report: the existing econometrics run on the collected observations. Cost only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..econometrics.icc import icc_anova
from ..econometrics.tails import hill, powerlaw_fit


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _f(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else float(v)


def build_report(obs: pd.DataFrame, engine=None, primary: str | None = None, target: float = 0.90) -> dict:
    """Summarise observations. ``engine`` (optional) adds each model's ACI state."""
    rep: dict = {"n_observations": int(len(obs)), "n_errors": int((obs["status"] == "error").sum())}
    ok = obs[(obs["status"] == "ok") & (obs["actual_cost"] > 0)].copy()
    rep["n_ok"] = int(len(ok))
    if ok.empty:
        return rep
    primary = primary or ok["model"].iloc[0]
    rep["primary_model"] = primary

    # --- run-to-run spread: rho_tau (ICC) on log cost of the variance runs (econometrics/icc.py)
    var = ok[(ok["kind"] == "variance") & (ok["model"] == primary)]
    per = var.groupby("task_id").size()
    var = var[var["task_id"].isin(per[per >= 2].index)]
    if var["task_id"].nunique() >= 2:
        rep["rho_tau"] = icc_anova(np.log(var["actual_cost"]), var["task_id"]).to_dict()
    else:
        rep["rho_tau"] = None

    # --- cost tail (econometrics/tails.py); needs enough runs to say anything
    c = ok["actual_cost"].to_numpy(float)
    tail = {"n": int(len(c))}
    if len(c) >= 20:
        k = max(5, len(c) // 5)
        tail["hill_alpha"] = _f(hill(c, k))
        tail["hill_k"] = int(k)
    if len(c) >= 100:
        tail["powerlaw"] = powerlaw_fit(c)
    rep["tail"] = tail

    # --- prediction accuracy
    ok["abs_error"] = (ok["predicted_cost"] - ok["actual_cost"]).abs()
    ok["pct_error"] = ok["abs_error"] / ok["actual_cost"]
    rep["accuracy"] = {
        "mean_abs_error": _f(ok["abs_error"].mean()), "median_abs_error": _f(ok["abs_error"].median()),
        "mean_pct_error": _f(ok["pct_error"].mean()), "median_pct_error": _f(ok["pct_error"].median()),
        "corr_pred_actual": _corr(ok["predicted_cost"], ok["actual_cost"]),
        "corr_log_pred_actual": _corr(np.log(ok["predicted_cost"]), np.log(ok["actual_cost"])),
    }

    # --- realised coverage of the ceiling vs target
    cov = ok["within_ceiling"].astype(bool)
    rep["coverage"] = {"realised": float(cov.mean()), "target": target, "n": int(len(ok)),
                       "by_model": {m: float(g.astype(bool).mean())
                                    for m, g in ok.groupby("model")["within_ceiling"]}}
    if engine is not None:
        rep["coverage"]["aci"] = {m: engine.aci_for(m).state() for m in ok["model"].unique()}

    # --- per-model summary and cheapest model per task (cross-model runs only: comparable)
    rep["per_model"] = {
        m: {"runs": int(len(g)), "mean_actual_cost": float(g["actual_cost"].mean()),
            "mean_input_tokens": float(g["actual_input_tokens"].mean()),
            "mean_output_tokens": float(g["actual_output_tokens"].mean()),
            "mean_wall_ms": float(g["wall_ms"].mean()),
            "completion_rate": float(g["success"].astype(bool).mean())}
        for m, g in ok.groupby("model")}
    cross = ok[ok["kind"] == "cross"]
    cheapest = {}
    if cross["model"].nunique() > 1:
        mean = cross.groupby(["task_id", "model"])["actual_cost"].mean().unstack()
        mean = mean.dropna(how="any")
        cheapest = mean.idxmin(axis=1).to_dict()
    rep["cheapest_model_per_task"] = cheapest
    return rep


def _fmt(v, p=4):
    return "n/a" if v is None else f"{v:.{p}g}"


def to_markdown(rep: dict, obs: pd.DataFrame | None = None, max_rows: int = 50) -> str:
    L = ["# PESH observation report", "",
         "_Cost only: no output-quality scoring. `success` means the call completed._", "",
         f"- observations: **{rep['n_observations']}** ({rep.get('n_ok', 0)} ok, {rep['n_errors']} errors)"]
    if rep.get("n_ok", 0) == 0:
        return "\n".join(L + ["", "No successful runs yet."]) + "\n"
    rho = rep["rho_tau"]
    L += ["", "## Run-to-run predictability (ICC, variance runs)", ""]
    if rho:
        L.append(f"- rho_tau = **{rho['rho']:.3f}** (between-task var {rho['sigma2_between']:.3g}, within-task "
                 f"var {rho['sigma2_within']:.3g}); correlation ceiling {rho['corr_ceiling']:.3f}; "
                 f"{rho['n_tasks']} tasks, {rho['n_runs']} runs")
    else:
        L.append("- not enough variance runs (need >= 2 tasks with >= 2 runs on the primary model)")
    t = rep["tail"]
    L += ["", "## Cost tail", ""]
    if "hill_alpha" in t:
        L.append(f"- Hill tail index alpha = **{t['hill_alpha']:.2f}** (k={t['hill_k']} of n={t['n']}); "
                 + ("alpha < 2: infinite variance" if t["hill_alpha"] < 2 else "alpha >= 2"))
    else:
        L.append(f"- n={t['n']} runs; need >= 20 for a Hill estimate")
    if t.get("powerlaw"):
        pl = t["powerlaw"]
        L.append(f"- Clauset power-law alpha = {pl['alpha']:.2f} (xmin={pl['xmin']:.3g}, n_tail={pl['n_tail']})")
    a = rep["accuracy"]
    L += ["", "## Prediction accuracy (predicted p50 vs actual)", "",
          f"- abs error: mean ${_fmt(a['mean_abs_error'])}, median ${_fmt(a['median_abs_error'])}",
          f"- pct error: mean {_fmt(a['mean_pct_error'])}, median {_fmt(a['median_pct_error'])} (fraction of actual)",
          f"- correlation predicted vs actual: {_fmt(a['corr_pred_actual'], 3)} "
          f"(log scale {_fmt(a['corr_log_pred_actual'], 3)})"]
    cv = rep["coverage"]
    L += ["", "## Ceiling coverage", "",
          f"- realised **{cv['realised']:.1%}** vs target {cv['target']:.0%} over {cv['n']} runs"]
    for m, v in cv["by_model"].items():
        s = cv.get("aci", {}).get(m)
        L.append(f"  - {m}: {v:.1%}" + (f" (ACI alpha_t={s['alpha_t']:.3f}, updates={s['n_updates']})" if s else ""))
    L += ["", "## Per-model summary", "",
          "| model | runs | mean cost ($) | mean in tok | mean out tok | mean wall (ms) | completed |",
          "|---|---|---|---|---|---|---|"]
    for m, s in rep["per_model"].items():
        L.append(f"| {m} | {s['runs']} | {s['mean_actual_cost']:.6f} | {s['mean_input_tokens']:.0f} | "
                 f"{s['mean_output_tokens']:.0f} | {s['mean_wall_ms']:.0f} | {s['completion_rate']:.0%} |")
    ch = rep["cheapest_model_per_task"]
    if ch:
        wins = pd.Series(ch).value_counts()
        L += ["", "## Cheapest model per task (cross-model runs)", "",
              "wins: " + ", ".join(f"{m}={n}" for m, n in wins.items()), ""]
        L += [f"- {tid}: {m}" for tid, m in list(ch.items())[:max_rows]]
    if obs is not None and len(obs):
        cols = ["task_id", "kind", "run_idx", "model", "status", "predicted_cost", "predicted_ceiling",
                "actual_cost", "within_ceiling", "pct_error"]
        raw = obs[cols].tail(max_rows).to_markdown(index=False, floatfmt=".5g") if _has_tabulate() \
            else obs[cols].tail(max_rows).to_string(index=False)
        L += ["", f"## Raw table (last {min(max_rows, len(obs))} of {len(obs)}; full data in observations.csv)",
              "", raw]
    return "\n".join(L) + "\n"


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def save_report(rep: dict, obs: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"markdown": out / "report.md", "json": out / "report.json", "csv": out / "observations.csv",
             "parquet": out / "observations.parquet"}
    paths["markdown"].write_text(to_markdown(rep, obs))
    paths["json"].write_text(json.dumps(rep, indent=2, default=float))
    obs.to_csv(paths["csv"], index=False)
    obs.to_parquet(paths["parquet"], index=False)
    return paths
