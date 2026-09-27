"""Markdown + PNG report of the E0-E5 reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# reference categorical palette (fixed order) and recessive text/grid inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

PAPER = {
    "E1": {"rho_tau": 0.804, "feature_r2_log": 0.422, "feature_corr_level": 0.551, "oracle_r2_log": 0.776},
    "E2_cqr": {"marginal": 0.902, "easy": 0.980, "hard": 0.788},
    "E2b": {"static_pre": 0.899, "static_post": 0.704, "aci_post": 0.900},
    "E3": {"naive_bias_repo": -0.40, "powell_bias_repo": -0.06},
    "E4_P4": {"mean_cost_change": -0.41, "cvar95_change": -0.76, "success_change_pp": -4.1},
}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.set_facecolor(SURFACE)


def _fig(plt):
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=130)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def make_figures(res: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []

    # E2 coverage by difficulty tercile
    t = res["E2"]["table"]
    fig, ax = _fig(plt)
    cols = ["marginal", "easy", "mid", "hard"]
    x = np.arange(len(t))
    w = 0.2
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1.5) * w, t[c], w - 0.02, color=SERIES[i], label=c)
    ax.axhline(1 - res["E2"]["alpha"], color=INK2, linestyle="--", linewidth=1)
    ax.set_xticks(x, [n.replace(" (", "\n(").replace("Gaussian ", "Gaussian\n") for n in t.index], fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    _style(ax, "E2  Coverage of a 90% pre-run quote, by latent difficulty", "", "P(C <= quote)")
    fig.tight_layout()
    fig.savefig(out / "e2_coverage.png")
    plt.close(fig)
    files.append("e2_coverage.png")

    # E2b drift
    p = res["E2b"]["periods"]
    fig, ax = _fig(plt)
    ax.plot(p["period"], p["static"], color=SERIES[0], linewidth=2, label="static split-conformal")
    ax.plot(p["period"], p["aci"], color=SERIES[1], linewidth=2, label="adaptive conformal (ACI)")
    ax.axhline(0.9, color=INK2, linestyle="--", linewidth=1)
    shift = int(p.loc[p["shifted"], "period"].min())
    ax.axvline(shift, color=INK2, linewidth=1, linestyle=":")
    ax.text(shift + 0.5, 0.62, "model update", color=INK2, fontsize=8)
    ax.set_ylim(0.6, 1.0)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    _style(ax, "E2b  Quote validity under a provider-side model update", "period", "coverage in period")
    fig.tight_layout()
    fig.savefig(out / "e2b_drift.png")
    plt.close(fig)
    files.append("e2b_drift.png")

    # E4 controllers: cost vs success
    t4 = res["E4"]["table"]
    fig, ax = _fig(plt)
    for i, (name, row) in enumerate(t4.iterrows()):
        ax.scatter(row["mean_cost"], row["success"], s=30 + 12 * row["cvar95"], color=SERIES[i % 5],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"{name.split(' ')[0]}  CVaR95 ${row['cvar95']:.1f}", (row["mean_cost"], row["success"]),
                    textcoords="offset points", xytext=(8, -3), fontsize=8, color=INK)
    _style(ax, f"E4  Execution policies at ceiling ${res['E4']['B']:.2f}", "mean cost per run (USD)",
           "success rate")
    fig.tight_layout()
    fig.savefig(out / "e4_controllers.png")
    plt.close(fig)
    files.append("e4_controllers.png")

    # E5 pooling
    pool = res["E5"]["pooling"]
    fig, ax = _fig(plt)
    for i, (k, lab) in enumerate([("uncapped", "uncapped"), ("capped", "capped at p90"),
                                  ("capped_shock", "capped + common shock")]):
        d = pool[k]
        ax.plot(list(d.keys()), list(d.values()), marker="o", markersize=5, linewidth=2, color=SERIES[i],
                label=f"{lab} (slope {pool['slope_' + k]:.2f})")
    ns = np.array(pool["sizes"], float)
    ref = pool["capped"][pool["sizes"][0]] * np.sqrt(ns[0] / ns)
    ax.plot(ns, ref, color=INK2, linestyle="--", linewidth=1, label="N^-1/2 reference")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    _style(ax, "E5  Pooling needs truncation; common shocks set a floor", "tasks in portfolio N",
           "99% loading / expected cost")
    fig.tight_layout()
    fig.savefig(out / "e5_pooling.png")
    plt.close(fig)
    files.append("e5_pooling.png")
    return files


def _md_table(df: pd.DataFrame, digits: int = 3) -> str:
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        vals = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row.values]
        lines.append("| " + " | ".join([str(idx)] + vals) + " |")
    return "\n".join(lines)


def _f(x, d=3):
    if isinstance(x, (tuple, list)):
        return "–".join(f"{v:.2f}" for v in x)
    return f"{x:.{d}f}"


def write_report(res: dict, out_dir: str | Path, figures: bool = True) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = make_figures(res, out) if figures else []
    e0, e1, e2, e2b, e3, e4, e5 = (res[k] for k in ["E0", "E1", "E2", "E2b", "E3", "E4", "E5"])
    L = ["# Pesh — reproduction of the simulation study (E0–E5)", "",
         "All numbers are **simulated** from the SMM-calibrated DGP (`pesh.sim`), not empirical estimates. "
         "Paper values are from Tiwari (2026), Section 5.", ""]

    L += ["## E0 — Calibration vs published moments (Table 4)", "", "| moment | target | simulated |", "|---|---|---|"]
    for k, v in e0["targets"].items():
        L.append(f"| {k} | {v} | {e0['simulated'][k]:.3f} |")
    L.append(f"| median / p90 / p99 cost | 1.18 / 4.22 / 15.55 (paper sim) | "
             f"{e0['simulated']['median_cost']:.2f} / {e0['simulated']['p90_cost']:.2f} / "
             f"{e0['simulated']['p99_cost']:.2f} |")

    L += ["", "## E1 — Predictability ceiling (Proposition 3)", "", "| quantity | paper | here |", "|---|---|---|"]
    for k in ["rho_tau", "feature_r2_log", "feature_corr_level", "oracle_r2_log"]:
        L.append(f"| {k} | {PAPER['E1'][k]} | {e1[k]:.3f} |")
    L.append(f"| corr ceiling sqrt(rho) | 0.897 | {e1['corr_ceiling']:.3f} |")

    L += ["", "## E2 — Pre-run quotes, target 90% (Table 5)", "", _md_table(e2["table"]), ""]
    if "finite_sample" in e2:
        L += ["Finite-sample behaviour of split conformal (Figure 4):", "",
              _md_table(e2["finite_sample"].set_index("n")), ""]
    if "e2_coverage.png" in files:
        L += ["![E2](e2_coverage.png)", ""]

    L += ["## E2b — Provider-side model update (Figure 5)", "",
          "| | paper | here |", "|---|---|---|",
          f"| static, pre-shift | 0.899 | {e2b['static_pre']:.3f} |",
          f"| static, post-shift | 0.704 | {e2b['static_post']:.3f} |",
          f"| ACI, post-shift | 0.900 | {e2b['aci_post']:.3f} |",
          f"| ACI, first 5 post-shift periods | — | {e2b['aci_post_first5']:.3f} |", ""]
    if "e2b_drift.png" in files:
        L += ["![E2b](e2b_drift.png)", ""]

    L += [f"## E3 — Caps censor the training data (Table 6), tau={e3['tau']}, "
          f"{e3['censored_share']:.0%} censored, Powell converged in {e3['powell_iterations']} iterations", "",
          _md_table(e3["table"]), ""]

    L += [f"## E4 — Execution controllers at B = ${e4['B']:.2f} (Table 7)", "",
          _md_table(e4["table"][["mean_cost", "cvar95", "p_exceed_B", "success", "cost_per_success",
                                 "mean_cost_change", "cvar95_change", "success_change_pp"]]), "",
          "Paper, P4: mean cost −41%, CVaR95 −76%, success −4.1pp.", ""]
    if "e4_controllers.png" in files:
        L += ["![E4](e4_controllers.png)", ""]

    m = e5["mechanism"]
    L += ["## E5 — Tails, pooling and the value of a ceiling", "",
          "| tail index (Hill) | steps | cost | cost/steps |", "|---|---|---|---|",
          f"| mechanism (Pareto steps, alpha_T=2.2) | {_f(m['alpha_steps'])} | {_f(m['alpha_cost'])} | "
          f"{_f(m['ratio'])} |",
          f"| calibrated DGP, no window | {_f(e5['calibrated_no_window']['alpha_steps'])} | "
          f"{_f(e5['calibrated_no_window']['alpha_cost'])} | |",
          f"| calibrated DGP, 300k window | {_f(e5['calibrated_300k_window']['alpha_steps'])} | "
          f"{_f(e5['calibrated_300k_window']['alpha_cost'])} | |", "",
          f"Pooling log-log slopes: uncapped {e5['pooling']['slope_uncapped']:.2f} (paper −0.02), "
          f"capped {e5['pooling']['slope_capped']:.2f} (paper −0.52), capped + shock "
          f"{e5['pooling']['slope_capped_shock']:.2f} (paper −0.12); floor at N=3000: "
          f"{e5['pooling']['floor_capped_shock']:.2f} (paper ≈0.23).", ""]
    if "e5_pooling.png" in files:
        L += ["![E5](e5_pooling.png)", ""]
    mo = e5["monthly"]
    L += ["Monthly spend, 2,000 tasks/month with a common shock:", "",
          "| | mean | sd | CV | CVaR95 | success |", "|---|---|---|---|---|---|"]
    for k in ["uncapped", "capped"]:
        r = mo[k]
        L.append(f"| {k} | {r['mean']:.0f} | {r['sd']:.0f} | {r['cv']:.3f} | {r['cvar95']:.0f} | {r['success']:.3f} |")
    L += ["", "Willingness to pay for the ceiling, Eq. (12), as a share of uncapped monthly spend:", "",
          "| buyer | WTP share | expected saving | risk saving | quality cost |", "|---|---|---|---|---|"]
    for k, w in e5["wtp"].items():
        L.append(f"| {k} | {w['wtp_share_of_spend']:.1%} | {w['expected_saving']:.0f} | {w['risk_saving']:.0f} | "
                 f"{w['quality_cost']:.0f} |")
    L.append("")
    path = out / "results.md"
    path.write_text("\n".join(L))
    (out / "results.json").write_text(json.dumps(_jsonable(res), indent=2, default=str))
    return path


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return o.reset_index().to_dict(orient="records")
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o
