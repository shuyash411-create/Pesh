"""Reproduction of the paper's simulation study (Section 5), experiments E0-E5.

Every function is deterministic given its seed. `scale` shrinks sample sizes for fast
tests; scale=1 matches the paper's sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..config import ControllerConfig, DGPParams, PricingParams, QuoteConfig
from ..cost_model import expected_cost_given_steps
from ..data.io import split_by_task
from ..econometrics.censored import chernozhukov_hong, powell_cqr
from ..econometrics.icc import icc_anova
from ..econometrics.quantreg import fit_quantile, ols
from ..econometrics.risk import cvar, loglog_slope, pooling_loading, willingness_to_pay
from ..econometrics.tails import hill_range
from ..features import DesignBuilder, add_task_history
from ..quote.aci import ACI
from ..quote.conformal import SplitConformal, theoretical_coverage
from ..quote.engine import QuoteEngine
from ..sim.calibrate import TARGETS, moments
from ..sim.dgp import execute, plan_runs, simulate, simulate_tasks
from ..control.controller import CostController, PolicySet
from ..control.feasibility import train_from_simulation
from ..control.replay import compare_policies
from .metrics import coverage, quote_summary, r2, terciles


def _n(x, scale, lo=50):
    return max(lo, int(round(x * scale)))


# ------------------------------------------------------------------------------- E0

def e0_calibration(scale: float = 1.0, seed: int = 0, params: DGPParams | None = None) -> dict:
    """Simulated moments vs the published targets (paper Table 4)."""
    df = simulate(_n(2000, scale), 4, params, seed=seed)
    m = moments(df)
    m["median_cost"] = float(df["cost"].median())
    m["p90_cost"] = float(df["cost"].quantile(0.9))
    m["p99_cost"] = float(df["cost"].quantile(0.99))
    return {"simulated": m, "targets": TARGETS}


# ------------------------------------------------------------------------------- E1

def e1_predictability(scale: float = 1.0, seed: int = 1, params: DGPParams | None = None) -> dict:
    """ICC ceiling vs a feature predictor vs a replicate-run oracle (Prop. 3)."""
    runs = 8
    df = simulate(_n(3000, scale), runs, params, seed=seed)
    y = np.log(df["cost"].to_numpy())
    icc = icc_anova(y, df["task_id"].to_numpy())

    train, test = split_by_task(df, (0.5, 0.5), seed=seed)
    design = DesignBuilder(["x_repo", "x_complex"], ["human_label"], quadratic=True).fit(train)
    beta = ols(design.transform(train), np.log(train["cost"]))
    f = design.transform(test) @ beta
    yt = np.log(test["cost"].to_numpy())
    smear = np.mean(np.exp(np.log(train["cost"]) - design.transform(train) @ beta))

    hist = add_task_history(test)
    oracle = hist["hist_log_cost"].to_numpy()
    return {
        "rho_tau": icc.rho,
        "sigma2_between": icc.sigma2_between,
        "sigma2_within": icc.sigma2_within,
        "corr_ceiling": icc.corr_ceiling,
        "feature_r2_log": r2(yt, f),
        "feature_corr_log": float(np.corrcoef(yt, f)[0, 1]),
        "feature_corr_level": float(np.corrcoef(test["cost"], np.exp(f) * smear)[0, 1]),
        "oracle_r2_log": r2(yt, oracle),
        "oracle_corr_log": float(np.corrcoef(yt, oracle)[0, 1]),
        "n_tasks": int(df["task_id"].nunique()),
        "runs_per_task": runs,
    }


# ------------------------------------------------------------------------------- E2

def e2_quotes(scale: float = 1.0, seed: int = 2, alpha: float = 0.10, params: DGPParams | None = None,
              finite_sample: bool = True) -> dict:
    """Five quoting methods; marginal and conditional coverage (Table 5, Figs 3-4)."""
    tr = simulate(_n(5000, scale), 1, params, seed=seed * 100 + 1)
    ca = simulate(_n(2500, scale), 1, params, seed=seed * 100 + 2)
    te = simulate(_n(20000, scale), 1, params, seed=seed * 100 + 3)
    z = norm.ppf(1 - alpha)
    design = DesignBuilder(["x_repo", "x_complex"], ["human_label"], quadratic=True).fit(tr)
    Xtr, Xte = design.transform(tr), design.transform(te)
    ytr = np.log(tr["cost"].to_numpy())
    cost = te["cost"].to_numpy()

    b = ols(Xtr, ytr)
    resid = ytr - Xtr @ b
    smear = np.mean(np.exp(resid))
    q_point = np.exp(Xte @ b) * smear * 1.5
    q_gauss = np.exp(Xte @ b + z * resid.std(ddof=Xtr.shape[1]))
    q_qr = np.exp(Xte @ fit_quantile(Xtr, ytr, 1 - alpha))
    cfg = QuoteConfig(alpha=alpha)
    eng = QuoteEngine(cfg).fit(tr, ca)
    q_cqr = eng.quote(te)["quote"].to_numpy()
    eng_m = QuoteEngine(QuoteConfig(alpha=alpha, mondrian_by="human_label")).fit(tr, ca)
    q_mon = eng_m.quote(te)["quote"].to_numpy()

    rows = []
    for name, q in [("Point x1.5", q_point), ("Gaussian log-OLS", q_gauss), ("Quantile regression", q_qr),
                    ("CQR", q_cqr), ("Mondrian CQR (by human label)", q_mon)]:
        rows.append(quote_summary(name, cost, q, te["difficulty"], te["human_label"], te["loop"]))
    out = {"table": pd.DataFrame(rows).set_index("method"), "alpha": alpha,
           "cqr_median_quote_over_median_cost": float(np.median(q_cqr) / np.median(cost))}

    if finite_sample:
        pool = simulate(_n(20000, scale), 1, params, seed=seed * 100 + 4)
        s_pool = np.log(pool["cost"].to_numpy()) - eng.base_log(pool)
        s_test = np.log(cost) - eng.base_log(te)
        rng = np.random.default_rng(seed)
        fs = []
        for n in (30, 100, 300, 1000):
            covs = []
            for _ in range(_n(400, scale, lo=40)):
                sc = SplitConformal(alpha).calibrate(np.zeros(n), s_pool[rng.choice(len(s_pool), n, replace=False)])
                covs.append(float(np.mean(s_test <= sc.global_offset_)))
            fs.append({"n": n, "theory": theoretical_coverage(n, alpha), "mean": float(np.mean(covs)),
                       "p05": float(np.quantile(covs, 0.05)), "p95": float(np.quantile(covs, 0.95))})
        out["finite_sample"] = pd.DataFrame(fs)
    return out


# ------------------------------------------------------------------------------- E2b

def e2b_drift(scale: float = 1.0, seed: int = 3, alpha: float = 0.10, periods: int = 80, per_period: int = 250,
              shift_at: int = 40, verbosity: float = 1.35, step_mult: float = 1.25, gamma: float = 0.01,
              window: int = 500, params: DGPParams | None = None) -> dict:
    """Static split-conformal vs ACI across a provider-side model update (Fig. 5)."""
    per = _n(per_period, scale, lo=100)
    tr = simulate(_n(5000, scale), 1, params, seed=seed * 100 + 1)
    ca = simulate(_n(2500, scale), 1, params, seed=seed * 100 + 2)
    eng = QuoteEngine(QuoteConfig(alpha=alpha)).fit(tr, ca)
    cal_scores = np.log(ca["cost"].to_numpy()) - eng.base_log(ca)
    aci = ACI(alpha, gamma, window, cal_scores)
    static_off = eng.conformal.global_offset_
    rows = []
    for p in range(periods):
        shifted = p >= shift_at
        df = simulate(per, 1, params, seed=seed * 10_000 + p,
                      verbosity=verbosity if shifted else 1.0, step_mult=step_mult if shifted else 1.0)
        s = np.log(df["cost"].to_numpy()) - eng.base_log(df)
        static_cov = float(np.mean(s <= static_off))
        hits = []
        for si in s:
            off = aci.offset()
            hits.append(si <= off)
            aci.update(si, off)
        rows.append({"period": p, "shifted": shifted, "static": static_cov, "aci": float(np.mean(hits)),
                     "alpha_t": aci.alpha_t})
    df = pd.DataFrame(rows)
    pre, post = df[~df.shifted], df[df.shifted]
    return {
        "periods": df,
        "static_pre": float(pre.static.mean()), "static_post": float(post.static.mean()),
        "aci_pre": float(pre.aci.mean()), "aci_post": float(post.aci.mean()),
        "aci_post_first5": float(post.aci.iloc[:5].mean()),
        "aci_post_after5": float(post.aci.iloc[5:].mean()),
    }


# ------------------------------------------------------------------------------- E3

def e3_censoring(scale: float = 1.0, seed: int = 4, cap_q: float = 0.80, tau: float = 0.75,
                 params: DGPParams | None = None) -> dict:
    """Naive QR on capped logs vs Powell censored QR (Table 6, Fig. 6)."""
    df = simulate(_n(8000, scale), 1, params, seed=seed * 100 + 1)
    truth_df = simulate(_n(60000, scale), 1, params, seed=seed * 100 + 2)
    test = simulate(_n(20000, scale), 1, params, seed=seed * 100 + 3)

    def X(d):
        return np.column_stack([np.ones(len(d)), d["x_repo"], d["x_complex"]])

    y = np.log(df["cost"].to_numpy())
    B = float(np.quantile(df["cost"], cap_q))
    y_c = np.minimum(y, np.log(B))
    censored = y >= np.log(B)
    betas = {
        "ground truth": fit_quantile(X(truth_df), np.log(truth_df["cost"]), tau),
        "uncensored, same n": fit_quantile(X(df), y, tau),
        "naive QR on capped logs": fit_quantile(X(df), y_c, tau),
    }
    pw = powell_cqr(X(df), y_c, np.log(B), tau, beta0=betas["naive QR on capped logs"])
    betas["Powell censored QR"] = pw["beta"]
    betas["Chernozhukov-Hong 3-step"] = chernozhukov_hong(X(df), y_c, censored, np.log(B), tau)["beta"]

    hard = terciles(test["difficulty"]) == "hard"
    Xt, ct = X(test)[hard], test["cost"].to_numpy()[hard]
    truth = betas["ground truth"]
    rows = []
    for name, b in betas.items():
        rows.append({
            "estimator": name,
            "slope_repo": float(b[1]),
            "slope_complex": float(b[2]),
            "bias_repo": float(b[1] / truth[1] - 1),
            "bias_complex": float(b[2] / truth[2] - 1),
            "coverage_hard_tercile": coverage(ct, np.exp(Xt @ b)),
        })
    return {"table": pd.DataFrame(rows).set_index("estimator"), "cap": B, "censored_share": float(censored.mean()),
            "powell_iterations": pw["iterations"], "tau": tau}


# ------------------------------------------------------------------------------- E4

def e4_controllers(scale: float = 1.0, seed: int = 5, ceiling_q: float = 0.90,
                   config: ControllerConfig | None = None, params: DGPParams | None = None) -> dict:
    """Five execution policies at a common ceiling (Table 7, Fig. 7)."""
    cfg = config or ControllerConfig()
    fm = train_from_simulation(n_tasks=_n(3000, scale, lo=500), seed=seed * 100 + 7, params=params)
    tasks = simulate_tasks(_n(7500, scale), params, seed)
    plan = plan_runs(tasks, 4, params, seed)
    B = float(np.quantile(execute(plan, None, params).runs["cost"], ceiling_q))
    table = compare_policies(plan, PolicySet(B, fm, cfg).all(), params, config=cfg, B=B)
    return {"table": table, "B": B, "feasibility_auc": fm.metrics.get("auc_in_sample")}


# ------------------------------------------------------------------------------- E5

def e5_tails_pooling(scale: float = 1.0, seed: int = 6, params: DGPParams | None = None) -> dict:
    """Tail-index halving (Prop. 2), pooling (Prop. 6) and the value of a ceiling (Eq. 12)."""
    p = params or DGPParams()
    rng = np.random.default_rng(seed)
    out: dict = {}

    # (a) mechanism isolation: Pareto steps, quadratic accumulation, no window
    n = _n(200_000, scale, lo=20_000)
    T = 20.0 * (1.0 - rng.random(n)) ** (-1.0 / 2.2)
    pr_inf = PricingParams(window=1e15)
    mu_a = np.exp(p.a_log_mean)
    mu_o = np.exp(p.o_log_mean)
    C = expected_cost_given_steps(T, np.exp(p.c0_log_mean), mu_a, mu_o, pr_inf) * np.exp(rng.normal(0, 0.1, n))
    a_t, a_c = hill_range(T), hill_range(C)
    out["mechanism"] = {"alpha_steps": a_t, "alpha_cost": a_c,
                        "ratio": (a_c[0] / a_t[1], a_c[1] / a_t[0])}

    # (b) calibrated DGP, with and without a binding window
    for label, pr in [("calibrated_no_window", pr_inf), ("calibrated_300k_window", PricingParams(window=300_000))]:
        df = simulate(_n(10_000, scale, lo=2000), 4, p, pr, seed=seed * 100 + 1)
        # the 2-3% tail sits below the probability mass that T_max = 250 truncates
        out[label] = {"alpha_steps": hill_range(df["steps"].to_numpy(float), (0.02, 0.03)),
                      "alpha_cost": hill_range(df["cost"].to_numpy(), (0.02, 0.03))}

    # (c) pooling: heavy-tailed mechanism population
    sizes = [10, 30, 100, 300, 1000, 3000]
    reps = _n(2000, scale, lo=400)
    capB = float(np.quantile(C, 0.9))
    Cc = np.minimum(C, capB)
    load_raw = pooling_loading(C, sizes, reps=reps, seed=seed)
    load_cap = pooling_loading(Cc, sizes, reps=reps, seed=seed)
    load_shock = pooling_loading(Cc, sizes, reps=reps, seed=seed, shock_mult=1.4, shock_prob=0.2)
    out["pooling"] = {
        "sizes": sizes,
        "uncapped": load_raw, "capped": load_cap, "capped_shock": load_shock,
        "slope_uncapped": loglog_slope(load_raw), "slope_capped": loglog_slope(load_cap),
        "slope_capped_shock": loglog_slope(load_shock),
        "floor_capped_shock": load_shock[sizes[-1]],
    }

    # (d) monthly budget risk & WTP: uncapped (P0) vs hard cap at p90 (P1), common shock per month
    tasks = simulate_tasks(_n(7500, scale), p, seed)
    plan = plan_runs(tasks, 4, p, seed)
    r0 = execute(plan, None, p).runs
    B = float(np.quantile(r0["cost"], 0.9))
    r1 = execute(plan, CostController(B), p).runs
    c0, c1 = r0["cost"].to_numpy(), r1["cost"].to_numpy()
    s0, s1 = r0["success"].to_numpy(), r1["success"].to_numpy()
    months, per_month = _n(5000, scale, lo=500), 2000
    m0, m1 = np.empty(months), np.empty(months)
    sr0, sr1 = np.empty(months), np.empty(months)
    for a in range(0, months, 250):
        b = min(months, a + 250)
        idx = rng.integers(0, len(c0), size=(b - a, per_month))
        shock = np.where(rng.random(b - a) < 0.2, 1.4, 1.0)
        m0[a:b] = c0[idx].sum(1) * shock
        m1[a:b] = np.minimum(c1[idx] * shock[:, None], B).sum(1)
        sr0[a:b], sr1[a:b] = s0[idx].mean(1), s1[idx].mean(1)
    month = {
        "uncapped": {"mean": float(m0.mean()), "sd": float(m0.std()), "cv": float(m0.std() / m0.mean()),
                     "cvar95": cvar(m0), "success": float(sr0.mean())},
        "capped": {"mean": float(m1.mean()), "sd": float(m1.std()), "cv": float(m1.std() / m1.mean()),
                   "cvar95": cvar(m1), "success": float(sr1.mean())},
        "B": B,
    }
    wtp = {}
    for eta, v in [(1.0, 0.0), (1.0, 20.0), (0.0, 20.0), (0.0, 50.0)]:
        wtp[f"eta={eta:g}, v=${v:g}"] = willingness_to_pay(m0, m1, sr0.mean(), sr1.mean(), per_month, eta, v)
    out["monthly"] = month
    out["wtp"] = wtp
    return out


# ------------------------------------------------------------------------------- all

def run_all(scale: float = 1.0, params: DGPParams | None = None, verbose: bool = True) -> dict:
    steps = [("E0", e0_calibration), ("E1", e1_predictability), ("E2", e2_quotes), ("E2b", e2b_drift),
             ("E3", e3_censoring), ("E4", e4_controllers), ("E5", e5_tails_pooling)]
    res = {}
    for name, fn in steps:
        if verbose:
            print(f"running {name} ...", flush=True)
        res[name] = fn(scale=scale, params=params)
    return res
