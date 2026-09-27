"""Calibrated data-generating process (paper Section 5.1) and a vectorised step engine.

The engine executes many agent runs step-synchronously so that execution policies
(hard cap, feasibility stop, degradation; paper Section 4.6 / E4) can be replayed on
exactly the same random draws (common random numbers): every per-step draw is taken
from an RNG seeded by (seed, step), so two policies see identical trajectories until
they intervene.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson

from ..config import DGPParams, PricingParams


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------------------------------------------------------- tasks

def simulate_tasks(n_tasks: int, params: DGPParams | None = None, seed: int = 0) -> pd.DataFrame:
    """Tasks with two standardised pre-execution features, latent difficulty, noisy human label."""
    p = params or DGPParams()
    rng = np.random.default_rng([seed, 1])
    x = rng.standard_normal((n_tasks, 2))
    u = rng.normal(0.0, p.sigma_u, n_tasks)
    d = x @ np.asarray(p.theta) + u
    noisy = d + rng.normal(0.0, p.label_noise, n_tasks)
    # population terciles (fixed cut-offs keep separately simulated batches exchangeable)
    sd = np.sqrt(np.sum(np.square(p.theta)) + p.sigma_u**2 + p.label_noise**2)
    cuts = norm.ppf([1 / 3, 2 / 3]) * sd
    label = np.digitize(noisy, cuts)  # 0: <15m, 1: 15m-1h, 2: >1h
    return pd.DataFrame({
        "task_id": np.arange(n_tasks),
        "x_repo": x[:, 0],
        "x_complex": x[:, 1],
        "human_label": label.astype(int),
        "difficulty": d,
    })


# ----------------------------------------------------------------------------- run plans

@dataclass
class RunPlan:
    """Everything about a batch of runs that is fixed before execution."""

    task_idx: np.ndarray          # row index into the tasks frame
    run_id: np.ndarray
    difficulty: np.ndarray
    c0: np.ndarray
    t_base: np.ndarray
    loop: np.ndarray              # bool: run enters an unproductive loop
    t_total: np.ndarray           # natural stopping step (<= t_max)
    truncated: np.ndarray         # bool: loop never exited before t_max
    p_success: np.ndarray
    u_success: np.ndarray
    u_degrade: np.ndarray
    tok_scale: np.ndarray         # multiplicative scale on per-step tokens
    seed: int

    @property
    def n(self) -> int:
        return len(self.task_idx)


def plan_runs(tasks: pd.DataFrame, runs_per_task: int, params: DGPParams | None = None, seed: int = 0,
              verbosity: float = 1.0, step_mult: float = 1.0) -> RunPlan:
    """Draw pre-execution randomness for `runs_per_task` runs of every task.

    `verbosity` and `step_mult` implement the provider-side model update of E2b.
    """
    p = params or DGPParams()
    rng = np.random.default_rng([seed, 2])
    idx = np.repeat(np.arange(len(tasks)), runs_per_task)
    run_id = np.tile(np.arange(runs_per_task), len(tasks))
    d = tasks["difficulty"].to_numpy()[idx]
    x_repo = tasks["x_repo"].to_numpy()[idx]
    n = len(idx)

    c0 = np.exp(p.c0_log_mean + p.c0_repo_slope * x_repo)
    m = np.exp(p.mu_T + p.lam * d) * step_mult
    G = np.exp(rng.normal(-0.5 * p.sigma_G**2, p.sigma_G, n))
    t_base = 1 + rng.poisson(m * G)
    loop = rng.random(n) < _expit(p.q0 + p.q1 * d)
    # Lomax(alpha, scale) via inverse CDF
    lomax = p.scale_L * ((1.0 - rng.random(n)) ** (-1.0 / p.alpha_L) - 1.0)
    L = np.where(loop, np.floor(lomax).astype(int) + 1, 0)
    t_uncapped = t_base + L
    t_total = np.minimum(t_uncapped, p.t_max)
    truncated = loop & (t_uncapped > p.t_max)
    p_succ = _expit(p.s0 + p.s1 * d + p.s2 * loop)
    tok_scale = np.exp(p.tok_difficulty_slope * d) * verbosity
    return RunPlan(idx, run_id, d, c0, t_base, loop, t_total, truncated, p_succ,
                   rng.random(n), rng.random(n), tok_scale, seed)


# ----------------------------------------------------------------------------- policies

@dataclass
class RunState:
    """Vectorised mid-execution state sigma_t = (t, c_t, z_t) for all runs."""

    t: np.ndarray
    spent: np.ndarray
    context: np.ndarray
    repeat_ewma: np.ndarray
    degraded: np.ndarray
    active: np.ndarray
    budget: np.ndarray
    last_step_cost: np.ndarray


class Policy(Protocol):
    """Execution policy interface used by `execute`.

    `budget(plan)` returns the per-run hard ceiling B (np.inf for none).
    `decide(state)` returns (stop, degrade) boolean arrays, evaluated after each step.
    """

    name: str

    def budget(self, plan: RunPlan) -> np.ndarray: ...

    def decide(self, state: RunState) -> tuple[np.ndarray, np.ndarray]: ...


class NoControl:
    name = "P0 no control"

    def budget(self, plan):
        return np.full(plan.n, np.inf)

    def decide(self, state):
        z = np.zeros_like(state.active)
        return z, z


# ----------------------------------------------------------------------------- engine

@dataclass
class ExecResult:
    runs: pd.DataFrame
    steps: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)


def execute(plan: RunPlan, policy: Policy | None = None, params: DGPParams | None = None,
            pricing: PricingParams | None = None, degrade_price_mult: float = 0.3,
            compact_to: float = 12_000.0, degrade_extra_fail: float = 0.12,
            record_steps: int = 0) -> ExecResult:
    """Execute all runs of `plan` under `policy`, step-synchronously.

    record_steps: number of runs (the first ones) whose step-level state is logged;
    used to train the feasibility model.
    """
    p = params or DGPParams()
    pr = pricing or PricingParams()
    policy = policy or NoControl()
    n = plan.n
    B = np.asarray(policy.budget(plan), float)

    t = np.zeros(n, int)
    spent = np.zeros(n)
    ctx_prior = np.zeros(n)            # sum of appended tokens since last compaction
    base_ctx = plan.c0.copy()           # context base (c0, or compacted size)
    new_tok = plan.c0.copy()            # tokens newly written at next call
    in_tok = np.zeros(n)
    out_tok = np.zeros(n)
    ewma = np.zeros(n)
    degraded = np.zeros(n, bool)
    active = np.ones(n, bool)
    capped = np.zeros(n, bool)
    stopped_feas = np.zeros(n, bool)
    last_cost = np.zeros(n)
    step_logs = []

    a_mu = p.a_log_mean - 0.5 * p.a_log_sd**2
    o_mu = p.o_log_mean - 0.5 * p.o_log_sd**2
    for step in range(1, p.t_max + 1):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        rng = np.random.default_rng([plan.seed, 1000 + step])
        a_all = np.exp(rng.normal(a_mu, p.a_log_sd, n))
        o_all = np.exp(rng.normal(o_mu, p.o_log_sd, n))
        r_all = rng.random(n)
        a = a_all[idx] * plan.tok_scale[idx]
        o = o_all[idx] * plan.tok_scale[idx]

        context = np.minimum(pr.window, base_ctx[idx] + ctx_prior[idx])
        new = np.minimum(new_tok[idx], context)
        cached = context - new
        mult = np.where(degraded[idx], degrade_price_mult, 1.0)
        cost = mult * (pr.w * pr.p_in * new + pr.delta * pr.p_in * cached + pr.p_out * o)

        # Pathwise ceiling: refuse the call that would breach B (block before paying).
        breach = spent[idx] + cost > B[idx]
        if breach.any():
            b_idx = idx[breach]
            capped[b_idx] = True
            active[b_idx] = False
        ok = ~breach
        idx, a, o, cost, context = idx[ok], a[ok], o[ok], cost[ok], context[ok]

        spent[idx] += cost
        last_cost[idx] = cost
        in_tok[idx] += context
        out_tok[idx] += o
        t[idx] = step
        ctx_prior[idx] += a
        new_tok[idx] = a

        looping = step > plan.t_base[idx]
        rate = np.where(looping, p.loop_repeat_rate, p.base_repeat_rate)
        # Poisson draw via inverse transform on the common uniform
        views = poisson.ppf(r_all[idx], rate)
        ewma[idx] = 0.7 * ewma[idx] + 0.3 * views

        done = step >= plan.t_total[idx]
        active[idx[done]] = False

        if record_steps:
            rec = idx < record_steps
            if rec.any():
                step_logs.append(pd.DataFrame({
                    "row": idx[rec], "step": step, "spent": spent[idx[rec]], "repeat_ewma": ewma[idx[rec]],
                    "context": context[rec], "last_step_cost": cost[rec], "looping": looping[rec],
                    "degraded": degraded[idx[rec]],
                }))

        # policy decisions for runs still going
        live = idx[~done]
        if live.size:
            state = RunState(t=t[live], spent=spent[live], context=context[~done], repeat_ewma=ewma[live],
                             degraded=degraded[live], active=active[live], budget=B[live],
                             last_step_cost=last_cost[live])
            stop, degrade = policy.decide(state)
            stop = np.asarray(stop, bool)
            degrade = np.asarray(degrade, bool) & ~degraded[live] & ~stop
            if stop.any():
                s_idx = live[stop]
                stopped_feas[s_idx] = True
                active[s_idx] = False
            if degrade.any():
                g_idx = live[degrade]
                degraded[g_idx] = True
                base_ctx[g_idx] = np.minimum(compact_to, base_ctx[g_idx] + ctx_prior[g_idx])
                ctx_prior[g_idx] = 0.0
                new_tok[g_idx] = base_ctx[g_idx]  # compacted summary is written fresh

    finished = ~capped & ~stopped_feas
    success = (finished & ~plan.truncated & (plan.u_success < plan.p_success)
               & ~(degraded & (plan.u_degrade < degrade_extra_fail)))
    runs = pd.DataFrame({
        "row": np.arange(n),
        "task_idx": plan.task_idx,
        "run_id": plan.run_id,
        "steps": t,
        "planned_steps": plan.t_total,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost": spent,
        "success": success,
        "capped": capped,
        "stopped_early": stopped_feas,
        "degraded": degraded,
        "loop": plan.loop,
        "cap": B,
    })
    steps = None
    if record_steps and step_logs:
        steps = pd.concat(step_logs, ignore_index=True)
        steps = steps.merge(runs[["row", "success", "steps", "cost"]]
                            .rename(columns={"steps": "final_steps", "cost": "final_cost"}),
                            on="row", how="left")
    return ExecResult(runs=runs, steps=steps)


def simulate(n_tasks: int, runs_per_task: int, params: DGPParams | None = None,
             pricing: PricingParams | None = None, seed: int = 0, verbosity: float = 1.0,
             step_mult: float = 1.0, policy: Policy | None = None, **exec_kw) -> pd.DataFrame:
    """Convenience: tasks + runs under a policy, as one run-log frame in the generic schema."""
    params = params or DGPParams()
    tasks = simulate_tasks(n_tasks, params, seed)
    plan = plan_runs(tasks, runs_per_task, params, seed, verbosity, step_mult)
    res = execute(plan, policy, params, pricing, **exec_kw)
    return attach_tasks(res.runs, tasks)


def attach_tasks(runs: pd.DataFrame, tasks: pd.DataFrame) -> pd.DataFrame:
    df = runs.join(tasks.reset_index(drop=True), on="task_idx")
    df["model_version"] = "sim"
    df["total_tokens"] = df["input_tokens"] + df["output_tokens"]
    return df.drop(columns=["task_idx"])
