# Pesh — plan: pricing AI agent runs before execution

Based on Tiwari (2026), *Pricing the Unpredictable: Pre-Execution Cost Prediction, Guaranteed
Ceilings, and the Economics of Cost-Bounded AI Agent Execution*. This document is the plan. The
code in `src/pesh` is the MVP that implements it, and `reports/results.md` is the evidence
that the MVP reproduces the paper's simulation study.

---

## 1. The problem, stated precisely

A buyer wants to know, before an agent runs, what the run will cost, and wants that cost
guaranteed not to exceed a ceiling. Token metering can't answer this, because agents loop and
decide at run time how long to run.

The paper's central reframing is that the literature optimises **budget in expectation**:

    max E[F(b(X),X)]   s.t.   E[C(b(X),X)] <= B̄                         (1)

Buyers, however, want **budget in probability** (a chance constraint) or a **pathwise** ceiling:

    Pr{ C(b(X_i),X_i) <= B_i | X_i } >= 1-α      or      C <= B_i        (2)

(2) needs conditional quantiles, calibrated uncertainty and control of the upper tail. That is
where agent cost behaves worst.

### Four levels of assurance (the product ladder)

| Level | Object | Contract analogue | Who bears cost risk | Pesh component |
|---|---|---|---|---|
| L0 | point estimate | cost-plus | buyer | (not sold; baseline only) |
| L1 | distributional quote q_α(X) with coverage ≥ 1-α | cost-plus + information | buyer | `quote/engine.py` |
| L2 | enforced pathwise ceiling + success SLA | guaranteed maximum price (GMP) | buyer below B, seller converts excess into quality risk | `control/`, `service/` |
| L3 | fixed price P per task | fixed price | seller | `quote/pricing.py` |

## 2. What the econometrics says (and what each finding forces in the design)

| Paper result | Consequence for the system | Where it lives |
|---|---|---|
| **Prop. 1** Cost is quadratic in steps because context accumulates: E[C\|T] = κ₂T² + κ₁T + κ₀ | Degradation that **compacts context** (resets the running sum) is worth more than stopping; cost must be modelled on **log** scale | `cost_model.py`, `control/controller.py` |
| **Prop. 2** Tail index halves: α_C = α_T/2 | Means and OLS on cost levels are unreliable; use quantiles of log cost; measure tails with Hill / Clauset | `econometrics/tails.py` |
| **Prop. 3** No predictor beats the ICC ρ_τ of repeated runs: R² ≤ ρ_τ | Measure ρ_τ on your own traffic first. It tells you how much better any model can ever get (information gap vs irreducible noise) | `econometrics/icc.py`, `pesh train` card |
| **Prop. 4** Split-conformal CQR gives finite-sample marginal coverage | Every quote is conformalised. The guarantee is **marginal only**, so report conditional coverage and warn on thin or out-of-range groups (adverse-selection risk) | `quote/conformal.py`, `quote/engine.py` |
| **§5.4** Model updates break exchangeability (static coverage 90% → 70%) | Online **ACI** per model version, a rolling coverage monitor and a drift alarm | `quote/aci.py`, `/monitor/coverage` |
| **Prop. 5** Caps censor the logs; naive QR underestimates slopes by ~40% | Retraining on capped logs **must** use Powell censored QR. Keep an **exploration slice** with a wider ceiling so upper quantiles stay identified; report the unidentified share | `econometrics/censored.py`, service `exploration_rate` |
| **§4.6 / BAGEN** Feasibility is learnable mid-run; amounts are not | The controller uses π̂(σ_t) = Pr(success within B \| state) and only a coarse remaining-cost estimate (rule 10) | `control/feasibility.py` |
| **Prop. 6** Pooling works only after truncation; common shocks never diversify | A fixed price needs a ceiling (policy limit), an idiosyncratic loading ∝ σ_B/√N, and a **systematic** loading, unless the ceiling is **index-linked** | `quote/pricing.py` |
| **Eq. 12** WTP = expected saving + risk-premium saving − quality cost | Target **risk-averse, high-volume, low-value-per-task** buyers first (support, extraction). Leave one-off high-value work on metering | `econometrics/risk.py` |
| **Bajari–Tadelis** Fixed price is efficient for simple, verifiable, numerous tasks | Go to market in order: L3 fixed price for verifiable repetitive tasks, L2 GMP for open-ended tasks | §6 below |

## 3. Architecture (paper Figure 1 as a closed loop)

```
 task features X ─► QuoteEngine ─► q_α(X), ceiling B = headroom·q, fixed price P
                      ▲   (CQR / Mondrian / ACI)            │
                      │                                      ▼
   Powell CQR  ◄── run logs (cost censored at B, outcome S) ◄── Runtime controller
   ICC, Hill        + exploration slice (B × 3)                   authorize → step → finish
                                                                  hard cap · feasibility stop · degrade
```

Service lifecycle: `POST /runs` issues the quote and ceiling. `POST /runs/{id}/authorize` runs
before every model call: it refuses any call whose worst case would breach B, which is what
makes the ceiling pathwise. `POST /runs/{id}/step` meters the call, and the controller returns
continue, degrade or stop. `POST /runs/{id}/finish` reports the outcome and updates ACI.
`GET /monitor/coverage` shows realised vs nominal coverage and raises a drift alarm.

## 4. The MVP (built)

| Module | What it does | Key method |
|---|---|---|
| `sim/dgp.py` | Calibrated agent DGP; vectorised step engine with **common random numbers** so policies are compared on identical trajectories | §5.1 |
| `sim/calibrate.py` | **SMM** calibration to published moments (upgrade over the paper's hand tuning) | Nelder–Mead on weighted relative errors |
| `econometrics/` | LP quantile regression (HiGHS), Powell / Chernozhukov–Hong censored QR, ICC (ANOVA/REML), Hill + Clauset, CVaR, pooling loadings, WTP | Koenker–Bassett, Powell 1986, Buchinsky 1994 |
| `quote/` | Split CQR, Mondrian, ACI, quote engine with quantile grid for E[min(C,B)], L3 pricing | Romano et al. 2019, Gibbs–Candès 2021 |
| `control/` | Budget-aware feasibility model (trained on-policy), policies P0–P4, offline replay | rule (10) |
| `data/` | Generic run/step schema, JSONL/CSV/Parquet, OpenHands adapter, leakage-free task splits | — |
| `service/` | FastAPI app, run store (SQLite), `BudgetSession` gateway, `PeshClient` | — |
| `eval/` | E0–E5 reproduction and markdown/PNG report | §5 |

### Reproduction status (simulated; see `reports/results.md`)

| Check | Paper | Pesh |
|---|---|---|
| ρ_τ / feature R² / oracle R² | 0.80 / 0.42 / 0.78 | 0.82 / 0.48 / 0.79 |
| CQR marginal / hardest-tercile coverage | 0.90 / 0.79 | 0.90 / 0.78 |
| Static coverage after model update → ACI | 0.70 → 0.90 | 0.78 → 0.90 |
| Naive vs Powell slope bias under 20% censoring | −40% / −6% | −36% / −1% |
| Full controller P4: cost / CVaR95 / success | −41% / −76% / −4.1pp | −42% / −80% / −2.4pp |
| Tail-index ratio α_C/α_T (mechanism) | 0.53–0.56 | 0.56–0.57 |
| Pooling slope capped / capped + shock | −0.52 / −0.12 | −0.51 / −0.13 |

## 5. Train → test → improve loop

1. **Ingest** real logs into the generic schema (`data/adapters`). Keep pre-execution features
   only (Prop. 3: nothing generated before the run knows the run's own randomness).
2. **Measure the ceiling first**: `pesh evaluate --logs real.parquet` reports ρ_τ (H1) and tail
   indices (H2). If ρ_τ is low, quote precision is capped, so lean on L2/L3 (controller +
   pooling) rather than a sharper quote.
3. **Train**: `pesh train` does a task-grouped split, fits linear QR (or GBM; or Powell when
   logs are capped), conformalises on one run per task, trains the feasibility model, and
   writes a versioned artifact with a model card.
4. **Test**: the card reports marginal coverage, coverage by strata (difficulty, label, loop),
   pinball loss and the gap to the ICC ceiling. The acceptance gates are marginal coverage in
   [1−α, 1−α+0.02], a monitored conditional gap, and P(C > B) = 0 under enforcement.
5. **Serve and monitor**: ACI keeps coverage at target under drift. The drift alarm fires when
   rolling coverage falls >3pp below target, and that triggers a retrain.
6. **Improve**, measured as closing (a) the information gap ρ_τ − R²_f with better features
   (repo map size, issue text embeddings, task history), GBM, or Mondrian groups on real
   difficulty proxies; and (b) the conditional-coverage gap on the hardest tercile. Retrain
   on logs that include the exploration slice, using Powell.

## 6. Commercial plan implied by the paper

- **Market sizing** (paper's Monte Carlo): a pure quoting/enforcement software layer is about
  USD 330M by 2029 (90% CI 126–801M), which is thin. The large version is **risk-bearing
  execution** (L3/GMP) on the governed spend (~USD 2.7B in 2026).
- **Sequencing**:
  1. *Now*: L1+L2 as infrastructure (quotes + enforced ceilings + coverage monitoring) for
     teams running agent fleets. Enforcement alone is commoditising (gateways), so the
     differentiator is **calibrated quotes + controller + censoring-aware learning**.
  2. *Next*: L2 GMP contracts for open-ended work (software engineering, research), with
     index-linked ceilings.
  3. *Then*: L3 fixed price per outcome for verifiable, repetitive tasks (tickets, document
     extraction), where Bajari–Tadelis says fixed price is efficient and pooling works.
- **Risks**: adverse selection on marginal coverage (price hard tasks separately, use
  Mondrian on real difficulty proxies); provider-side shocks (index-linked ceilings,
  continuous recalibration); success loss from aggressive control (tune the π̂ threshold
  against the buyer's v in Eq. 12).

## 7. Empirical research agenda (H1–H7) mapped to commands

| Hypothesis | Data | How to run with Pesh |
|---|---|---|
| H1 predictability ceiling | Bai et al. (2026) SWE-bench trajectories (4 runs/task) | adapter → `pesh evaluate --logs` (ICC) |
| H2 tail index | same + pooled public trajectories | `pesh evaluate` (Hill, Clauset) |
| H3 conditional under-coverage | stratified new runs + independent difficulty proxy | `pesh train --mondrian-by <proxy>`; card coverage by strata |
| H4 coverage under model change | runs before/after a version bump | tag `model_version`; `/monitor/coverage`, `e2b_drift` pattern |
| H5 censoring bias | operator's capped logs + exploration slice | engine auto-switches to Powell; compare against exploration rows |
| H6 controller dominance | live/replayed OpenHands A/B | `control.replay` on step logs; `compare_policies` in simulation |
| H7 WTP for a ceiling | conjoint / priced pilot | map estimates onto `willingness_to_pay(η, v)` |

**Minimum viable sequence:** H1 and H2 on already-released trajectories (analyst time only).
They decide whether the quote route or the GMP/fixed-price route carries the product.

## 8. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M0 (done) | MVP engine, simulator, service, tests, E0–E5 report | `pytest` and `pytest -m slow` green |
| M1 | Ingest Bai et al. trajectories; H1/H2 on real data | ρ_τ, α_C, α_T with CIs |
| M2 | Real-feature quote model (text embeddings, repo stats) + GBM | information gap closed ≥ 30% vs linear |
| M3 | Pilot: gateway in front of a partner's agent fleet, exploration slice on | P(C>B)=0, coverage within 2pp of target for 4 weeks |
| M4 | GMP contract pilot with index-linked ceilings | realised margin ≥ loading, no ruin events |
| M5 | Fixed-price offer on one verifiable workflow | loss ratio within plan; WTP validated (H7) |
