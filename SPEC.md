# Volatility-Targeted QQQ: Project Specification

Version 1.0. Frozen 2026-08-06.

The parameters in Section 2 were fixed before the first backtest. Changing any of them
requires an explicit entry in the decision log (Section 13).

---

## 1. Research question

Does the choice of volatility forecasting model materially change the performance
of a monthly-rebalanced, volatility-targeted position in QQQ, relative to
buy-and-hold and relative to naive sizing rules?

Two sub-questions carry equal weight.

1. Are the seven models statistically distinguishable as forecasters of 21-day
   realized variance?
2. If they are distinguishable statistically, does the ranking survive into
   economic outcomes after financing costs and the leverage cap?

The expected finding is that vol targeting improves risk-adjusted return over
buy-and-hold through volatility clustering and the negative volatility-return
relationship, while the seven models land within a standard error of each other.
A null result is a publishable result here and must be reported as such.

---

## 2. Frozen configuration

| Parameter | Value |
|---|---|
| Underlying | QQQ, total return (dividends reinvested) |
| Evaluation period | 2000-01-03 to the most recent complete month |
| Rebalance frequency | Monthly, at the close of the last trading day |
| Target volatility | 20% annualized |
| Leverage cap | 2.0x |
| Leverage floor | 0.0x (no shorting) |
| Starting capital | $100,000 |
| Borrow rate | Effective fed funds (FRED `DFF`) + 50bp |
| Cash credit rate | Effective fed funds (FRED `DFF`) |
| Transaction cost | 2bp round trip on traded notional |
| Forecast horizon | h = 21 trading days |
| Realized-variance proxy | Daily Yang-Zhang (5-minute intraday deferred, Section 3) |

Weekly rebalancing was considered and rejected on spread and slippage grounds.

Alternative configurations run **only after** the main study is complete, and
reported as a separate additional-results section with every cell populated.

- Target ∈ {10, 15, 20, 25}%
- Cap ∈ {1.5, 2.0, 3.0}
- Spread ∈ {0, 50, 150}bp

---

## 3. Data

### 3.1 Series and sources

Every series is daily and comes from a public source. Yahoo Finance returns split- and
dividend-adjusted prices, so no manual corporate-action adjustment is applied.

| Series | Purpose | Source |
|---|---|---|
| QQQ daily OHLC, adjusted, 1999-03 onward | Traded returns and daily realized measures | Yahoo Finance (`yfinance`) |
| NDX daily close, 1985 onward | GARCH-family estimation burn-in | Yahoo Finance |
| IXIC (Nasdaq Composite) daily OHLC, 1986 onward | Realized-measure burn-in, rescaled to the NDX level | Yahoo Finance |
| Effective fed funds (`DFF`), daily | Financing leg | FRED |
| NYSE trading calendar, including half-days | Session validation | `pandas_market_calendars` |

A Stooq second-vendor cross-check was attempted, but its endpoint now sits behind a bot
challenge that was not bypassed. The Yahoo series were instead validated internally
against the NYSE session count, the 2000-03-20 split, and OHLC integrity.

### 3.2 Data checks

- **Split reconciliation.** QQQ split 2-for-1 on 2000-03-20, five weeks into the
  evaluation window. Yahoo's adjusted series already accounts for it, and the
  acquisition check confirms the adjusted overnight return that day is small.
- **Seed-window open prices.** NDX and IXIC OHLC before the mid-1990s frequently report
  the session open as mechanically equal to the prior close, which invalidates any
  range estimator. A diagnostic runs before a range-based measure touches the seed
  sample; where it fails, close-to-close volatility is used for the seed window only.

### 3.3 Splice

- GARCH-family estimation history: NDX daily returns from 1985 to 1999-03-09, then QQQ
  total return from 1999-03-10.
- Realized-measure history: IXIC (clean opens from 1986) rescaled by a single scalar to
  the NDX volatility level over the seed window, then QQQ from 1999-03-10. Section 13
  records the scalar and how it was derived.
- Traded series: QQQ total return only, from 1999-03-10.

### 3.4 Intraday data

The original design called for 5-minute realized variance (Section 4). The candidate
intraday source stored its monthly files unsorted by ticker, so isolating QQQ would have
required transferring the full multi-decade archive, which was not warranted for a
monthly-rebalanced strategy. Intraday sampling was deferred, and the study runs on daily
data throughout, using the pre-specified daily fallback for realized variance. Two-scale
realized variance and realized kernels, which need intraday data, are out of scope for
the daily build. Section 13 records the deferral.

---

## 4. Realized measures

The forecast target is realized variance over the next 21 trading days. On daily data the
realized-variance proxy is the Yang-Zhang estimator over a rolling 21-day window, which
uses the full OHLC bar and is robust to overnight gaps and drift.

The daily log-realized-variance series that feeds the HAR-RV lags and the rough-volatility
(RFSV) model is built from the squared overnight log return plus the Rogers-Satchell
intraday range estimator. Yang-Zhang is undefined for a single day, since its
drift-independent term averages across days, so the 21-day target and the daily proxy are
two distinct constructions and are reported as such.

The 5-minute intraday construction, `RV_t = RV_t(5min, RTH) + r_overnight^2`, was
specified but deferred with the intraday source (Section 3.4). Two-scale realized variance
(Zhang, Mykland and Aït-Sahalia 2005) and realized kernels (Barndorff-Nielsen, Hansen,
Lunde and Shephard 2008) require intraday data and are not part of the daily build.
Section 14 collects the citation basis for the estimators used.

---

## 5. Volatility models

The seven models below are fixed. Adding or removing one requires a decision-log entry
(Section 13).

| ID | Model | Estimation | h-step aggregation |
|---|---|---|---|
| `garch` | GARCH(1,1) | QMLE, variance targeting | Closed form |
| `egarch` | EGARCH(1,1) | QMLE | Monte Carlo, 10,000 paths |
| `gjr` | GJR-GARCH(1,1,1) | QMLE, variance targeting | Closed form |
| `ewma` | RiskMetrics EWMA, λ = 0.94 | Fixed λ | Identity (flat term structure) |
| `rv` | Trailing realized variance | None | Direct |
| `har` | HAR-RV (Corsi 2009) | OLS on log RV | Direct-h regression |
| `rfsv` | Rough fractional stochastic volatility | Variogram estimate of H | Prediction formula + lognormal correction |

The innovation distribution is a skewed-t primary with a Gaussian QML robustness row.
QQQ daily excess kurtosis is far above zero, so Gaussian ML is consistent but
inefficient.

Use variance targeting for the GARCH-family intercept so the unconditional level
is pinned to the sample second moment.

For `rfsv`, report the estimated Hurst parameter with a standard error, and state
in the write-up that Ĥ from noisy daily proxies is biased downward (see the
measurement-error critique in Section 14). Treat rough volatility as one
competitor among seven; do not claim a roughness parameter for the latent process.

---

## 6. Forecast aggregation to h = 21

The sizing object is the expected average variance over the holding period:

```
V_t(h) = E_t[ (1/h) · Σ_{k=1..h} σ²_{t+k} ]
```

For GARCH(1,1) with φ = α + β:

```
σ²_{t+k|t} = σ̄² + φ^(k-1) · (σ²_{t+1|t} − σ̄²)
V_t(h)     = σ̄² + λ_h · (σ²_{t+1|t} − σ̄²)
λ_h        = (1 − φ^h) / (h · (1 − φ))
```

λ_21 is the fraction of the one-day signal surviving to the sizing decision.
At φ = 0.99, λ_21 = 0.91. At φ = 0.97, λ_21 = 0.75. At φ = 0.94, λ_21 = 0.55.
Small differences in fitted persistence turn into large differences in the leverage
path, so a one-step forecast will not do.

GJR aggregates in closed form under symmetric innovations with φ = α + γ/2 + β, using
E[z²·1{z<0}] = 0.5. EWMA is IGARCH with φ = 1, so λ_h = 1 at every horizon and its term
structure is flat; that separates it from the mean-reverting models, and the separation
only shows up once the forecast is aggregated to 21 days. EGARCH has no closed form for
E[σ²] because its recursion is in log variance, so it is simulated.

HAR-RV and RFSV use direct-h regressions on `log RV_{t+1:t+21}` rather than
iterating a daily recursion 21 times forward, which avoids compounding
specification error. The direct-versus-iterated multi-step split adopted here
(closed-form/MC iteration for the GARCH family, direct-h regression for HAR and
RFSV) follows the published treatment in Ghysels et al. (2019).

Report a Mincer-Zarnowitz regression per model:

```
RV_{t+1:t+21} = a + b · V_t(21) + ε
```

The coefficients map directly onto systematic over- or under-leveraging, because
position size scales as 1/√V and a model unbiased in variance is still biased in
leverage by Jensen's inequality.

---

## 7. Estimation protocol

- Expanding window, minimum 1000 observations.
- Seeded on NDX daily returns from 1985, switching to QQQ from 1999-03-10.
- Refit every model at every rebalance date, using data through date t only.
- 320 refits per model across the sample (one per rebalance date, the 1999-12-31
  warmup through 2026-07-31; 319 forecasts are scored from the 2000-01-03 evaluation
  start). Count verified against the trading calendar, where 320 distinct month-end
  rebalance dates span 1999-12 through the complete month 2026-07.
- Persist the full parameter path. The evolution of α + β across the sample is a
  figure in its own right.
- `arch` (Sheppard) covers GARCH, EGARCH, and GJR with fixed-window refitting.

No look-ahead. Every quantity indexed at date t is computable from information available
at the close of t, and the test suite enforces it.

---

## 8. Portfolio construction

Sizing at each rebalance date t:

```
w_t = clip( 0.20 / sqrt(252 · V_t(21)), 0.0, 2.0 )
```

Daily portfolio return:

```
r_p,t = w · r_QQQ,t + (1 − w) · c_t

c_t = DFF_t              when w ≤ 1   (cash credit on the uninvested fraction)
c_t = DFF_t + 0.0050     when w > 1   (borrow cost on the levered fraction)
```

The cash leg carries real return. Short rates ran between 1% and 6.5% from 2000 to 2007,
so crediting the uninvested fraction, and charging the levered fraction, is what keeps the
comparison to buy-and-hold honest through the periods when the strategy holds cash.

**Weight drift.** Leverage floats within the month. Turnover at rebalance is
measured against the drifted weight, not the previous target:

```
w_t^drift = w_{t−1} · (1 + r_QQQ, month) / (1 + r_p, month)
turnover_t = | w_t^target − w_t^drift |
cost_t     = 0.0002 · turnover_t
```

No intramonth rebalancing. Run a Reg-T maintenance check at 25% equity and report
the count of breach months.

---

## 9. Benchmarks

Buy-and-hold on its own does not show where a gain comes from, so the strategies are
measured against four benchmarks.

1. **Buy-and-hold QQQ.** Full equity exposure, no cash leg.
2. **Ex-post constant leverage.** Fixed multiple chosen so full-sample realized
   volatility equals 20%. Infeasible in real time; serves as the static upper
   bound.
3. **Trailing 21-day RV sizing.** Uses RV_t(21) directly with no model. Every one
   of the seven models must beat this rung to justify its existence.
4. **Unconditional volatility sizing.** Expanding-mean variance. Isolates how much
   of the gain comes from time variation versus level.

---

## 10. Evaluation

### 10.1 Layer 1, forecast accuracy

- Target: realized variance over the next 21 days.
- Loss functions: QLIKE and MSE. Both are robust to a noisy volatility proxy
  (Patton 2011). Do **not** use MAE or R² on volatility; they are not robust.
- Primary sample: non-overlapping monthly forecasts, 319 scored from 2000, matching
  the actual decision points.
- Secondary sample: overlapping daily forecasts with Newey-West at lag ≥ 25.
- Tests: pairwise Diebold-Mariano with HAC errors, then Hansen's Model Confidence
  Set at 90% to identify the set indistinguishable from the best.
- Mincer-Zarnowitz per model, per Section 6.

### 10.2 Layer 2, economic outcome

Primary metric is target adherence, not Sharpe:

- Full-sample realized annualized volatility versus 20%.
- Mean absolute deviation of trailing 21-day realized portfolio volatility from
  the target.
- Volatility of volatility: the standard deviation of the (non-overlapping 21-day)
  realized portfolio volatility. Harvey et al. (2018) report this falling from 4.6%
  to 1.8% for scaled US equities, but they target 10% vol while this study targets
  20%, so the raw std is **not** directly comparable. Normalize as a coefficient of
  variation (std ÷ target), which gives Harvey 0.46 → 0.18 (a 2.6× reduction), this study
  0.73 → 0.325 (a 2.25× reduction). The CV pair is the published comparison; the raw
  std (this study's own numbers) is reported alongside.

The study then reports the following:

- Sharpe in excess of the cash rate, with the Ledoit-Wolf (2008) robust test for
  pairwise differences. The standard Jobson-Korkie test assumes iid normality,
  which fails badly on levered equity.
- Deflated Sharpe (Bailey and López de Prado 2014) with an honest trial count.
- Maximum drawdown, skew, kurtosis, Calmar, Sortino.
- Turnover and cost drag.
- Leverage distribution, including fraction of months pinned at the 2x cap.
  Expect heavy pinning in 2004-2006 and 2017.
- Decomposition of the return gap versus buy-and-hold into market exposure and
  financing cost.

The subperiods are 2000-2002, 2003-2007, 2008-2009, 2010-2019, 2020, 2022, 2023-present.

### 10.3 Cross-layer

Report the rank correlation between the Layer 1 and Layer 2 orderings. Weak
correlation is a common and defensible headline.

---

## 11. Deliverables

### 11.1 Write-up

The paper follows the standard sections of introduction, data, methodology, results, robustness, and
conclusion, and every figure and table regenerates from a script under a fixed seed.

### 11.2 Static figures

- Cumulative growth of $100,000 by strategy, log scale.
- Rolling 21-day realized portfolio volatility against the 20% target line, per
  strategy.
- Leverage path per strategy with the 2x cap marked.
- Drawdown curves.
- α + β parameter path across the sample.
- Layer 1 loss table with MCS membership marked.

### 11.3 Animations

1. **Forecast term-structure surface.** Each model's forecast term structure V_t(h)
   over horizons up to 63 days, animated across rebalance dates, which makes the λ_h
   shrinkage from Section 6 visible frame by frame.
2. **Strategy performance through time.**
3. **Per-strategy running account balance.**
4. **Leverage path with the 2x cap.**

---

## 12. Repository layout

```
vol-target-qqq/
  README.md                  # overview, performance, robustness
  SPEC.md                    # this document
  DECISIONS.md               # frozen parameters and corrections
  LICENSE
  config/config.yaml         # frozen parameters
  reproduce.sh               # end-to-end pipeline
  src/volteq/
    data/                    # loaders, splice, calendar
    rv/                      # realized measures (Yang-Zhang, Rogers-Satchell)
    models/                  # garch, egarch, gjr, ewma, rv, har, rfsv
    forecast/                # h-step aggregation, Mincer-Zarnowitz
    backtest/                # sizing, financing, costs, engine
    eval/                    # layer 1, layer 2, DM, MCS, deflated Sharpe
    viz/                     # figure style
  scripts/                   # the pipeline stages
  tests/                     # includes the no-look-ahead invariant tests
  outputs/tables/            # result tables
  assets/                    # README performance chart and summary
  gallery/                   # animations (GIF + MP4)
```

---

## 13. Decision log

| Decision | Rationale |
|---|---|
| Weekly rebalancing rejected | Spread and slippage cost |
| CEV dropped from the model list | Conditional volatility as a deterministic function of price level fits awkwardly on an index that compounds over 26 years |
| HAR-RV added | Standard benchmark against which rough volatility is normally measured |
| Tenor 512 dropped, grid ends at 252 | No calendar anchor |
| NDX seeding adopted | 210 trading days of QQQ history before 2000 is far short of stable GARCH estimation |
| Config frozen | Pre-specification, per Section 2 |
| Intraday source `mito0o852/OHLCV-1m` deferred, not rejected | Phase 2A confirmed QQQ/QQQQ 1-minute coverage back to 1999-04, but Phase 2B found the monthly files unsorted by ticker (row-group pruning impossible), so extracting one ticker needs the full ~84 GB repo. Section 3.4 fallback (daily Yang-Zhang) is now active. FirstRate Data (1-min, adjusted, from 2000-01) is the upgrade path. No frozen parameter changed. |
| Daily and seed vendors set to Yahoo (`yfinance`), cross-checked against Stooq | Free, dividend/split-adjusted daily QQQ from 1999-03-10 and ^NDX from 1985; Stooq provides an independent close cross-check. Replaces the prior `TBD`. |
| NDX seed window uses close-to-close volatility, not range estimators | Per the Section 3.2 diagnostic: NDX open == prior close exactly for 80.5% of 1980s and 87.5% of 1990s sessions (89.6% pre-1995), the mechanical-open pathology. Yang-Zhang and Rogers-Satchell are therefore invalid across the entire 1985-10..1999-03 seed window; close-to-close volatility is used there. Range-based daily proxies begin on the QQQ traded series (opens valid; 0.03% degenerate) from 1999-03-10. |
| Stooq independent cross-check deferred | `pandas_datareader` 0.11.1 has no Stooq route, and the stooq.com CSV endpoint is now behind a JavaScript proof-of-work anti-bot challenge, which was not bypassed. Yahoo daily data was validated internally instead (session count vs NYSE calendar, split reconciliation, OHLC integrity). A second-vendor cross-check remains an open item before final results. |
| RV-family estimation seed = ^IXIC (1986-01-01+), rescaled by a single scalar c to the NDX close-to-close level | ^NDX has valid OHLC opens for none of 1985-1999; ^IXIC (Nasdaq Composite) has clean opens (< 5% mechanical) from 1986 and correlates 0.951 in returns with ^NDX. GARCH family stays on NDX cc (unchanged); RV family uses Yang-Zhang and Rogers-Satchell on ^IXIC. c = var(NDX cc returns) / mean(IXIC daily RV) over the seed window 1986-01-01..1999-03-09 = **2.7812**. The scalar decomposes into an index difference (NDX cc 22.67% vs IXIC cc 16.25% annualized vol; 1.95× variance, the concentration effect of 100 vs several thousand names) and an estimator effect (IXIC cc 16.25% vs IXIC RV 13.60%; 1.43× variance, the range proxy understates close-to-close variance). Rescaling by c gives both families one unconditional volatility level (variance targeting pins σ̄² to the seed second moment), so an index-level difference is not baked into a comparison meant to isolate a model difference. Effective estimation history at 2000-01-03: GARCH family 3,602 obs, RV family 3,540 obs (was 208 without the seed; both now clear the §7 1,000 minimum). |
| October 1987 retained in the seed (no exclusion) | Excluding Oct-1987 from the seed window moves NDX cc annualized vol −5.6%, IXIC RV vol −3.4%, the scalar c −4.7%, and the GARCH(1,1) implied σ̄ −4.7%, all below the 10% flag threshold, because a single 22-session month is 0.7% of a 3,331-session seed. No exclude-1987 robustness row is triggered. Nothing was excluded on the agent's initiative. |
| Daily series currently rests on a single vendor (Yahoo) | The Stooq second-vendor cross-check remains unobtained (endpoint bot-gated; not bypassed). Yahoo QQQ/NDX/IXIC and FRED DFF are validated internally (calendar, split reconciliation, OHLC integrity) but not against an independent vendor. Open item to resolve before final results. |
| QQQ-era estimator bias measured, none found; methodology flag withdrawn | Test statistic d = mean(rv_daily) / var(QQQ close-to-close returns) = **1.025** over 2000-01-03..present (inside the [0.95, 1.05] no-correction band; by-year mean 1.038, no drift), versus d = 0.700 on the ^IXIC seed. The 1.43× estimator effect is an ^IXIC composite-index OHLC artifact (sampled highs/lows understate continuous extremes), absent in QQQ trade-print OHLC. No point-in-time correction needed; the earlier methodology flag is withdrawn. The scalar c remains the sole seed-level correction. |
| Variance targeting implemented Engle-Mezrich (σ̄² = sample second moment, ω profiled out) | `arch` has no native variance-targeting option. For garch and gjr, σ̄² is pinned to the sample second moment of the estimation window and ω = σ̄²·(1−φ) is profiled out; only the dynamics (α, β[, γ]) and skew-t shape (η, λ) are estimated by QMLE. The profiling pins the unconditional level to the sample second moment exactly as SPEC §5 requires and matches the §6 aggregation, which uses σ̄² and φ separately. egarch is fit freely (no VT, per config); ewma is fixed λ. Verified: fitted σ̄² equals the window sample variance; closed-form V_t(21) matches simulation within MC error. |
| RFSV forecast-error variance for the retransformation set to s_Δ² = min(½·ν²·Δ^{2H}, Var(log RV)) | The RFSV variance forecast is (1/h)Σ exp(Ê[X_{t+Δ}] + ½ s_Δ²). The conditional log-variance forecast-error variance is taken as half the variogram value ν²Δ^{2H} (ν² from the variogram intercept), capped at the unconditional log-RV variance. The construction yields a large retransformation (mean 60%, up to 80%) because the single-day overnight²+RS proxy is a noisy estimate of daily integrated variance (ν²≈1.55); it is the main modeling lever for RFSV and is stated in `models_memo.md`. Ĥ (0.036→0.055) is reported with its downward bias from the noisy proxy noted (SPEC §5). RFSV is one competitor among seven; no claim is made about the latent process's roughness. **[Superseded by F1 (below, same date): the ½ is the H→0 limit of the exact fBm constant c(H), which replaces it; under c(H) the mean retransformation is ~69% (0.694), not the 60% quoted here, and `models_memo.md` carries the current value. The deleted orphan `direct_h_summary.json` held the pre-F1 0.599.]** |
| Hurst parameter H treated as UNIDENTIFIED on daily proxies; RFSV run across a fixed H grid | Superseding an earlier plan to fix H=0.10: Fukasawa et al. (2022) show regression-based Hurst estimates land near 0.10 regardless of the true parameter, and measurement-error-aware methods (Cont-Das critique) report even stronger roughness; estimates of 0.05-0.20 appear across thousands of assets. So 0.10 is not a ground-truth anchor and the estimated 0.036-0.055 sits at the low edge of a range the field cannot pin down. RFSV is therefore run at fixed H ∈ {0.02, 0.05, 0.10, 0.15} alongside the estimated-H version, and the question posed is whether the strategy outcome is sensitive to H at all, insensitivity would itself resolve the identification problem. No single H is claimed. FINDING: across the grid the strategy spread is realized-vol 0.007, Sharpe 0.017, CAGR 0.002, economically insensitive to H. **[Pre-F1 spreads. U2 (2026-08-10) recomputed them under c(H) to realized-vol 0.009, Sharpe 0.013, CAGR 0.006; the H-insensitivity finding is unchanged and the paper carries the corrected figures.]** |
| Backtest financing accrues actual/360 over calendar days; warmup rebalance at 1999-12-31 | The cash/borrow leg accrues at DFF (or DFF+50bp when levered) over calendar days on the actual/360 money-market convention (fed-funds standard); SPEC did not fix a day-count. A warmup rebalance at 1999-12-31 (last trading day before eval_start) sizes the position held on 2000-01-03, so the equity curve starts exactly at the frozen eval_start with $100k rather than one month in. No frozen parameter changed. |

| D2: strategy outcome insensitive to the RFSV Hurst parameter across the plausible rough range | Across a within-plausible-range grid H ∈ {0.02, 0.05, 0.10, 0.15} plus estimated H (≈0.04), the cross-variant realized-volatility spread is negligible (max−min ≈ 0.007 absolute). Mechanism: Δ^{2H}→1 as H→0 (the flat region of the parameterization, where the estimate and the literature's 0.05-0.20 all sit), compounded by the square root in w = 0.20/√(252 V), the V-stage spread (~0.48) halves at √ and is not compressed thereafter. The 2× leverage cap is **passive**: it binds on 0% of dates for every variant, so it is not the mechanism. Reported separately: an extended grid adding the near-Brownian anchors H = 0.35 and 0.50 shows a larger ~0.28 relative spread, but those carry MZ R² of 0.459 and **0.011**, i.e. they document RFSV kernel degeneration (an implausible, useless forecast for equity vol), not genuine strategy sensitivity. **[The ≈0.007 realized-vol spread here is pre-F1; U2 recomputed it to 0.009 under c(H), conclusion unchanged.]** |
| D3: the ewma/rv Sharpe advantage is Moreira-Muir volatility timing, not leverage | The excess-return Sharpe decomposition E[excess] = w̄·E[x_Q] + Cov(w,x_Q) − s·E[max(w−1,0)] (reconstructs realized mean excess to 2e-19) attributes ewma's and rv's edge to their volatility-timing covariance term Cov(w,x_Q), quantified at **0.021 and 0.024 annualized** (vs 0.011-0.018 for the calibrated models). Excess Sharpe is invariant to a constant leverage multiple up to a strictly-negative borrow-spread drag (confirmed: synthetic constant leverage at w=1.5 and 2.0 sits below buy-and-hold), so their above-1 average leverage (1.11-1.12) cannot explain the advantage. Corrects the earlier "higher leverage in a rising market" wording in the memos. |
| Backtest turnover cost moved into the rebalance-day return | The engine previously recorded each day's return before the rebalance turnover cost (the cost hit only the equity path), so the cached daily returns were gross of cost while equity was net. The cost is now applied before recording the return; cumulating the daily returns reproduces the equity path to 1e-15 for every strategy. Effect on excess Sharpe is ≤ 0.0027 (rankings unchanged); `backtest_metrics.csv` regenerated. |

| F1: RFSV retransformation constant 0.5 → fBm conditional-variance constant c(H) | s_Δ² = c(H)·ν²·Δ^{2H} with c(H) = Γ(3/2−H)/(Γ(H+½)·Γ(2−2H)) (Gripenberg-Norros 1996 / Nuzman-Poor 2000; RFSV application GJR 2018), replacing the ad-hoc 0.5, which was the H→0 limit c(0⁺)=½. **Justification is theory-consistency only** (c(H) is the exact fBm conditional-variance constant; c(½)=1, monotone increasing on (0,½]); the calibration improvement is a consequence. A stable fractional-Gaussian-noise numerical converges to the closed form (dt=1 sits ~2% above per discrete-vs-continuous sampling; Richardson to dt→0 within ~0.003, exact at the Brownian anchor). Effect on rfsv (est H): mean V̂/RV 0.81→0.86, MZ slope 1.43→1.35, realized vol 0.208→0.202, Sharpe unchanged; grid tightens (D2 conclusion reinforced). **The residual ~12-14% variance-level bias is a reported RFSV result**, non-normality of the log-RV forecast residual (skew +0.23, excess kurtosis +0.91), which no variance constant fixes; rfsv stays theory-driven. Reconciliation: realized vol is governed by E[RV/V̂]=1.075 (→0.207≈observed 0.202), not 1/(mean V̂/mean RV)=1.16, because sizing w=0.20/√V̂ down-weights high-V̂ periods. |

| G0: c(H) validation closed analytically (no numerical solver) | The three-Gamma c(H) is validated in closed form by three exact anchors, c(0⁺)=½, c(¼)=Γ(5/4)/(Γ(3/4)Γ(3/2)), c(½)=1, plus monotonicity on (0,½]. The discrete fBm-conditioning numerical is retained only as documentation of its (slow) convergence rate, not as a check on c(H). No Toeplitz/Levinson solver built. |

| G1: egarch included in the Layer-1 secondary (overlapping-daily) sample | Daily egarch forecasts are generated by freezing each month's parameters and running one 10,000-path 21-step Monte Carlo per date (state updated daily, no refitting; path count matches the primary; seed from config). Wall time 12.0 min, the earlier "computationally intractable" characterization was wrong and no omission note is recorded. No look-ahead (the conditioning state at each date is invariant to future returns). With egarch, the daily MCS[QLIKE] = {garch, egarch, gjr, har, rfsv} matches the primary monthly set exactly; egarch's inclusion changes no other model's retained-set membership under either loss. |

| H2: vol-of-vol comparison to Harvey (2018) normalized by target vol | The §10.2 note comparing raw vol-of-vol (this study 14.6%→6.5%, Harvey 4.6%→1.8%) was wrong: Harvey target 10% vol, this study 20%, and vol-of-vol scales with target. Corrected to a coefficient of variation (÷target): Harvey 0.46→0.18 (2.6× reduction), this study 0.73→0.325 (2.25× reduction). CV pair is the published comparison; raw std stays this study's own number. §10.2 and `layer2_memo.md` updated. No backtest number changes. |
| H3: Sharpe inference, Ledoit-Wolf pairwise + Deflated Sharpe | LW (2008) delta-method Sharpe-difference test, Newey-West HAC lag 10 = ⌊4(n/100)^(2/9)⌋, n=6688 daily excess returns. Result: model-vs-benchmark 45/45 significant at 5%, model-vs-model 11/105 (mostly discarded rfsv fixed-H internals; none among the seven headline models; median p=0.226). DSR (Bailey-LdP 2014) with N=19 trials (14 live configs + 5 discarded rfsv variants, per rule 3; passive benchmarks excluded from N): all strategies survive deflation (targeting DSR 0.994-0.998, passive 0.953-0.956). `src/volteq/eval/sharpe.py`, `sharpe_inference{,_lw_pmatrix}.csv`. Inference only; no Phase A-G number changes. |
| H4: cross-layer rank correlation (descriptive, n=8) | Spearman(QLIKE order, adherence order) = +0.833; Spearman(QLIKE order, Sharpe order) = −0.690 over the 8 primary. The Sharpe inversion is the D3 volatility-timing covariance (ewma 0.021, rv 0.024) on a higher-vol book, not skill, and (per H3) is inside the model-to-model noise band. n=8 → no significance claimed. `cross_layer_ranks.csv`. |
| H1: monthly-vs-daily QLIKE↔MSE rank-correlation divergence (0.881 vs 0.619) explained | MSE top-1% observation share 0.44-0.62 vs QLIKE 0.26-0.44 (MSE dominated by extreme realizations; QLIKE scale-free). Crisis-window exclusion reorders MSE 4/8 on both samples (har near-best full-sample MSE → mid-pack excluded). Overlapping daily target repeats each crisis ~21×, amplifying MSE's crisis weighting in the daily sample and pulling it off the QLIKE ranking. Daily MSE labelled a crisis-window-fit measure; QLIKE is the cross-regime ranking loss. `loss_concentration.csv`. Diagnostic only. |
| H5: headline is narrower and sharper than SPEC §1, not a confirmation | Forecast (Layer 1) and adherence (Layer 2) layers show a two-tier structure, the four conditional-variance models {garch, egarch, gjr, har} separate from the naive {ewma, rv, trailing_rv21} with MCS + DM significance (dynamics clear the §9 trailing-RV rung at p=0.001-0.011; rfsv p=0.13 does not), not the flat "all seven indistinguishable" null SPEC §1 anticipated. The flat null holds only on Sharpe (models mutually indistinguishable, jointly beating passive). Recorded as such in `phase_h_memo.md`. **REVISED M0 → N1 → O0/O1 (see below); current (v4): the Layer-1 half stands; on adherence the one robust separation is resizing-vs-static, the finer dynamics-vs-naive separation being block-sensitive, four-model, and excluding rfsv.** |

| I1: har qualified, robust member of the separating four, not robustly second | Crisis-window-excluded robustness read on existing forecasts (no model change). har QLIKE rank of 8 slips full→excluded 2→4 (primary) and 1→3 (secondary): garch and rfsv edge ahead, and on the crisis-excluded daily sample gjr/har/rfsv are a near-tie (QLIKE 0.2106/0.2107/0.2096). But har's *membership* is robust, it stays in the block-5 QLIKE MCS {garch, egarch, gjr, har, rfsv} and still DM-clears the §9 trailing-RV rung (p=0.015 primary, 0.005 secondary) with the crises removed. Per the Phase I rule (qualify if rank *or* DM moves materially), the rank moves, so H5 is revised: the four are a set, not a ranking; set membership is significant, within-set order is not. gjr_skewt's primary-layer first place is unaffected (QLIKE #1 full and crisis-excluded; adherence #1). har adherence over the non-crisis decade 2010-2019 = 0.0636 MAD (calm-decade undershoot shared by all strategies). `scripts/phase_i.py`, `development/phase_h_memo.md` (I1 section + H5 revision). |
| I2: implied-vol surface out of scope; model-forecast term structure is the in-scope reading | The original figure list's implied-volatility surface animation is dropped, it needs an options-chain vendor (strikes × expiries) not in the frozen SPEC §3 data plan. The in-scope substitute is the model-forecast V_t(h) term structure over h=1…63 (the SPEC §6 λ_h mechanic on the study's own forecasts: ewma flat at 1.0, estimated models shrinking toward the unconditional level). No figures produced in Phase I; `development/figure_inventory.md` catalogues the full backlog and its data-status (mostly cheap re-plots of existing CSVs; the term structure needs a moderate egarch-MC + har-refit rerun since only h=21 is stored). |

| J: paper outline + two framing corrections checked against sources | `development/paper_outline.md` (structure, gaps, style spec; no prose/figures/code). Two asserted framings corrected: (1) Hansen-Lunde (2005) is a *related* question, not a restatement, it fits the within-set Layer-1 indistinguishability (garch(1,1) not beaten; garch vs gjr DM p=0.34) but not the separating result or Layer 2, and this study matches their FX null rather than their equity finding that asymmetry beats GARCH(1,1). (2) vs arXiv 2212.07288 (Bernardi-Bianchi-Bianco, "Smoothing volatility targeting"): monthly-rebalancing is **rejected** as a difference, that paper also rebalances monthly; breadth is **rejected/reframed**, they benchmark six methods (RV, RV6, RV-AR(1), HAR, SV, GARCH(1,1)) vs this study's seven, so the real difference is composition (this study adds asymmetric GARCH + rough vol, has no SV; they center on SV smoothing), not count; only the **financing** difference stands (this study charges fed-funds+50bp borrow on leverage>1; they charge transaction costs on turnover only). Structural decisions: qualifications inline with a thin cross-cutting robustness section; frozen-parameter discipline gets its own subsection. Orphan flagged: `sharpe_decomposition.csv` (D3) superseded by `return_gap_decomposition.csv` (G4). |

| K: figure/animation specification (no figures built) | `development/figure_style.md` + `src/volteq/viz/style.py` (shared standard: 200 dpi / 8.5 in for ~850 CSS px; DejaVu Sans; Okabe-Ito hues for the 7 models + achromatic black/grays for the 4 benchmarks, each paired with a non-hue line-style/marker channel, deuteranopia/protanopia-safe; display names + SPEC-§5/§9 canonical order; titles in the caption not the canvas). `development/figure_inventory.md` K2 audit: all 7 existing figures rebuilt (not patched) to standard; coverage gaps named with the claim each serves; weak/redundant candidates (financing accumulation, weight drift, DM heatmap) marked gallery-only; decorative animations and expanding-window MCS rejected as claimless. `development/animation_spec.md` K3-K5: no animation earns inline placement on evolution-is-the-claim; the term-structure is better as a static small-multiples panel (rerun justified for the panel, not the animation); animations are gallery-only; inline animation budget 5 MB, GIF ≤ 2 MB with MP4 fallback; one-render ffmpeg GIF+MP4 path; build order separates draft-independent statics from placement/caption/rerun work that waits for prose. Nothing decided about building the term-structure animation. |

| L0: animated visualizations, deliverable preserved in the gallery, none inline | The original project description named animated visualizations as a paper deliverable. Phase K evaluated them on the evolution-is-the-claim test and found none earning inline placement: the strongest case (the model-forecast term structure) is served better by a static small-multiples panel, and the rest (wealth race, rolling Sharpe, realized-tenor surface) are carried by their static forms. The deliverable is preserved as a repository gallery (`gallery/`, GIF + MP4) with its own README, specified in `animation_spec.md`. A reader arriving from the original framing finds this reasoning here rather than inferring an omission. |
| L0: signature plot removed (needs intraday vendor); drawdown figure to gallery-only | The K2 buildable list is corrected. A volatility signature plot (SPEC §3.4: RV across intraday sampling frequencies) cannot be built from disk, `rv_panel.parquet` is daily-only and the intraday source is deferred at ~84 GB, so it is removed from the buildable list and recorded as a vendor-dependent gap beside the implied-vol surface. A separate drawdown figure carries only drawdown *duration* beyond what the log-equity curve and the max-DD scalar (`layer2_riskreturn.csv`, −0.49…−0.58 vs −0.83) already show; the study computes no time-underwater statistic and makes no duration claim, so drawdowns move to gallery-only. `development/figure_inventory.md` updated. |

| L4 quality pass: drawdown figure built (reverses the L0 gallery-only call); three formatting overlaps fixed | The L0 decision to keep drawdowns gallery-only was reversed on review: an underwater curve is the most direct view of the drawdown-compression result (max DD −0.49…−0.58 vs buy-hold −0.83) and is cheap from `backtest_equity.parquet`, so `drawdown.png` is now an inline figure (18 figures total). A full visual QA of all figures also fixed three overlaps in `scripts/build_figures.py`: `rolling_vol` legend moved below the axes (was over the 2000-02 spikes), `cross_layer_scatter` Spearman notes moved to data-free corners (were over the top point), and `benchmark_comparison` rebuilt to source the Trailing-RV rung's adherence from `cross_layer_ranks.csv` (was missing + a note overlapped data). `qlike_mse_rank` re-ranked within the 8 primary so it reproduces the H1 Spearman +0.881. No cached result changed. |

| M1: SPEC §7 refit count corrected 310 → 320, direction checked against the schedule | The forecast panel has 320 rows; §7 said "roughly 310". Rather than infer the SPEC figure was loose from the panel alone, the count was checked against the trading calendar: the rebalance dates are 320 distinct month-ends spanning 1999-12 (warmup) through the complete month 2026-07, so monthly rebalancing from the frozen eval_start reproduces 320 exactly. First rebalance 1999-12-31, last 2026-07-31; 319 forecasts scored from eval_start 2000-01-03. §7 and the drafted methods prose (`methods-02-models.md`) both set to 320. The SPEC figure was the loose one, confirmed not assumed. |

| M0: H5 headline split by layer once the trailing-RV rung entered the adherence panel | The rung's non-overlapping-21-day adherence MAD is 0.0533 (recomputed from a fresh trailing_rv21 backtest through the same code path that produced `layer2_adherence.csv`; gjr cross-checks at 0.0477 in both). The MAD places the naive rung inside the resizer pack (GARCH family + har 0.0477-0.0500, ewma/rv/rfsv/rung 0.0533-0.0536) and far from the static benchmarks (0.090-0.097). So the Phase H5 claim that conditional dynamics separate from naive sizing on *both* layers holds only on Layer 1 (QLIKE MCS excludes the naive trio): on Layer-2 adherence the separation is **resizing vs not resizing**, and the naive trailing-RV rung adheres about as well as the dynamic models. Headline rewritten in `phase_h_memo.md` (H5); propagated to `development/paper_outline.md` (header box + §2.1) and `development/layer2_memo.md`. Wording neither softened nor strengthened; the number drove the split. **[Superseded by O0/O1: the "naive rung adheres about as well as the dynamic models" clause is wrong, the dynamic models beat it; but the clean dynamics-vs-naive separation N1 then claimed was circular, so the corrected reading is close to this M0 one.]** |

| M2-M5: Layer 1 results section drafted; layer1_memo rung claim corrected | `paper/results-01-layer1.md` drafted per the outline (loss tables, DM matrix, MCS with block-length sensitivity, crisis-excluded rerun), with the gjr/rfsv/har qualifications inline (gjr first + indistinguishable from garch at DM p=0.34; rfsv in the MCS jointly but fails the rung at p=0.13; har a robust member whose rank moves 2→4 under crisis exclusion). QLIKE primary per Patton (2011), with the guarantee's limits stated. **Discrepancy corrected:** `layer1_memo.md` said "every model beats the §9 rung"; the `dm_matrix` shows only the GARCH family + har clear it (rfsv p=0.13, ewma p=0.99 do not, rv is the rung), so the memo is fixed to match the table. M3 style report (`M3_style_report.md`): 6→0 consecutive-opening pairs, no banned terms, and one reported spec conflict (the setup-then-reversal cap collides with the required inline qualifications; accuracy kept). M3 audit (`M3_self_audit.md`) maps every number to its source. M4: `qlike_mse_rank.png` is now two panels reproducing the H1 within-8 Spearman (0.881 primary, 0.619 secondary); time underwater added to `layer2_riskreturn.csv` (total + longest-spell). M5: λ_21 alone is adequate for the aggregation section; the across-h term-structure panel is an enhancement, not a gap; rerun not run. No cached A-L number changed. |

| N0: adherence numbers reconciled, two subsets of one table, not a conflict | The "0.048-0.050" and "0.0533-0.0536" ranges are the dynamic-resizer (GARCH family + har) and naive/rough-resizer subsets of the same non-overlap-21d-MAD table (SPEC §10.2). Recomputing all 18 strategies through one code path reproduces `layer2_adherence.csv` to <5e-4; nothing stale, neither range wrong. Canonical going forward: `adherence_inference.csv` (18 entries + intervals); `layer2_adherence.csv` remains the frozen point-estimate table and agrees. `development/adherence_reconciliation.md`. |
| N1: adherence inference added AFTER the metric was frozen, recorded as such | The adherence metric was frozen in SPEC §10.2 before results were seen; the bootstrap inference here is added after and does not change the metric. Stationary bootstrap on the daily portfolio return series per strategy (B=2000, seed 20260806; second seed 19990310 shifts CI bounds ≤7e-4), recomputing the full statistic per replicate through the point-estimate path (not resampling the rolling-vol series). Block grid {42,105,210} days = MCS {2,5,10} months, all exceed the 21-day window (none dropped); CI widths grow ~30% across the grid, conclusions stable; coverage 90% (matched to MCS). The interval covers sampling variation in the realized-vol path with forecasts held fixed, not model-estimation uncertainty. `outputs/tables/adherence_inference.csv`, `adherence_pairwise.csv`. |
| N1 supersedes M0's Layer-2 conclusion: adherence carries BOTH separations | M0 read adherence as resizing-vs-static only (the ~0.5pp dynamic-vs-naive gap looked small on point estimates). The paired bootstrap resolves that gap: resizers beat static in 100% of replicates, and the dynamic models (GARCH family + har) beat the naive resizers (ewma/rv/rfsv/trailing) in 94-100% (15/16 pairs at the 90% level). So dynamics separate from naive on **both** layers, in the same direction, an order of magnitude smaller on adherence than the resizing-vs-static gap. Marginal intervals overlap across resizers, so no single adherence leader is named (gjr leads on point estimates and in 92-96% of paired replicates over the other dynamics, decisive only over egarch). H5 headline re-revised in `phase_h_memo.md`; propagated to `paper_outline.md` and `layer2_memo.md`. Neither softened nor strengthened; the inference drove it. **[Superseded by O0/O1: this N1 partition placed rfsv with the naive resizers on its adherence value, circular. Corrected on model class, the dynamics-vs-naive separation drops to 11/15, is block-sensitive, and excludes rfsv.]** |

| N2: "every model beats the rung" traced, one bad claim (already fixed), all else correct | Searched SPEC, outline, all memos, the decision log, and drafted sections. The only incorrect standing claim was `layer1_memo.md` (fixed in M3). All other occurrences are the SPEC §9 normative requirement (correct as the bar) or correct statements that only the GARCH family + har clear the rung, or claims about models vs buy-and-hold (not the rung). `development/N2_rung_trace.md`. Separately, the Layer 1 results section now states ewma's p=0.99 against the rung as its own sentence: at that level ewma does not improve on a 21-day trailing average under QLIKE in any measurable sense, a sharper claim than MCS exclusion. |
| N3: setup-then-reversal style cap amended to exempt result qualifications | The M3-flagged conflict (the once-per-section cap collides with the requirement that qualifications travel in the sentence stating the result) is resolved by exempting result-qualification sentences from the cap, which still applies to rhetorical reversals. Written into `paper_outline.md` J4. Reasoning: the cap targets an AI-prose tell (rhetorical reversal); result qualifications are the opposite, an honesty requirement; conflating them would force dropping mandated caveats. N5 confirms the amendment resolved the conflict on the Layer 2 section (4 exempt result-qualifications, 0 rhetorical). |
| N4/N5: Layer 2 results section drafted | `paper/results-02-layer2.md`: adherence (primary) with the N1 bootstrap intervals and the two-separation reading (no leader named); risk-return with every Sharpe beside its realized vol; LW matrix + deflated Sharpe (N=19 justified in text); the ewma/rv Sharpe carrying the Moreira-Muir vol-timing covariance 0.021/0.024 in the same passage; return-gap decomposition; turnover and the 4-6 bp/yr cost drag (return-gap component, not the growth-inflated `cost_drag_annual`); leverage/cap-pinning and zero Reg-T breaches as portfolio mechanics; subperiods. Vol-of-vol vs Harvey uses the normalized CV. Longest-spell underwater (14.9y buy-hold vs 6.8-10.0y targeting) is the discriminating statistic; total-days ~93% stays in the table. Style pass 8→0 opening pairs. `N5_style_report.md`, `N5_self_audit.md`. |
| N6: drawdown figure moved gallery-only → inline candidate | The longest-spell statistic (M4) now exists, so `drawdown.png` is enhanced with a bottom gantt marking each strategy's longest spell below a prior peak. The figure carries a claim the log-equity curve and the max-DD scalar do not: drawdown *duration and recovery speed* (buy-hold 14.9y underwater vs targeting 6.8-7.6y). Reversal of the L0 gallery-only verdict; reason recorded. `scripts/build_figures.py`, `development/figure_inventory.md`. |

| O0/O1: N1's adherence separation was circular for rfsv; corrected and weakened | N1 grouped rfsv with the naive resizers on adherence, but rfsv has dynamics and is in the Layer-1 QLIKE MCS, so that placement read membership off the adherence value (circular). Rebuilt on model class (rfsv dynamic): the paired comparison drops from a circular 15/16 to 11/15, rfsv does not beat the naive resizers (46-56%), its 12-14% variance-level retransformation bias sinking its adherence [mechanism reverted by Q0]. Multiplicity-aware (O1 preferred route): Hansen's MCS on the per-block adherence loss |vol−0.20|, same bootstrap, block grid {2,5,10} blocks = N1's {42,105,210} days, retains {garch,egarch,gjr,har} at blocks 2/5 and all eight at block 10. So the one robust adherence separation is resizing-vs-static; the dynamics-vs-naive separation is block-sensitive, four-model, and excludes rfsv. The Layer-2 adherence set differs from the Layer-1 credible set by rfsv. rfsv resolution: dynamic on both layers (category consistency), poor adherence explained by the level bias, the option that weakens rather than strengthens the result. H5 re-revised to v4; `scripts/phase_o_grouping.py`, `development/phase_o_grouping.md`, `outputs/tables/adherence_mcs.csv`. |

| P0: the adherence-MCS block sensitivity is effect size, not a power vacuum | Diagnostic: at the 210-day block the MCS (run on all 11 incl static) still EXCLUDES the static benchmarks and only the naive resizers return. Effective resampled units fall 159→63→31 across the {42,105,210}-day grid, and resizer CIs widen ~34% (gjr 0.0089→0.0119), enough to blur a half-point dynamic-vs-naive gap but not the four-point resizing-vs-static gap. So the finer separation is a small effect the 42/105-day blocks resolve and the 210-day block cannot, not an uninformative set. "Block-sensitive, dissolves at the longest" replaced with this in the Layer 2 section, robustness section, layer2 memo, H5 (v4), and outline. Not presented as strengthening: the finer effect is still four-model, excludes rfsv, rests on 11/15 pairs. `scripts/phase_p_power.py`, `development/P0_power_check.md`. |
| P1: rfsv level-bias mechanism verified and promoted to a result | Two checks. QLIKE penalty of a 14% multiplicative under-forecast (b≈0.86) = 1/b+ln b−1 ≈ 0.009-0.012, small (so rfsv stays in the QLIKE MCS) and ~half of rfsv's 0.017-0.024 within-set QLIKE gap (residual non-normality, skew 0.23 / xs-kurt 0.91, adds the rest). Adherence: E[RV/V̂]=1.067 → realized vol 0.20·√1.067 = 0.207, a systematic 0.0066 above target, matching rfsv's excess adherence MAD of 0.0055 over the unbiased dynamic models. Both check out in direction and magnitude, so the mechanism (a proportional loss weights a multiplicative level bias weakly while the sizing rule uses the level directly, so the same forecast passes Layer 1 and fails Layer 2) is promoted into the cross-layer section beside the vol-timing covariance, with the caveat that the QLIKE penalty is ~half the gap. `development/P1_rfsv_mechanism.md`. **[SUPERSEDED by Q0: the two checks used different bias objects (agg variance 0.86 vs per-obs E[RV/V̂] 1.067) and √(E[X]) where E[√X] was needed; the like-for-like signed deviation contradicts the mechanism (rfsv −0.0103 vs har −0.0110, har adheres better). Reverted to a conjecture.]** |
| P2: discussion section drafted on three mechanisms, disagreements foregrounded | `paper/results-05-discussion.md`: dynamics-vs-naive on forecast loss (garch inside the set; Hansen-Lunde relation per J); vol-timing covariance (ewma/rv high Sharpe, ewma p=0.99); rfsv level bias (opposite direction, per P1). Covers what the study does NOT establish: single-instrument scope (QQQ 2000-2026), the four-model rfsv-excluding block-limited adherence separation, the H insensitivity as convenience not resolution, and the post-hoc adherence inference on a frozen metric. Written to not read as though every result pointed the same way. Style pass 8→0 pairs; `P3_style_and_audit.md`. |

| Q0: rfsv level-bias mechanism reverted from result to conjecture | The two P1 checks ran on different bias objects: the QLIKE penalty on the aggregate variance ratio mean(V̂)/mean(RV)=0.869, the adherence side on the per-obs E[RV/V̂]=1.067, and the adherence prediction used √(E[RV/V̂]) where E[√(RV/V̂)]=0.959 was needed (putting realized vol on the wrong side of target). At the volatility level the sizing rule uses, the aggregate bias is 1.021, essentially none. The like-for-like signed deviation contradicts the mechanism: rfsv under-shoots target by 0.0103, LESS than har's 0.0110, yet har adheres better (MAD 0.0492 vs 0.0539), so rfsv's adherence disadvantage is dispersion, not a systematic level offset. **[R1/S0 canonical: rfsv −0.0102/MAD 0.0536, har −0.0109/MAD 0.0493 on the layer2_eval blocks that reproduce `layer2_adherence.csv`; the pct_change path quoted here gives −0.0103/0.0539, −0.0110/0.0492. Conclusion unchanged; S0 decomposes the signed deviation into a small systematic level term minus a dispersion-driven concavity term, the MAD disadvantage remaining dispersion.]** Mechanism reverted to a conjecture in the cross-layer section and discussion; numerical-agreement-as-evidence removed. The cross-layer FACT (rfsv credible on L1, out of the L2 adherence MCS) and the abstract asymmetry (proportional loss weak on level, sizing uses level directly) survive; the magnitude does not. `development/Q0_mechanism_reconciliation.md`. Do not adjust a bias figure to force agreement, none was. |
| Q1-Q3: conclusion, introduction, abstract drafted | `paper/{conclusion,introduction,abstract}.md`. Conclusion restates the two-layer result, the three threads (dynamics-vs-naive forecast loss; vol-timing covariance verified; rfsv divergence with the level-bias explanation NOT established per Q0) and the scope limits, introducing nothing. Introduction motivates from Harvey (2018) and positions against 2212.07288 using the J2 corrected differences (financing + model composition only; NOT rebalancing or breadth), states the two-layer design and why the layers are kept separate, previews the adherence separation with all its qualifications (four-model, block-limited, rfsv-excluded), and names the N1 bootstrap as the one post-hoc procedure. Abstract 243 words, every number matched to a section, every qualification carried. Q4 style pass 19→0 pairs; `paper/Q4_style_and_audit.md`. |
| R0-R2: variance-level relabeling, dispersion finding, undershoot diagnostic | No frozen change; no model recompute. **R0** (`development/R0_bias_relabeling.md`): the ~12-14% RFSV residual bias labeled **variance-level** at every live location (methods-04, results-02/03/05, outline, SPEC F1/O0, memos); methods-04 gains the volatility-level counterpart (aggregate forecast/realized vol ratio **1.021**, essentially no bias in the quantity the weight consumes); the one surviving sizing-consequence implication in a drafted section (results-02 "because its retransformation level bias sinks its adherence") removed, as were the causal survivals in the H5-canonical `phase_h_memo.md` and the outline. **R1** (`development/R1_R2_undershoot_dispersion.md`, `outputs/tables/adherence_signed.csv`): the dispersion finding written as FACT into the cross-layer section, rfsv undershoots by **0.0102** (canonical layer2_eval blocks; the summary's 0.0103 was the one-day-offset pct_change path), LESS than har's **0.0109**, yet adheres worse (MAD 0.0536 vs 0.0493) because its realized vol is more dispersed (vov 0.0718 vs 0.0644); signed+absolute per strategy for all 18 in the CSV; "why more dispersed" kept a marked conjecture. **R2** (gates R4; into results-02 as portfolio mechanics): the block-mean undershoot is NOT universal (4 of 18 over-shoot: ewma/rv/trailing/buy-hold); it is the concavity of √ on the non-overlapping-block statistic (gap ≈ vov²/2·mean, corr 0.986 with dispersion, power-mean good to 0.005; const_lev −0.032 block-mean with 0.198 on-target full-sample vol isolates it), NOT rebalance drift and NOT the cap (corr +0.44 wrong sign, binds ≤4%, never for the undershooting dynamics); the forecast level sets the side of target. Described, not corrected (parameters frozen). `scripts/phase_r_signed.py`. |
| R3-R6: adherence-intervals figure, placement, consistency, style | **R3** (`scripts/build_figures.py::fig_adherence_intervals`, `figures/adherence_intervals.png`): per-strategy adherence MAD with 90% bootstrap CI and adherence-MCS membership, the Layer-2 twin of `layer1_qlike_mcs.png` (solid retained / hollow-hatch excluded), 42- and 105-day panels, 210-day in caption; colorblind-safe by fill-pattern channel + Okabe-Ito, canonical order. Figure count 18→**19**. **R4** (`development/R4_figure_placement.md`): every figure placed with section, supporting sentence, and caption; 18 of 19 anchored inline in the prose, `rolling_vol.png` the sole gallery figure; figure-shaped gaps (V_t(h) term structure per M5; Jensen-gap and signed-deviation scatters) reported, not built. **R5** (`development/R5_consistency.md`): end-to-end pass; 4 inconsistencies fixed (intro "adheres poorly"→"no better than a trailing average"; conclusion "level-bias"→"variance-level bias"; a wrong "316 refits" caption; a stale "6.8-7.6y" inventory figure); all other recurring numbers traced to source and shown equal or differing only by disclosed scope (primary vs +Gaussian) or rounding (vol-timing 0.021≈0.0215). **R6** (`development/R6_style_and_captions.md`): document-wide sentence length mean 26.3 sd 13.7 (broad, human-varied), zero banned constructions, zero em dashes; two consecutive-"The" runs broken; all 18 captions ≤45 words. STOP after R6, no term-structure rerun, no animations, no README, no repo restructure. |
| S0: signed target deviation decomposed into level and concavity terms | The R2 concavity account (signed ≈ −vov²/2·mean) predicts the wrong rfsv/har ordering, and using each strategy's own mean does not fix it. Exact identity: signed_dev = level_term − concavity_term, level = fullvol−0.20, concavity = fullvol−blockmean. rfsv undershoots less than har (−0.0102 vs −0.0109) despite the larger concavity gap (0.0122 vs 0.0097) because its **positive level term** (+0.0021, from its nearly unbiased 1.021 volatility forecast placing fullvol above target) offsets it, while har's four-percent-high forecast (1.040) puts har below (−0.0012). Residual from a pure-concavity account = level term: corr(residual, level)=+0.999, corr(residual, forecast ratio)=−0.909, and the residual (mean |0.0107|) is 5.6× the power-mean approximation error (0.0019), so systematic not noise. Four overshooters (ewma, rv, trailing, buy_hold); buy_hold overshoots because fixed exposure at no target gives it a +0.068 level term dwarfing its concavity gap, while const_lev/uncond_vol are scaled to target (level≈0) and undershoot via concavity. Layer 2 mechanics passage rewritten to carry both terms; cross-layer signed comparison completed to note the net; R1 dispersion finding unchanged (MAD is dispersion-dominated). `scripts/phase_s_decompose.py`, `outputs/tables/adherence_decomposition.csv`, `development/S0_deviation_decomposition.md`. No frozen change, no model recompute. |
| S1: canonical adherence value sweep of memos/log/SPEC/inventory | R5 covered the paper; this covered the rest. Canonical figures rfsv MAD 0.0536 / har 0.0493 / signed −0.0102 / −0.0109 on the `layer2_eval` non-overlapping blocks (reproduce `layer2_adherence.csv`). Corrected one current figure (`phase_o_grouping.md` rfsv 0.0539→0.0536); marked two un-superseded dated records (`Q0_mechanism_reconciliation.md` header note, this log's Q0 entry inline) with the canonical values and "conclusion unchanged"; left superseded records (P1 file + entry, M0 entry) and the N0 reconciliation table (shows both paths by design) intact. Canonical construction named once in `adherence_reconciliation.md` with every sourcing table listed (adherence_inference/mcs/signed/decomposition + fig_adherence_intervals). `development/S1_canonical_sweep.md`. |
| S2: prose length spot check, two paragraphs | Document mean 26.3 words is mostly content. methods-05 cash-leg paragraph: length is content (rate range, mechanism, timing), left unchanged. discussion Hansen-Lunde paragraph: the 68-word sentence carried content but "without being the same question" was a signpost the following colon made redundant, cut and the run-on split (68 → 25 + 37 words), no content lost. No document-wide reduction run. `development/S2_prose_spotcheck.md`. |
| S3: term-structure rerun V_t(h), h=1..63, all models | New panel `data/processed/forecast_vh.parquet` (282,240 rows, 14 cols, 320 dates), NOT overwriting `forecast_v21.parquet`. Each model by its declared method: closed form from `param_paths` for garch/gjr (no refit) and identity for ewma; egarch Monte-Carlo path (refit + horizon-63 sim per date, ~11 min, the cost); direct-h OLS per horizon for har; RFSV prediction-formula running mean for rfsv+grid; trailing-h mean for rv. h=21 reconciliation vs forecast_v21: **every deterministic column exact to ≤9e-19**; egarch to ≤1.3e-5 (MC + non-bit-reproducible MLE; the same-horizon refit validation is ≤1.8e-5), invisible to the surface animation. `scripts/fit_term_structure.py`; `src/volteq/models/direct_h.py` refactored (shared `_rfsv_pointwise`, new `rfsv_term_structure`/`har_term_structure`, output of `rfsv_fit_forecast` bit-identical); `aggregate.py` adds `v_egarch_mc_path`. M5 stands (paper needs no across-h panel); the rerun runs for the S4 volatility-surface animation only. |
| S4-S5: five gallery animations, built and verified | `scripts/build_animations.py`, `gallery/`. Five GIFs + poster PNGs through the shared style module: (1) forward term-structure surface V̂ₜ(h) from `forecast_vh`, (2) wealth race, (3) $ account balance, (4) rolling 2-year Sharpe, (5) leverage path with the 2× cap. 12 fps, frames semiannual through calm years and monthly through 2007-2009; canonical order/names/Okabe-Ito + per-model marker as the non-hue channel; every frame from cached data. **MP4 not produced, this environment has no ffmpeg** (the render branches on `writers.is_available("ffmpeg")`; install ffmpeg and re-run to add them). Total GIF+poster 15.0 MB; gallery only, paper embeds none (K4). **S5 verification:** every GIF has one unique frame per date, zero dropped/duplicated (the 2008 window carries all 12 months, leverage 1.0→0.75→0.4 across it); content matches data (cap-pinning 3.8/4.1%, surface shrinkage, $80-90k vs $34k crisis balance). One fix: the surface vol axis clipped the 2008 GARCH curves, widened to 0.05-0.85. The **rolling-Sharpe animation adds nothing over a static rolling-Sharpe line** (K3 agreed) and is flagged as the drop candidate; kept as one of the five requested. `development/S4_animations.md`, `development/S5_animation_verification.md`. STOP after S5, no repository README, no repo restructure, no animation in the paper. |
| T0-T7: repository build, reproducibility, and release preparation | No frozen change; no result recomputed except stale-table regeneration. **T0** (`development/T0_venue_report.md`): venue inputs reported, decision left to the user, arXiv needs a LaTeX conversion (~4-8 h) plus first-time-submitter endorsement (tightened Jan 2026) and AI disclosure; SSRN takes the PDF directly, no endorsement (cheapest timestamp); no preprint/public-repo conflict. **T1**: restructured for a cold reader, `paper/`, `gallery/`, `figures/` (consumed) split from `development/` (memos/logs/diagnostics); `DECISIONS.md` promoted as the reader-facing decision log (frozen parameters + every documented reversal); ~80 cross-references and 5 script path-constants updated and verified. **T2** (`development/DATA_AND_REPRODUCIBILITY.md`, `reproduce.sh`): deps pinned (Python 3.13.13, +pillow), single-entry pipeline, ~45-60 min. Clean-clone test: **17 of 22 tables bit-exact**; the test caught **five real gaps, all fixed**, `adherence_mcs` was written by no script (phase_o now writes it), `layer2_riskreturn`'s two underwater columns were out-of-band (added to layer2_eval), and three diagnostic tables were stale pre-F1 (rfsv corrected ratio 0.813→0.862, realized vol 0.208→0.202, now paper-consistent; regenerated). EGARCH stays reproducible to ≤1.3e-5; fresh Yahoo downloads may drift (vendor revisions). **T4**: rolling-Sharpe animation removed (no paper cite; gallery now four). **T5**: `brew install ffmpeg` (8.1.2) succeeded; the four animations re-encoded with MP4 companions (0.53-1.74 MB). **T6** (`LICENSE`): code MIT, writing CC BY 4.0; no vendor data committed (`data/` git-ignored, tables are derived); no un-attributed third-party material. **T7** (`development/T7_first_visitor.md`): first-visitor read, 10 dev audit files moved out of `paper/` (now 14 content sections), 7 phase-letter tags + 1 memo ref + 1 "Phase E" stripped from the paper, no overstatement; `SPEC §` cross-refs reported as venue-dependent. 46 tests pass. STOP after T7, nothing submitted, repository not made public. |
| U0-U4: disclosure, licensing, pre-commit hygiene, commit preparation | No frozen change; one stale paper number corrected. **U0** (`paper/disclosure.md`, `development/U0_disclosure_log_check.md`): AI-disclosure section added as back matter; the log verified to carry each specification choice with reasoning and every named reversal, R0 was the one the reader-facing `DECISIONS.md` lacked and was added, so the disclosure's final sentence is backed not softened. **U1** (`LICENSE`): copyright line set to `2026 [AUTHOR NAME]` (the author's name is the one open item); MIT (code) / CC BY 4.0 (writing) stated in `LICENSE` and the README. **U2** (`development/U2_pre_f1_sweep.md`): swept every generated artifact for pre-F1 residue. All 19 figures pixel-match a current-data rebuild; the three stale tables were T2's (fixed); one stale orphan JSON (`direct_h_summary.json`, rfsv retransform 0.599 vs current 0.694, no producer) deleted; and **a number in the paper was pre-F1**, the D2 Hurst-grid spreads (realized-vol/Sharpe/CAGR), computed in Phase D and never recomputed, corrected 0.007/0.017/0.002 → **0.009/0.013/0.006** in methods-04 and results-04 (the "barely moves" conclusion unchanged; max change the CAGR spread, ×3). MZ slope 1.35, variance ratio 0.86, realized vol 0.202, 12-14% bias all confirmed current. **U3** (`paper.pdf`, `development/U3_read_package.md`): the 14 sections + disclosure assembled in reading order, **10,258 words / 23 pages**; TinyTeX-2023 lacked packages and the version mismatch blocked tlmgr, so the PDF was produced via pandoc → HTML → headless Chrome (a clean reading copy; the arXiv LaTeX conversion is still the T0 estimate); a read-priority list flags the 14 load-bearing passages (late-revised claims, marked conjectures, overstatement-preventing qualifications). **U4** (`development/U4_commit_plan.md`): repo `git init`-ed on `main`, **206 files / 20 MB staged, nothing committed, no remote**; `data/`, `.venv/`, `.claude/`, caches, and `.pyc` excluded; figures/tables/gallery/`paper.pdf` deliberately tracked; no vendor data, credentials, absolute sandbox paths, or env config staged; commit message and repo description drafted. STOP after U4, not committed, not pushed, not made public, nothing submitted to SSRN. |

| V0-V4: byline, license, carried verification items, open-item log | No frozen change; no result recomputed. **V0**: author **Cristian Gualy** applied to `LICENSE` (`Copyright (c) 2026 Cristian Gualy`), the paper (a title + `Cristian Gualy · 2026` byline added to `paper/abstract.md`, which had no title block), the README license line, and the U4 repo description, reading identically for an SSRN match; `paper.pdf` regenerated with the visible byline and author in the title metadata. **V1** (`methods-04`, `results-04`): the H-grid magnitude adverb "barely moves", chosen when the CAGR spread was 0.2 pp and tripled to 0.6 pp under F1, softened to "moves little" with one honest clause ("the CAGR gap is not literally nothing"); the corrected spreads 0.009/0.013/0.006 confirmed as the only occurrence and the H-insensitivity conclusion unchanged. **V2**: traced the deleted orphan `direct_h_summary.json` (pre-F1 rfsv retransform 0.599), nothing in the paper sourced it (the paper cites no retransformation percentage); the current 0.694 is the mean lognormal-retransformation uplift, live in `rfsv_retransform_audit.csv` and `rfsv_grid_summary.json`; 0.599 survives only as labeled pre-F1 provenance; the `[Superseded by F1]` bracket added to the §13 pre-F1 "60%" entry above. **V3** (`session_log.md`, `DECISIONS.md` open-items section): open items logged cold-start-actionable, trial LaTeX build (to diagnose typesetting-vs-prose; check the c(H) and QLIKE equations against source), the author's full read (U3 priority list), then the staged commit (207 files, no remote), make public, SSRN, the last three behind the presentation question. **V4**: reasoning-coverage check, SPEC §13 and `DECISIONS.md` have complete coverage, no decision recorded without a reason (three terse §13 reasons and the collective frozen-parameter justification noted, neither a gap); as a by-product, the two dated D2 entries carrying pre-F1 grid spreads (0.007/0.017/0.002) were cross-linked to the U2 correction per S1's precedent. STOP after V4, not committed, not pushed, not made public, LaTeX conversion not begun, nothing submitted to SSRN. |

| First public commit, paper, PDF, and static figures held back | On instruction the repository was committed and pushed to a **public** GitHub repo with the paper deliverables held back: `paper/`, `paper.pdf`, and `figures/` added to `.gitignore` and removed from the index (a file committed once persists in history; the paper is in preparation). No frozen parameter changed and no result recomputed. README revised for a reader without the paper (paper stated in preparation, its and the figures' links removed; the gallery, the abstract-level findings summary, the two-layer paragraph, the reproduction steps, and all three scope caveats kept); `DECISIONS.md`'s open-items section no longer states the PDF is present. No other tracked file links to `paper/` or `figures/`. The commit is **172 files** (207 − 15 `paper/` − 1 `paper.pdf` − 19 `figures/`); the U4 commit message and repo description were adjusted to drop the held-back paper/figures/disclosure references. The paper, its PDF, and the static figures follow in a later commit when ready. |

| Public-repo revision: portfolio summary added, README/DECISIONS professionalized, development history made private | On the author's review of the public repository. A reader-facing portfolio summary was added, `outputs/tables/portfolio_summary.csv` and `assets/performance.png` (growth of $100k, seven models vs buy-and-hold), both regenerated by `scripts/build_readme_assets.py` from cached backtest output (no recomputation). The README was rewritten to lead with the chart and a headline statistics table (CAGR, volatility, Sharpe, max drawdown, Calmar, terminal value), with plainer section titles; `DECISIONS.md` headings tightened (the generic "What this record is for" folded into "Corrections"). The `development/` history (phase memos, session log, kickoff prompts, self-audits) was pulled from the public repository the same way as `CLAUDE.md`, gitignored, removed from the index, and force-pushed out of history, and is retained as a private working log; the `development/…` citations throughout this §13 refer to it. No frozen parameter changed and no result recomputed. |

| Subsample robustness and public-repo revision | Portfolio statistics recomputed over later start dates to test whether the edge over buy-and-hold depends on the dot-com crash. Buy-and-hold's Sharpe rises from 0.37 on the full sample to 0.72 from 2003 (which keeps 2008) and 0.87 from 2010, so the models' risk-adjusted advantage narrows from about 0.20 to roughly 0.05 to a few hundredths; the durable benefits are lower maximum drawdown and the near-equivalence of the seven models. `scripts/subsample_robustness.py`, `outputs/tables/portfolio_summary_from2003.csv` and `_from2010.csv`, README robustness section. Same round: the Section 3 data sources corrected to the daily yfinance and FRED series actually used, the intraday acceptance-gate narrative removed, the status and date columns dropped, and a prose pass across the file. No frozen parameter changed and no result recomputed. |

Append every subsequent change here with a short reason.

---

## 14. References

Andersen, T.G. and T. Bollerslev (1998). Answering the skeptics: Yes, standard
volatility models do provide accurate forecasts. *International Economic Review*
39(4), 885-905.

Andersen, T.G., T. Bollerslev, F.X. Diebold and P. Labys (2000). Great
realizations. *Risk* 13, 105-108. (Volatility signature plot.)

Bailey, D.H. and M. López de Prado (2014). The deflated Sharpe ratio. *Journal of
Portfolio Management* 40(5), 94-107.

Bandi, F.M. and J.R. Russell (2008). Microstructure noise, realized variance, and
optimal sampling. *Review of Economic Studies* 75(2), 339-369.

Barndorff-Nielsen, O.E., P.R. Hansen, A. Lunde and N. Shephard (2008). Designing
realized kernels to measure the ex post variation of equity prices in the presence
of noise. *Econometrica* 76(6), 1481-1536.

Barroso, P. and P. Santa-Clara (2015). Momentum has its moments. *Journal of
Financial Economics* 116(1), 111-120. (Six-month realized-variance scaling window.)

Clements, A. and D.P.A. Preve (2021). A practical guide to harnessing the HAR
volatility model. *Journal of Banking and Finance* 133, 106285. (Retransformation
of log-RV forecasts.)

Corsi, F. (2009). A simple approximate long-memory model of realized volatility.
*Journal of Financial Econometrics* 7(2), 174-196.

Fukasawa, M., T. Takabatake and R. Westphal (2022). Consistent estimation for
fractional stochastic volatility model under high-frequency asymptotics.
*Mathematical Finance* 32(4), 1086-1132. (Regression-based Hurst estimates cluster
near 0.10 regardless of the true value; H is weakly identified on noisy proxies.)

Gatheral, J., T. Jaisson and M. Rosenbaum (2018). Volatility is rough.
*Quantitative Finance* 18(6), 933-949.

Ghysels, E., et al. (2019). Direct versus iterated multiperiod volatility
forecasts. *Journal of Econometrics*. (Published basis for the direct-h / iterated
multi-step split used in Section 6.)

Hansen, P.R. and A. Lunde (2005). A forecast comparison of volatility models: does
anything beat a GARCH(1,1)? *Journal of Applied Econometrics* 20(7), 873-889.

Hansen, P.R. and A. Lunde (2006). Realized variance and market microstructure
noise. *Journal of Business and Economic Statistics* 24(2), 127-161.

Hansen, P.R., A. Lunde and J.M. Nason (2011). The model confidence set.
*Econometrica* 79(2), 453-497.

Harvey, C.R., E. Hoyle, R. Korgaonkar, S. Rattray, M. Sargaison and O. van Hemert
(2018). The impact of volatility targeting. *Journal of Portfolio Management*
45(1), 14-33. (Vol-of-vol falling from 4.6% to 1.8% for scaled US equities.)

Ledoit, O. and M. Wolf (2008). Robust performance hypothesis testing with the
Sharpe ratio. *Journal of Empirical Finance* 15(5), 850-859.

Liu, L.Y., A.J. Patton and K. Sheppard (2015). Does anything beat 5-minute RV? A
comparison of realized measures across multiple asset classes. *Journal of
Econometrics* 187(1), 293-311. Over 400 estimators, 31 assets, five asset classes,
tick data from January 2000 to December 2010, ranked with the model confidence
set. Little evidence any measure outperforms 5-minute RV.

Moreira, A. and T. Muir (2017). Volatility-managed portfolios. *Journal of Finance*
72(4), 1611-1644.

Patton, A.J. (2011). Volatility forecast comparison using imperfect volatility
proxies. *Journal of Econometrics* 160(1), 246-256.

Rogers, L.C.G. and S.E. Satchell (1991). Estimating variance from high, low and
closing prices. *Annals of Applied Probability* 1(4), 504-512.

Yang, D. and Q. Zhang (2000). Drift-independent volatility estimation based on
high, low, open, and close prices. *Journal of Business* 73(3), 477-491.

Zhang, L., P.A. Mykland and Y. Aït-Sahalia (2005). A tale of two time scales:
Determining integrated volatility with noisy high-frequency data. *JASA* 100(472),
1394-1411.

"Smoothing volatility targeting" (working paper, arXiv:2212.07288). Closest prior
work to the design here.

Also consult the measurement-error critique of rough volatility estimation
(Cont and Das) when reporting Ĥ.
