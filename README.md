# Portfolio Engine

A quantitative portfolio research engine for long-horizon asset allocation, portfolio optimization, and risk simulation.

This project implements two repeatable workflows:

- a live engine for rebalance decisions on the current portfolio
- a research engine for testing new baskets and portfolio ideas

Core and research analytics across the repo include:

- Black-Litterman expected return modeling  
- Mean-Variance Portfolio Optimization  
- Efficient Frontier and Capital Market Line  
- Rolling market regime analysis  
- Monte Carlo long-horizon risk simulations  
- Portfolio risk decomposition  

The model is designed to allow rapid asset rotation by editing the live or research configuration and re-running the relevant entrypoint.

---

# Requirements

- Python 3.10+
- Internet connection (for downloading market data)

---

# Usage

There are two different launcher files depending on which engine you want to run:

- Live engine launcher: `run_live.bat` on Windows or `run_live.command` on macOS
- Research engine launcher: `run_research.bat` on Windows or `run_research.command` on macOS

Run the live rebalance engine with:

```
python main_live.py
```

Or use the launcher:

- Windows: `run_live.bat` for the live engine
- macOS: `run_live.command`

Run the research engine with:

```
python main_research.py
```

Or use the launcher:

- Windows: `run_research.bat` for the research engine
- macOS: `run_research.command`

For backwards compatibility, `python main.py` still runs the research workflow.

The `.bat` and `.command` files will:

1. Create `venv/` if it does not exist
2. Sync `requirements.txt` on every run
3. Run the matching engine

Update shared assumptions in:

```text
config/shared.py
```

Update live portfolio definitions in:

```text
config/live.py
```

Update research basket definitions in:

```text
config/research.py
```

---

## Changing Assets

To rotate assets or test new portfolios, modify:

```
config.py
```

Then rerun:

```bash
python main.py
```

---

# Key Features

## Portfolio Optimization

Computes optimal allocations using Modern Portfolio Theory.

Outputs include:

- Efficient Frontier  
- Maximum Sharpe portfolio  
- Minimum volatility portfolio  

---

## Black-Litterman Return Model

Combines:

- historical returns  
- market equilibrium returns  
- investor views  

to produce more stable expected return estimates than historical averages alone.

---

## Long Horizon Risk Simulation

Uses multivariate correlated Monte Carlo simulations to estimate:

- terminal wealth distributions  
- drawdown probabilities  
- probability of portfolio loss  
- long-term compounding outcomes  

---

## Portfolio Diagnostics

The engine generates a decision dashboard including:

- Current Portfolio Performance  
- Optimized Portfolio  
- Suggested Weight Changes  
- Risk Contribution by Asset  
- Monte Carlo Risk Profile  

---

# Example Outputs

## Efficient Frontier

Shows the optimal tradeoff between expected return and volatility.

Key points plotted:

- Starting portfolio  
- Maximum Sharpe portfolio  
- Minimum volatility portfolio  
- configured benchmark  

---

## Rolling Market Environment

Uses a rolling 5-year window to show:

- changing return regimes  
- volatility clustering  
- market cycles  

---

## Monte Carlo Risk Distribution

Simulates long-term outcomes to estimate:

- terminal wealth  
- maximum drawdown risk  
- loss probabilities  
- tail risk scenarios  

---

# Project Structure

```text
Portfolio-Engine
│
├ config/
│ ├ __init__.py
│ ├ shared.py
│ ├ live.py
│ └ research.py
├ data.py
├ mpt.py
├ frontier.py
├ black_litterman.py
├ simulation.py
├ rollingfront.py
├ dashboard.py
├ report.py
├ main_live.py
├ main_research.py
└ main.py
```

---

# Module Overview

### config/shared.py
Defines shared model assumptions and implementation-cost settings.

### config/live.py
Defines the live holdings universe, current weights, live constraints, and rebalance thresholds.

### config/research.py
Defines the candidate basket, research window, backtest settings, and research outputs.

### data.py
Downloads historical asset prices and computes returns.

### mpt.py
Implements portfolio optimization functions.

### frontier.py
Constructs the efficient frontier.

### black_litterman.py
Calculates equilibrium returns and posterior expected returns using the Black-Litterman framework.

### simulation.py
Runs multivariate Monte Carlo portfolio simulations.

### rollingfront.py
Calculates rolling return and volatility statistics.

### dashboard.py
Generates portfolio diagnostics including weight changes and risk contribution.

### report.py
Builds performance summaries and cumulative return series.

### main_live.py
Runs the live rebalance workflow.

### main_research.py
Runs the research workflow.

### main.py
Backwards-compatible dispatcher to the research workflow.

---

# Usage

There are two different launcher files depending on which engine you want to run:

- Live engine launcher: `run_live.bat` on Windows or `run_live.command` on macOS
- Research engine launcher: `run_research.bat` on Windows or `run_research.command` on macOS

Run the live rebalance engine with:

```
python main_live.py
```

Or use the launcher:

- Windows: `run_live.bat` for the live engine
- macOS: `run_live.command`

Run the research engine with:

```
python main_research.py
```

Or use the launcher:

- Windows: `run_research.bat` for the research engine
- macOS: `run_research.command`

For backwards compatibility, `python main.py` still runs the research workflow.

The `.bat` and `.command` files will:

1. Create `venv/` if it does not exist
2. Sync `requirements.txt` on every run
3. Run the matching engine

Update shared assumptions in:

```text
config/shared.py
```

Update live portfolio definitions in:

```text
config/live.py
```

Update research basket definitions in:

```text
config/research.py
```

---


This project is designed as a portfolio research framework that can be reused whenever assets are rotated or portfolio assumptions change.

The goal is to provide a repeatable and transparent method for long-term portfolio construction and evaluation.

---

# Author

**Nicholas Clervi**  
Economics / Quantitative Finance Research

## Phase 1: Market Intelligence and correctness audit

Run `python market_intelligence.py` for a compact console dashboard, or
`python market_intelligence.py --json` for the structured snapshot and numeric
features. Optional `--start YYYY-MM-DD` and `--end YYYY-MM-DD` control retrieval.
Yahoo's end date is exclusive; FRED's end date is inclusive. No Streamlit
application or additional dependency was introduced.

The existing architecture is retained: `main_live.py` handles holdings and
rebalance recommendations; `main_research.py` handles optimization, factor
analysis, walk-forward testing, simulation, and reports; `main.py` dispatches
research. `mpt.py` uses PyPortfolioOpt optimization and covariance estimation
(including Ledoit-Wolf); `frontier.py` constructs the frontier; `factors.py`
loads Ken French data and estimates regressions; `simulation.py` remains the
Gaussian baseline. `dashboard.py`, `report.py`, and `implementation.py` retain
risk contributions, reporting, cost calculations, and approximate tax drag.
Holdings/constraints remain in `config/live.py` and `config/research.py`.

`market_intelligence.py` reuses `data.py` downloads and `dashboard.py` tables.
Yahoo adjusted Close series cover equities, duration, cash proxies, credit,
TIP, gold, commodities, REITs, dollar (`DX-Y.NYB`) and volatility (`^VIX`).
Direct Treasury yields use keyless [FRED CSV data](https://fred.stlouisfed.org/series/DGS10)
(DGS3MO, DGS2, DGS5, DGS10, DGS30), never Treasury ETF prices. Source downloads
are cached for one hour under `outputs/market_cache/`; missing providers,
maturities, or tickers are reported without preventing the other signals.

Backend interfaces:

- `build_market_snapshot(prices, yields, as_of=..., zscore_window=63)` is
  deterministic and independent of downloads. Prices have logical ticker
  columns; yield columns are `3MO`, `2Y`, `5Y`, `10Y`, `30Y`, in percent per year.
  The snapshot separates observations, derived signals, and interpretations.
- Horizons 1/5/21/63/252 count available observations, not calendar days.
  Relative signals use joint valid observations and subtract simple returns.
  Price-return changes are decimal fractions; yield/spread changes are basis
  points (one percentage point = 100 bp). Data is never forward-filled.
  Each metric carries its observation date and stale flag. Stale observations
  remain visible but cannot generate features or interpretations. Mixed yield
  moves are labeled twists/mixed moves, rather than forced bull/bear labels.
- Z-scores use a full configurable window and sample standard deviation;
  insufficient history and zero variance produce missing values. These are
  descriptive level z-scores, not calibrated prediction statistics.
- `get_regime_features(snapshot, horizon="21D")` returns a flat mapping of
  finite numeric proxies; missing inputs are omitted. TIP/IEF is a relative
  ETF proxy, not a direct real yield. No regimes, probabilities, portfolio
  adjustments, or ML models are implemented.

Correctness changes and conventions:

- BL confidence is in [0,1], using PyPortfolioOpt's
  [Idzorek uncertainty](https://pyportfolioopt.readthedocs.io/en/latest/BlackLitterman.html):
  Omega = tau * (1-c)/c * P Sigma P'. Zero-confidence views are excluded.
  Absolute views are annual total returns; relative tuples are
  `(outperformer, underperformer, annual_return_difference, confidence)`.
  The total-return prior adds the resolved risk-free rate to equilibrium excess
  returns. `BL_STRATEGIC_PRIOR_WEIGHTS` is separate from holdings. Until supplied,
  an explicitly warned equal-weight modeling prior is used; it is not a claim
  about market capitalization. `BL_MARKET_WEIGHTS` remains a legacy alias.
- `resolve_risk_free_rate()` / `get_risk_free_rate()` in `data.py` centralize the
  annual decimal rate. Set `config/shared.py:RISK_FREE_RATE` to override; otherwise
  DGS3MO is used if no more than seven calendar days old. The labeled fallback
  assumption is `RISK_FREE_FALLBACK=0.0`. Research resolves historical yields at
  each rebalance date and records the actual rate in diagnostics.
- Walk-forward windows contain exactly `window_days` observations through t;
  their weights first affect the next tradable return. Monthly decisions use
  the last available trading date, including weekend month ends; final-date
  decisions with no subsequent return are excluded. Empty backtests are safe.
  Solver weights retain precision instead of rounding away constraints.
- Price returns no longer fill missing prices. Optimizer samples use complete
  asset rows. Drawdowns include starting wealth, and Monte Carlo loss/multiple
  probabilities use the supplied initial value. Simulation metadata labels
  IID Gaussian simple returns, constant moments/weights, and excluded costs,
  taxes, tails, and regimes. `method="gaussian"` reserves an explicit extension
  point without adding alternative simulation implementations.
- Tax outputs are explicitly approximate, based on assumed gains/holding
  periods without lots. Gains are fractions of market value sold. Cost turnover
  is two-sided traded notional; the existing return penalty is a holdings-bias
  heuristic, not an actual turnover-constrained optimizer.

Validation: `python -m unittest discover -s tests -v` runs offline regression
checks for signal arithmetic, units, curve cases, missing/stale data, source
outages/cache, BL confidence/relative views/prior, risk-free timing, optimizer
constraints/shrinkage, walk-forward timing, and initial-wealth risk statistics.
Python 3.11 syntax is supported; the available validation runtime is Python 3.12.

Remaining modeling limits: FRED observation dates are not release timestamps
or vintage data; historical views are static user assumptions and may carry
hindsight bias. The walk-forward baseline uses constant weights daily, excludes
realized trading/tax costs and weight drift, and reports gross performance.
Historical Sharpe reports use a scalar annual risk-free assumption. ETF credit
and TIP comparisons carry duration/mix effects. Normal simple-return simulations
can theoretically draw below -100%. No tax-lot accounting or causal inference
is provided. Live provider availability must be checked in a network-enabled run.

## Phase 2: historical regimes and forward asset outcomes

Run `python regimes.py` for the current state, descriptive 63-observation median
forward returns, and historical diagnostics. `--json` returns structured current
state, diagnostics and full 21/63/126-observation outcome tables.
`--non-overlapping` selects disjoint outcome intervals. `--start` defaults to
2007-01-01 to provide standardization history; `--end` is an optional evaluation
cutoff (Yahoo retrieval remains exclusive at the end date).

Phase 1 interfaces and portfolio workflows are unchanged. `regimes.py` provides:

- `build_regime_feature_history(prices, yields)` vectorizes the existing Phase 1
  feature definitions for 21D and 63D, using the price-session calendar. Returns
  are differences of adjusted-price returns; yield changes are basis points.
  Available features can be carried forward for at most seven calendar days;
  nothing is backfilled. Treasury observations become available after a default
  one-calendar-day lag. This conservative lag is configurable and does **not**
  establish actual publication-time or vintage accuracy.
- `standardize_regime_features(features)` uses a trailing 756-session window,
  requires 252 valid observations per feature, and clips z-scores to [-3,3].
  The current observation is included. Zero variance or insufficient history
  stays missing; there is no full-sample normalization.
- `composite_regime_scores(standardized)` averages horizons within each proxy
  first, then averages proxy families. At least two families per composite are
  required. Available family weights are renormalized and coverage reported.
  Growth uses IWM/SPY, RSP/SPY, XLY/XLP, HYG/LQD and SPY momentum. Inflation/long-rate
  pressure uses 10Y changes, TIP/IEF, DBC and GLD; gold has half the weight of
  the other inflation proxies. Financial conditions use 2Y changes, DXY, VIX
  and **negative** HYG momentum: positive means tighter conditions. Stress uses
  VIX z-score, SPY realized volatility, credit weakness and breadth weakness.
- `classify_regime_history(features)` exposes raw composite scores, 10-session
  EWMA scores, `raw_regime`, `smoothed_regime`, `primary_regime`, probabilities,
  confidence, counts, coverage, and overlays. `build_regime_history(prices,
  yields, feature_options=..., **model_options)` wraps feature construction.
  Growth-positive/inflation-nonpositive is Goldilocks; both positive Reflation;
  both nonpositive Slowdown; growth-nonpositive/inflation-positive Stagflation.
  These are **market-proxy** quadrants relative to trailing feature norms,
  not measured GDP/inflation states or causal explanations.
- Probabilities are softmax of (+g-i, +g+i, -g-i, -g+i) on smoothed scores, with
  temperature 1.0. They are **heuristic normalized scores, not calibrated
  economic probabilities**. Confidence is the top-two probability gap times
  minimum growth/inflation coverage. Insufficient core data yields an
  `unavailable` regime, uniform scores, and zero confidence. Missing scores
  cannot be silently carried through the smoothing output.
- Financial conditions are easy <= -0.5, tight >= +0.5, otherwise neutral.
  Stress is low < -0.5, normal [-0.5,0.5), elevated [0.5,1.5), severe >= 1.5.
  These are overlays; they never add primary regimes.
- `get_current_regime(prices, yields, as_of=..., **model_options)` returns
  JSON-safe current scores, raw/smoothed labels, probabilities, confidence,
  overlays, coverage, quality flags and Phase 1 numeric input features. It
  reuses one Phase 1 snapshot, not repeated historical snapshots. Use `as_of`
  to check freshness or evaluate a historical cutoff. Defaults use the last
  input date; `observation_as_of` identifies the last history row separately.
- `analyze_regime_forward_returns(history, prices, horizons=(21,63,126),
  non_overlapping=False, as_of=...)` measures P[t+h]/P[t]-1: all return intervals
  start **after** the classification date. It reports count, mean, median,
  sample standard deviation, positive frequency, and 25th/75th percentiles for
  the configured equity, bond, cash, credit, TIP, gold, commodity and REIT ETFs.
  SGOV and BIL are reported separately when available. Missing asset endpoints
  and unfinished outcomes are excluded. Horizons use the shared observed price
  calendar. Daily samples overlap and are **not independent**; non-overlapping
  selection is global within each asset/horizon, not separate per regime.
- `estimate_regime_expected_returns(history, prices, as_of=...)` returns raw,
  unconditional, and shrunk same-horizon cumulative return estimates, counts,
  and lambda. Default lambda=n/(n+20), using non-overlapping completed outcomes;
  `lambda_override` is optional. No conditional sample yields the unconditional
  fallback. Estimates are not annualized, calibrated forecasts, or BL views.
  **For every historical decision, supply its as_of cutoff**, which truncates
  prices before calculating both conditional and unconditional outcomes.
- `regime_diagnostics(history)` reports frequency, mean run duration in
  observations, transitions, sample counts and latest state. Missing states
  break runs; first/last runs are censored, and frequency excludes unavailable
  states.

All defaults live in `config/shared.py`. Weights and thresholds are static;
none are optimized using forward returns. No new dependencies, portfolio
allocation changes, BL view changes, or regime simulation were introduced.

Tests: `python -m unittest discover -s tests -v`. The added checks cover Phase 1
feature parity, prefix/future-perturbation invariance, trailing standardization,
economic signs, smoothing, overlay thresholds, probability directions, missing
and stale inputs, Treasury availability lag, post-decision returns, disjoint
samples, cutoff-safe shrinkage, diagnostics, and offline CLI output.

Limitations: regimes reflect relative market behavior rather than observed
macro fundamentals; ETF duration and composition affect proxies. Softmax scores
and confidence are uncalibrated. Missing history changes component coverage,
new ETFs have shorter samples, overlapping outcomes inflate nominal counts,
and results remain descriptive. A one-day Treasury lag cannot substitute for
release timestamps or historical vintages. Portfolio integration is deferred
to Phase 3.

# Streamlit dashboard (Phase 4)

From the repository root, install `requirements.txt` into your environment and run:

```bash
python -m streamlit run ui/app.py
```

With the existing project environment, use `venv/bin/python -m streamlit run ui/app.py`.
Edit percentage weights, strategic policy and settings in the sidebar, then click
**Run Analysis**. Seven views share the saved backend result. Changing pages,
chart horizons or inputs does not run optimization. **Refresh Market Data**
explicitly refreshes the provider data; run analysis again to incorporate it.
State lasts for the browser session; download the recommendation JSON to retain it.
Defaults are illustrative. Per-position costs are unavailable because the engine
provides aggregate estimates only. Existing CLI and research commands remain available.
