# Portfolio Engine

A local portfolio-analysis app with a browser dashboard for market intelligence,
macro indicators, portfolio construction, risk, robustness and rebalance review.
It runs on your computer and does not connect to a broker or execute trades.

## Download and launch — no Python experience needed

You need an internet connection, a web browser and a writable folder on a
Mac, Windows PC or Linux computer. **You do not need to install Python or Git.**
The launchers use [Astral uv](https://docs.astral.sh/uv/getting-started/installation/)
to download Python 3.11 and install the app's dependencies automatically.

### 1. Download from GitHub

1. Open [the Portfolio Engine repository](https://github.com/v8nick/Portfolio-Engine).
2. Click the green **Code** button, then **Download ZIP**.
3. Extract/unzip the download. On Windows, right-click it and choose **Extract All**.
4. Open the extracted folder, usually named `Portfolio-Engine-main`. You should
   see `README.md`, `run.command`, `run.ps1`, `requirements.txt` and the `ui` folder.
   Run the app from this extracted folder, not from inside the ZIP preview.

### 2. Start the app

**macOS**

Double-click **run.command**. A Terminal window opens and performs setup.

If macOS reports a permission error or will not open the downloaded launcher:

1. Open **Terminal** (press Command+Space, type Terminal, then press Return).
2. Type `bash `, including the space after it.
3. Drag `run.command` from Finder into Terminal, then press Return.

This runs the same launcher without needing to change its executable permissions.
Do not disable macOS security settings. If macOS blocks downloaded software,
review its notice; the manual Python option below is also available.

**Windows**

1. In File Explorer, open the extracted app folder.
2. Click the address bar, type `powershell`, then press Enter. A PowerShell
   window opens in that folder.
3. Paste this command and press Enter:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1
```

The execution-policy option applies only to this launcher process; it does not
permanently change your Windows policy. The script downloads the official uv
installer from `astral.sh`. No administrator window is required. If your employer
blocks scripts or downloads, follow its IT policy rather than changing it.

**Linux**

Open a terminal in the extracted folder and run:

```bash
bash run.command
```

Bash and curl must be available; install curl with your distribution's package
manager if it is missing.

### 3. Wait for setup, then open the dashboard

The first launch downloads Python and the required packages. It can take several
minutes; progress appears in the terminal. Later launches reuse the downloaded
runtime and cached packages. Do not close the window while setup is running.

The browser should open automatically. If it does not, visit
[http://localhost:8501](http://localhost:8501).

**Keep the terminal window open while using the app.** Closing the browser does
not stop it. To stop the app, press **Control+C** in the terminal. To start again,
use the same launcher.

The setup tool, Python and package cache are stored in `.runtime/` inside the
app folder. The launchers do not overwrite an existing `venv/` or `.venv/`,
modify your shell profile or require a system Python installation. If uv is
already on your PATH, the launcher can reuse it.

## Use the dashboard

The app opens without downloading market data or running portfolio analysis.

To import a portfolio, expand **Import portfolio CSV** on the Home page,
choose a `.csv` file, select **Add / update tickers** or **Replace table**, and
click **Import CSV**. A `Ticker` (or `Symbol`) column is required; a single-column
list without a header also works. Download the basic CSV template for ticker,
current weight and target weight, or the advanced template for all policy columns.
`Target %` and `Strategic %` mean the same thing. A simple file looks like:

```csv
Ticker,Current %
SPY,60
IEF,40
```

Weights use percentage points (`25` or `25%` means 25%; `0.25` means 0.25%).
Add/update preserves other tickers and settings omitted from the CSV. Replace
keeps only the imported tickers, preserving their existing settings where columns
are omitted. New tickers begin at 0% current/strategic weight with satellite roles;
review weights, roles, groups and limits before analysis. Set both weight totals
to 100%; imports do not normalize weights or run the analysis automatically.
Duplicate tickers, unsupported columns and invalid percentages are rejected
without changing the table. Broker exports with share counts or dollar values
must be converted to this percentage format first.

1. On **Home**, enter your **portfolio value** and edit the portfolio table.
   All weights in the editor are percentages: enter `25` for 25%.
2. The basic editor shows only **Ticker**, **Current %** and **Target %**.
   Set your current weights and strategic targets. Each weight column must total
   100%. Strategic policy represents your intended long-term allocation and is
   separate from what you currently hold.
3. Enable **Show advanced allocation settings** if you want to edit tactical
   **Low / High**, hard **Minimum / Maximum**, optional **Fixed**
   weights, asset groups and core/satellite roles. Scroll the table horizontally
   or use its fullscreen button to reach all columns. Add/delete rows as needed.
   Leave Fixed blank to allow a weight to move; a fixed weight is used only when
   it matches the target. Known individual stocks are classified as satellites.
   Advanced fields are optional. In the basic view, bands are generated at target
   ±5 percentage points (clipped to 0–100%); hidden hard limits and fixed weights
   are ignored. In the advanced view, supplied bands/limits expand to include the
   entered target, and a conflicting fixed weight is ignored. Optional concentration
   limits also expand when targets exceed them. The app displays adjustments and
   records them with the saved analysis. Main-editor targets always take priority;
   percentages must still be valid and current/target totals must each be 100%.
   Known tickers receive configured classifications when omitted; unknown tickers
   start as satellites. Classifications can be edited in the advanced view.
4. Review satellite limits and the settings expander. The default strategy is
   illustrative; customize it before relying on any recommendation.
5. Click **Apply Portfolio & Run Analysis** and wait for completion. For a quicker initial check,
   reduce bootstrap repetitions from 100 to 20. This changes how many sensitivity
   checks run, not the underlying financial methodology.
6. Navigate between pages to inspect the saved result:

| Page | What it shows |
| --- | --- |
| Executive overview | Underlying growth and inflation/rate-pressure metrics, allocation comparison, risk metrics and actions |
| Market intelligence | Treasury yields, curve, cross-asset signals and backend interpretations |
| Macro indicators | Actual model input changes, score construction, data coverage and historical returns by market pattern |
| Portfolio construction | Strategic, minimum variance, BL max Sharpe, risk parity, CVaR and recommended allocations |
| Risk | Tail risk, concentration, ticker correlation heatmap, asset risk contributions and configured group exposures |
| Robustness | BL max-Sharpe bootstrap weight stability |
| Rebalance | Backend action labels, required policy repairs, aggregate cost and approximate tax estimates |
| Backtest | Point-in-time strategy comparisons, realized costs, rolling metrics and historical regime outcomes |
| Simulation | Four forward simulation methods, cash flows, scenario transitions, fan charts and stress replays |

Page navigation, chart selections and input edits do not rerun the portfolio
engine. Click **Apply Portfolio & Run Analysis** to apply edited inputs. **Refresh Market Data**
requests fresh provider data; it leaves the saved portfolio recommendation intact
until you run analysis again. A failed run retains the last successful result.

The Risk page correlation matrix uses daily returns over the saved analysis’s
common risk sample and includes every ticker in its universe. Blue (≤ −0.30)
indicates opposite movement / a potential hedge; green (between −0.30 and +0.30)
indicates low correlation; orange (≥ +0.30) indicates assets tending to move
together. These descriptive thresholds do not guarantee hedging or diversification.
Hover for exact values or download the correlation CSV. Gray cells represent
unavailable correlations, such as constant-return series. Run Analysis again to
add the matrix to a result saved before this feature was available.

State lasts for the current browser session. Use **Download recommendation JSON**
to save the result. It is an export, not a configuration-import feature.

## Troubleshooting

| Problem | What to do |
| --- | --- |
| First launch appears slow | Keep the terminal open and read the download progress. Initial setup is larger than subsequent launches. |
| Setup/download error | Check your connection, available disk space and folder permissions. Retry the same launcher; cached downloads can be reused. |
| Browser does not open | Open `http://localhost:8501` manually after setup finishes. |
| Port 8501 is already in use | An earlier copy may still be running. Stop it with Control+C, then relaunch. |
| Invalid weights or policy | Both current/target totals must equal 100%, percentages must be finite and between 0 and 100, and tickers must be unique. Optional limits adjust to accommodate targets. |
| Missing ticker or insufficient history | Check the Yahoo ticker spelling and history start date. Required portfolio assets are not silently dropped. New listings may lack enough common history. |
| Treasury/Yahoo data unavailable | Review the warnings, refresh data and retry later. Optional market signals may be missing; missing required asset data can prevent a new recommendation. |
| Optimizer or bootstrap unavailable | Other available results still render. Read the warnings and confidence assessment. |
| Results show old inputs | Click Run Analysis after editing; until then, pages show the last successful analysis and its timestamps. |

For setup diagnostics on macOS/Linux, run `bash run.command --check`. On Windows,
use the launch command above with `-Mode check` appended. These install/check the
runtime and import the app and engines without downloading market history.

## Update the app

Download and extract a newer ZIP from GitHub, then launch it the same way. A new
folder performs its own setup. Export any recommendation you want to retain first.
Browser-session settings do not transfer automatically. If you customized files
in `config/`, preserve those changes before replacing a checkout.

## Existing Python users and research workflows

You can still use a Python 3.11 virtual environment directly. From the repository root:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux, or
`.venv\Scripts\Activate.ps1` in Windows PowerShell, then run:

```bash
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
```

An existing working environment can launch Streamlit without reinstalling Python.
For example, this repository's local `venv` uses
`venv/bin/python -m streamlit run ui/app.py` on macOS/Linux.

All existing non-UI entry points remain available:

| Command in your Python environment | Workflow |
| --- | --- |
| `python main_live.py` | Legacy live portfolio console report |
| `python main_live.py --decision` | Phase 3 recommendation JSON |
| `python portfolio_decision.py` | Same decision engine CLI |
| `python main_research.py` | Research, frontier, factors, walk-forward analysis, simulation and reports |
| `python main.py` | Backwards-compatible research entry point |
| `python market_intelligence.py --json` | Structured market snapshot |
| `python regimes.py --json` | Regime assessment, diagnostics and conditional outcomes |
| `python -m unittest discover -s tests -v` | Offline backend and UI tests |

Without managing Python yourself, use `bash run.command --live` or
`bash run.command --research` on macOS/Linux. On Windows, append `-Mode live` or
`-Mode research` to the PowerShell launch command. The macOS/Linux live mode also
passes through options, e.g. `bash run.command --live --decision`.

The old `.bat` files and duplicate live/research `.command` launchers were replaced
by these two consolidated launchers. The console workflows themselves remain.

## Backend and configuration

- `ui/`: Streamlit presentation, input validation, cached data adapters and views.
  It calls the backend; backend modules do not import Streamlit.
- `portfolio_decision.py`: `build_portfolio_recommendation(prices, yields,
  current_weights=None, strategic_allocation=None, *, as_of=None, manual_views=None,
  relative_views=None, risk_free_rate=None, options=None)`, a pure JSON-safe
  financial API. Supply adjusted prices and Treasury yields; it does not fetch data.
- `market_intelligence.py`: cached source retrieval, deterministic market snapshots
  and numeric features. Treasury yields come from keyless FRED CSV; adjusted prices
  come from Yahoo Finance. Source data is cached in `outputs/market_cache/`.
- `regimes.py`: historical market-proxy classification, heuristic probabilities,
  diagnostics, completed forward outcomes and shrunk conditional return estimates.
- `mpt.py`, `black_litterman.py`: existing optimization, covariance estimation,
  strategic prior, manual/regime view merging, Idzorek BL confidence, risk parity
  and CVaR routines.
- `dashboard.py`, `report.py`, `implementation.py`: console tables, risk metrics,
  costs and approximate taxes. `dashboard.py` is a backend reporting module,
  separate from the browser UI.
- `frontier.py`, `factors.py`, `rollingfront.py`, `simulation.py`, `walkforward.py`:
  research tools used by the existing research workflow.
- `config/live.py`: default holdings, illustrative strategic policy, satellite
  mappings and configured manual BL opinions.
- `config/shared.py`: shared model, cost, tax and decision assumptions.
- `config/research.py`: research basket and research workflow settings.
- `.streamlit/config.toml`: dashboard appearance.

Home edits affect the current session; they do not rewrite `config/` files.
The UI continues using configured manual BL opinions and cost/tax assumptions.

## Interpretation and limitations

- The dashboard shows underlying market metrics rather than assigning a current economic regime or displaying regime confidence/probabilities. Composite scores are standardized market proxies, not GDP growth or CPI inflation.
- The backend still uses heuristic historical market-pattern weights for portfolio return adjustments. The downloadable JSON and command-line regime report retain those model diagnostics; this presentation change does not alter allocation calculations.
  Probabilities and confidence are heuristic and are not calibrated certainty.
- Conditional outcomes describe history, not forecasts. Expected returns use the
  backend's annual arithmetic convention. Yield levels are percent per year;
  yield changes and curve spreads are basis points.
- FRED observation dates and the backend availability lag do not establish
  release-time or historical-vintage accuracy. ETF credit and inflation proxies
  also reflect duration and composition differences.
- Portfolio historical risk uses constant daily weights and excludes realized
  trading costs and taxes. VaR/CVaR are empirical daily losses. Bootstrap stability
  tests BL max-Sharpe only, not every optimizer.
- Recommended ranges are policy-bounded ranges, not statistical confidence
  intervals. Rebalance actions are review labels, not a self-financing order list.
- Costs are aggregate approximations; the UI cannot show genuine per-asset costs.
  Taxes use assumed gains and holding periods, without tax-lot optimization.
- Legacy console walk-forward weights first apply after the decision observation;
  those outputs remain gross constant-weight approximations. The console Monte Carlo baseline
  uses IID multivariate Gaussian simple returns and constant moments/weights;
  it excludes costs, taxes, regime changes and realistic tail behavior.
- Live provider availability is not guaranteed. No broker integration, account
  connection, authentication or trade execution is included.

Nicholas Clervi · Economics / Quantitative Finance Research

## RESEARCH VALIDATION & REGIME SIMULATION

This phase retains the live recommendation engine, the console workflows and the
legacy `rolling_black_litterman_backtest` API. It adds **Backtest** and **Simulation**
to the top navigation. Run portfolio analysis once to save the input history, then run
research explicitly on those pages. Editing a research control never downloads
market data or reruns a study automatically. Last successful results remain
visible with a stale-input warning until another run succeeds.

An **optimizer** chooses weights from supplied expected returns, covariance and
constraints. A **backtest** measures what a specified rule would have done through
historical observations. A **simulation** generates hypothetical future paths
under stated assumptions. Their outputs are separate: backtest performance is
never fed into current expected-return estimates, and no parameter search is
performed to maximize historical Sharpe.

### Expected returns and estimation

`expected_returns.estimate_expected_returns` is the research return-estimation
interface. Covariance is supplied separately by `mpt.annualize_mean_cov`, using
existing Ledoit–Wolf shrinkage by default. Supported sources are:

- `bl`: Black–Litterman strategic equilibrium prior, plus explicitly dated views
  when provided. With no views, the posterior equals the prior. Strategic prior
  weights are fixed modeling assumptions, not inferred market capitalizations.
- `historical_mean`: annual arithmetic sample mean, labeled a research comparison;
  historical fit does not establish better forecasting.
- `bl_regime`: the existing conservative BL regime adjustments, based only on
  completed historical conditional-return samples available at each decision.

Every optimized backtest decision records the model, estimation window, covariance
method and risk-free provenance. Today's configured manual opinions are excluded
from backtests. The API optionally accepts a dated `view_history`; only the latest
record available on the decision date is applied. Production recommendations
retain their existing BL/manual-view/regime behavior and now identify their source
in exported expected-return metadata.

### Walk-forward methodology

`walkforward.walk_forward_backtest(prices, yields, current_weights,
strategic_allocation, ...)` compares current and strategic allocations, equal
weight, SPY, 60/40 SPY/IEF, BL max Sharpe, minimum volatility, risk parity and minimum
CVaR. Monthly and quarterly decisions occur on actual observed period-end sessions.
The estimation window ends on the decision date. New holdings take effect on the
**next** observation, with shares drifting between rebalances.

Optimized allocations enforce the supplied policy, fixed positions and group
limits. Small changes below the no-trade threshold are skipped unless a constraint
repair is required. Reference current/equal-weight/SPY/60-40 portfolios are explicitly
unconstrained comparisons; they are not presented as compliant optimized portfolios.
References begin endowed at their stated allocation; optimized strategies begin
with the supplied current holdings and pay for any initial reallocation.

Actual two-sided turnover is the sum of absolute buy and sell weights. At a trade,
`cost_rate = turnover × (transaction_cost_bps + slippage_bps) / 10,000`, and net
return is `(1 − cost_rate) × (1 + gross_return) − 1`. No one-time cost is subtracted
from an annual expected return. Diagnostics show turnover, NAV cost rates and
actual cost amounts per initial $10,000. Taxes are excluded; existing production
tax estimates remain approximate and are not tax-lot calculations.

The first test decision needs the full trailing lookback and a subsequent trading
observation. The page defaults to the first complete estimation window for the
selected universe; **Use available history** updates an existing start date after
changing assets or lookback. This accommodates later asset inception dates without
inventing earlier prices. Trading sessions use dates with at least one requested
asset quote. Macro-only dates (for example dollar-index quotes on stock-market
holidays) are excluded before calculating returns; individual missing asset quotes
remain missing. This is an observed-data calendar, not an exchange-calendar feed,
and cannot identify a provider outage affecting every requested asset simultaneously.
Missing out-of-sample history makes that strategy unavailable; no
assets are silently substituted or prices forward-filled. Benchmark returns align
by date; unavailable benchmarks leave relative metrics unavailable. An optimizer
failure is recorded and existing holdings are retained rather than switching to
another optimizer. This can affect performance and must be reviewed in diagnostics.

Look-ahead protection includes input truncation, trailing covariance/means,
causal regime scores, exclusion of incomplete forward outcomes, dated views and
next-observation execution. Prior-session regime labels condition realized returns.
These controls do not eliminate survivorship bias or reconstruct publication-vintage
macro/adjusted-price data. A current ticker universe and policy used historically
are hypothetical assumptions, not a history of the user's actual account.

### Performance measures

The research report uses 252 observations per year. CAGR is compounded geometric
return. Volatility and tracking error use sample standard deviation × √252.
Sharpe uses arithmetic mean daily excess return ×252 divided by annualized
volatility; annual risk-free rates are divided by 252. Sortino uses the root mean
square of negative daily excess returns over **all** observations. Alpha is the
annual arithmetic excess-return intercept, and beta uses aligned excess returns.
Calmar is CAGR divided by absolute maximum drawdown. Undefined ratios show N/A.

VaR and expected shortfall are empirical **daily** losses at 95% confidence.
Monthly and calendar returns compound actual observations (edge months/years can
be partial). Rolling one-year returns, volatility and Sharpe need 252 observations;
three-year annualized return and Sharpe need 756. Drawdown includes initial wealth,
and a trailing one-year maximum drawdown series is also provided.

Regime tables show observation count, annualized return, volatility, Sharpe (with
at least 20 observations), drawdown and positive-day frequency. Regime subsets
concatenate noncontiguous days, so their drawdowns and annualized returns are
**descriptive historical outcomes**, not a tradeable path or forecasts.

### Forward simulation methods

`simulation.run_simulation(weights, mu, cov, ...)` returns compact statistics,
percentile paths, terminal values and assumptions. The old
`simulate_portfolio_paths` function remains available and returns full paths when
explicitly requested by legacy callers.

- **Gaussian:** IID multivariate normal simple returns using the supplied annual
  mean/covariance. This is a baseline, not an assertion of realistic tails.
- **Student-t:** multivariate normal innovations divided by a shared chi-square
  scale for each synchronized asset draw. Scaling uses `(df−2)/chi-square(df)`
  so the finite target covariance is retained before loss-floor handling. Default
  degrees of freedom is 5; values must exceed 2.
- **Block bootstrap:** preferred nonparametric mode. Samples synchronized,
  contiguous multi-asset historical blocks, preserving cross-asset rows and local
  serial dependence. Block length and seed are configurable. Only supplied data
  through the simulation cutoff is used.
- **Regime switching:** uses the existing Goldilocks, Reflation, Slowdown and
  Stagflation market-proxy classifications. A daily Markov transition matrix is
  estimated from adjacent historical labels, including persistence. Unavailable
  labels break transitions. Rows shrink toward smoothed observed occupancy; no
  classified history produces an explicitly labeled uniform prior. Returns are
  synchronized empirical asset rows associated with the **prior-session** state.
  With fewer than 30 conditional observations, draws mix that pool with the
  unconditional pool using probability `n/30`; missing regimes use unconditional
  history entirely. Every fallback is reported. This mode samples daily conditional
  rows, not contiguous within-regime blocks.

Starting state can be Current, any explicit regime, or Random historical. If the
current state is unavailable, select an explicit state or Random historical.
Scenarios modify destination probabilities and persistence, never apply arbitrary
asset losses. Before row normalization, destination multipliers in the order
Goldilocks/Reflation/Slowdown/Stagflation are:

| Scenario | Multipliers | States with self-transition additionally doubled |
|---|---|---|
| Baseline | 1, 1, 1, 1 | None |
| Growth boom | 2, 2, 0.5, 0.5 | Goldilocks, Reflation |
| Persistent slowdown | 0.5, 0.5, 3, 1 | Slowdown |
| Persistent inflation | 0.5, 2, 0.5, 2 | Reflation, Stagflation |
| Stagflation stress | 0.5, 0.5, 1, 4 | Stagflation |
| Severe financial stress | 0.25, 0.25, 3, 3 | Slowdown, Stagflation |

These are transparent hypothetical scenarios, not calibrated crisis probabilities.
The UI displays the modified transition matrix and simulated occupancy. A
four-state macro proxy cannot capture every crisis mechanism.

**Include estimation uncertainty** forms up to 32 moving-block-resampled parameter
populations shared by path groups. Parametric means are perturbed around the
supplied model estimate, and covariance is re-estimated with shrinkage. Empirical
methods resample their historical populations. This captures some finite-sample
uncertainty; it is not a complete Bayesian posterior or a treatment of all model risk.

### Portfolio life, cash flows and outputs

Initial holdings match the selected portfolio. Between scheduled rebalances,
holdings drift. Trading costs apply to actual two-sided traded notional. Model
months/quarters use 21/63 observations with the default 252-session year. At each
model year-end, after market return/rebalancing, the contribution is added and then
the withdrawal is taken. Contributions grow at the supplied contribution-growth
rate; withdrawals grow with inflation. Flows enter/leave proportionately and do
not create a free rebalance. Unfunded withdrawals are reported, and wealth cannot
be negative. No borrowing or tax-lot accounting is assumed.

Parametric simple-return draws below −100% are floored at −100%; their count is
reported because this changes the far tail. Drawdowns use unitized investment
performance including trading costs and excluding contributions/withdrawals.
Nominal terminal wealth probabilities include external flows. Additional real
percentile paths divide by assumed cumulative inflation; goals remain nominal.

Outputs include mean/median terminal wealth; 5/25/75/95th percentiles; loss,
doubling and goal probabilities; mean/median maximum drawdown; the worst 5% drawdown
quantile; probabilities of drawdowns beyond 30% and 50%; and 5/25/50/75/95th fan
paths. UI/session state retains these summaries and one terminal value per path,
not every simulated path. No simulation fraction is a guarantee or a calibrated
probability of the future.

Historical stress replays cover the dot-com bust, global financial crisis, COVID
crash and 2022 rate/inflation shock when complete asset history exists. Replays
are buy-and-hold and report return/drawdown. Missing coverage is identified rather
than filled using another ticker. They exclude contributions, withdrawals and costs.

### Recommendation diagnostics

The Robustness page combines central weights, bootstrap median and 10th/90th
percentiles, stability, optimizer agreement, strategic target, effective policy
range, annual regime-view adjustment and reasons. Conviction is HIGH at ≥0.75,
MEDIUM at ≥0.40, otherwise LOW, using the minimum of existing stability, agreement
and overall recommendation confidence. It is an evidence classification, not the
probability an asset will outperform. The recommendation algorithm is preserved.

### Dashboard navigation and saved state

Home is the default landing page and the only portfolio editor. Apply Portfolio &
Run Analysis and Refresh Market Data sit above the editor; the saved executive
summary appears below it. Collapse the editor to focus on results. The horizontal
native Streamlit radio menu wraps on small screens and keeps all existing analytical
pages accessible without a permanent sidebar or additional navigation dependencies.

Draft table edits and settings survive page changes. The last successful analysis
stores a separate applied configuration; editing the draft only marks results as
outdated. Applying validates and runs the existing pipeline once. A failed run
retains previous results. Backtest and simulation settings/results also persist
across navigation. State is session-local; it is not durable account storage.
