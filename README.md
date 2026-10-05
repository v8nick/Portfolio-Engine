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

To import a portfolio, expand **Import portfolio CSV** above the sidebar table,
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

1. In the sidebar, enter your **portfolio value** and edit the portfolio table.
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
5. Click **Run Analysis** and wait for completion. For a quicker initial check,
   reduce bootstrap repetitions from 100 to 20. This changes how many sensitivity
   checks run, not the underlying financial methodology.
6. Navigate between pages to inspect the saved result:

| Page | What it shows |
| --- | --- |
| Executive overview | Underlying growth and inflation/rate-pressure metrics, allocation comparison, risk metrics and actions |
| Market intelligence | Treasury yields, curve, cross-asset signals and backend interpretations |
| Macro indicators | Actual model input changes, score construction, data coverage and historical returns by market pattern |
| Portfolio construction | Strategic, minimum variance, BL max Sharpe, risk parity, CVaR and recommended allocations |
| Risk | Tail risk, concentration, asset risk contributions and configured group exposures |
| Robustness | BL max-Sharpe bootstrap weight stability |
| Rebalance | Backend action labels, required policy repairs, aggregate cost and approximate tax estimates |

Page navigation, chart selections and input edits do not rerun the portfolio
engine. Click **Run Analysis** to apply edited inputs. **Refresh Market Data**
requests fresh provider data; it leaves the saved portfolio recommendation intact
until you run analysis again. A failed run retains the last successful result.

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

Sidebar edits affect the current session; they do not rewrite `config/` files.
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
- Research walk-forward weights first apply after the decision observation;
  outputs remain gross constant-weight approximations. The Monte Carlo baseline
  uses IID multivariate Gaussian simple returns and constant moments/weights;
  it excludes costs, taxes, regime changes and realistic tail behavior.
- Live provider availability is not guaranteed. No broker integration, account
  connection, authentication or trade execution is included.

Nicholas Clervi · Economics / Quantitative Finance Research
