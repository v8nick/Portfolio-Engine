# Portfolio Engine

A quantitative portfolio research engine for **long-horizon asset allocation, portfolio optimization, and risk simulation**.

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

The model is designed to allow **rapid asset rotation** by editing the live or research configuration and re-running the relevant entrypoint.

---

# Requirements

- Python 3.10+
- Internet connection (for downloading market data)

---

# Usage

## Windows

Double-click:

```
run.bat
```

This will automatically:

- create a virtual environment  
- install dependencies  
- run the portfolio engine  

---

## Mac / Linux

Open Terminal in the project folder and run:

```bash
chmod +x run.command
./run.command
```

This will:

- create a virtual environment  
- install dependencies  
- run the portfolio engine  

---

## Manual Run

If you prefer running manually:

```bash
python main.py
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

Computes optimal allocations using **Modern Portfolio Theory**.

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

to produce **more stable expected return estimates** than historical averages alone.

---

## Long Horizon Risk Simulation

Uses **multivariate correlated Monte Carlo simulations** to estimate:

- terminal wealth distributions  
- drawdown probabilities  
- probability of portfolio loss  
- long-term compounding outcomes  

---

## Portfolio Diagnostics

The engine generates a **decision dashboard** including:

- Current Portfolio Performance  
- Optimized Portfolio  
- Suggested Weight Changes  
- Risk Contribution by Asset  
- Monte Carlo Risk Profile  

---

# Example Outputs

## Efficient Frontier

Shows the optimal tradeoff between **expected return and volatility**.

Key points plotted:

- Starting portfolio  
- Maximum Sharpe portfolio  
- Minimum volatility portfolio  
- configured benchmark  

---

## Rolling Market Environment

Uses a **rolling 5-year window** to show:

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

Run the live rebalance engine with:

```
python main_live.py
```

Or use the launcher:

- Windows: `run_live.bat`
- macOS: `run_live.command`

Run the research engine with:

```
python main_research.py
```

Or use the launcher:

- Windows: `run_research.bat`
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

# Model Purpose

This project is designed as a **portfolio research framework** that can be reused whenever assets are rotated or portfolio assumptions change.

The goal is to provide a **repeatable and transparent method for long-term portfolio construction and evaluation**.

---

# Author

**Nicholas Clervi**  
United States Air Force Academy  
Economics / Quantitative Finance Research
