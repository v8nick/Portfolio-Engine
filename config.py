# config.py

TICKERS = [
    "QQQ", "MSFT", "NVDA", "AAPL", "AVGO",
    "AMZN", "PLTR", "ARKG", "IWM", "LMT", "BA"
]

BENCHMARK = "SPY"

BASE_WEIGHTS = {
    "QQQ": 0.34,
    "MSFT": 0.09,
    "NVDA": 0.09,
    "AAPL": 0.07,
    "AVGO": 0.07,
    "AMZN": 0.07,
    "PLTR": 0.05,
    "ARKG": 0.05,
    "IWM": 0.06,
    "LMT": 0.06,
    "BA": 0.05,
}

START_DATE = "2018-01-01"
END_DATE = None  

TRADING_DAYS = 252
RISK_FREE_RATE = 0.04  # 4% placeholder; replace later with live Treasury yield

ALLOW_SHORTS = False
WEIGHT_BOUNDS = (0.0, 0.40)  # no position bigger than 40%

#monte carlo simulation parameters
MC_YEARS = 10
MC_N_SIMS = 10000
MC_INITIAL_VALUE = 1.0
MC_SEED = 42
# Black-Litterman settings

USE_BLACK_LITTERMAN = True

# Use your base portfolio as the market/prior portfolio for now
BL_MARKET_WEIGHTS = BASE_WEIGHTS.copy()

# Risk aversion parameter (delta)
BL_RISK_AVERSION = 2.5

# Uncertainty scaling for prior
BL_TAU = 0.05

# Absolute views: asset -> (expected annual return, confidence)
# confidence in (0, 1], where higher means stronger confidence
BL_ABSOLUTE_VIEWS = {
    "QQQ": (0.14, 0.70),
    "MSFT": (0.15, 0.75),
    "NVDA": (0.18, 0.55),
    "AVGO": (0.16, 0.60),
    "AMZN": (0.14, 0.55),
    "PLTR": (0.17, 0.35),
    "ARKG": (0.13, 0.25),
    "IWM": (0.11, 0.45),
    "LMT": (0.10, 0.70),
    "BA": (0.12, 0.30),
}
BL_RELATIVE_VIEWS = [
    ("QQQ", "IWM", 0.03, 0.75),   # QQQ outperform IWM by 3%
    ("QQQ", "LMT", 0.04, 0.70),   # QQQ outperform LMT by 4%
    ("NVDA", "QQQ", 0.03, 0.55),  # NVDA outperform QQQ by 3%
]
FIXED_WEIGHTS = {
    "QQQ": 0.35,
}

MIN_WEIGHTS = {
    "IWM": 0.05,
    "LMT": 0.05,
}
USE_PYPORTFOLIOOPT = True
GENERATE_QUANTSTATS_REPORT = True
QUANTSTATS_OUTPUT = "quantstats_report.html"
USE_COV_SHRINKAGE = True
COV_SHRINKAGE_METHOD = "ledoit_wolf"
# =========================
# IMPLEMENTATION LAYER
# =========================
IMPLEMENTATION_LAYER = False

TURNOVER_PENALTY_LAMBDA = 0.50
TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5

APPLY_TAX_EFFECTS = True
SHORT_TERM_TAX_RATE = 0.35
LONG_TERM_TAX_RATE = 0.15
LONG_TERM_FRACTION = 0.50

# Simple placeholder assumption:
# each position has 20% embedded unrealized gain
DEFAULT_UNREALIZED_GAIN_RATE = 0.20
# =============================
# COVARIANCE SETTINGS
# =============================

USE_COV_SHRINKAGE = True
COV_SHRINKAGE_METHOD = "ledoit_wolf"
