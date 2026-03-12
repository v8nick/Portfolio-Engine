TICKERS = [
    "QQQ",
    "MSFT",
    "NVDA",
    "AAPL",
    "AVGO",
    "AMZN",
    "PLTR",
    "ARKG",
    "IWM",
    "LMT",
    "BA",
]

BENCHMARK = "SPY"

STARTING_WEIGHTS = {
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

WEIGHT_BOUNDS = (0.0, 0.40)

FIXED_WEIGHTS = {
    "QQQ": 0.35,
}

MIN_WEIGHTS = {
    "IWM": 0.05,
    "LMT": 0.05,
}

BL_MARKET_WEIGHTS = STARTING_WEIGHTS.copy()

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

# Placeholder only. Relative views are not implemented in black_litterman.py yet.
BL_RELATIVE_VIEWS = [
    ("QQQ", "IWM", 0.03, 0.75),
    ("QQQ", "LMT", 0.04, 0.70),
    ("NVDA", "QQQ", 0.03, 0.55),
]

IMPLEMENTATION_LAYER = False
TURNOVER_PENALTY_LAMBDA = 0.50

MC_YEARS = 10
MC_N_SIMS = 10000
MC_INITIAL_VALUE = 1.0
MC_SEED = 42

RUN_WALKFORWARD_BACKTEST = True
WF_WINDOW_DAYS = 252 * 3
WF_REBALANCE_FREQ = "ME"

GENERATE_QUANTSTATS_REPORT = True
QUANTSTATS_OUTPUT = "outputs/quantstats_report.html"
WALKFORWARD_QUANTSTATS_OUTPUT = "outputs/quantstats_walkforward_report.html"
