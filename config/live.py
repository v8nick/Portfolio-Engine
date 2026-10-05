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

# Live holdings are still modeled as target weights, not shares, cash, or tax lots.
CURRENT_WEIGHTS = {
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

LOOKBACK_START_DATE = "2018-01-01"
LOOKBACK_END_DATE = None

WEIGHT_BOUNDS = (0.0, 0.40)

FIXED_WEIGHTS = {
    "QQQ": 0.35,
}

MIN_WEIGHTS = {
    "IWM": 0.05,
    "LMT": 0.05,
}

BL_STRATEGIC_PRIOR_WEIGHTS = None  # Supply strategic weights; None uses labeled equal weights.
BL_MARKET_WEIGHTS = BL_STRATEGIC_PRIOR_WEIGHTS  # Legacy alias.

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

# (outperformer, underperformer, annual return difference, confidence in [0,1]).
BL_RELATIVE_VIEWS = [
    ("QQQ", "IWM", 0.03, 0.75),
    ("QQQ", "LMT", 0.04, 0.70),
    ("NVDA", "QQQ", 0.03, 0.55),
]

IMPLEMENTATION_LAYER = False
TURNOVER_PENALTY_LAMBDA = 0.50
TRADE_ACTION_THRESHOLD = 0.005

# Phase 3 illustrative core policy, independent of CURRENT_WEIGHTS. Customize
# these strategic assumptions before using recommendations for a real mandate.
STRATEGIC_ALLOCATION = {
    'SPY': {'strategic_weight': .30, 'minimum_weight': .15, 'maximum_weight': .45, 'tactical_low': .25, 'tactical_high': .35, 'group': 'us_equity', 'role': 'core'},
    'IWM': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'us_equity', 'role': 'core'},
    'VEA': {'strategic_weight': .10, 'minimum_weight': 0., 'maximum_weight': .20, 'tactical_low': .07, 'tactical_high': .13, 'group': 'international_equity', 'role': 'core'},
    'VWO': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'international_equity', 'role': 'core'},
    'IEF': {'strategic_weight': .15, 'minimum_weight': .05, 'maximum_weight': .30, 'tactical_low': .10, 'tactical_high': .20, 'group': 'treasuries', 'role': 'core'},
    'TIP': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'inflation_linked', 'role': 'core'},
    'LQD': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'credit', 'role': 'core'},
    'BIL': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .20, 'tactical_low': .02, 'tactical_high': .10, 'group': 'cash', 'role': 'core'},
    'VNQ': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'real_estate', 'role': 'core'},
    'GLD': {'strategic_weight': .05, 'minimum_weight': 0., 'maximum_weight': .15, 'tactical_low': .02, 'tactical_high': .08, 'group': 'gold', 'role': 'core'},
}
MAX_SATELLITE_ALLOCATION = .15
MAX_INDIVIDUAL_SATELLITE_WEIGHT = .05
# Existing growth/thematic funds and stocks stay supported as satellites.
SATELLITE_GROUPS = {ticker: 'us_equity' for ticker in TICKERS if ticker not in STRATEGIC_ALLOCATION}
