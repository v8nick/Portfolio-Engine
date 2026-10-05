TRADING_DAYS = 252
RISK_FREE_RATE = None  # Annual decimal override; None resolves the 3-month Treasury yield.
RISK_FREE_FALLBACK = 0.0  # Labeled modeling assumption when yields are unavailable.
RISK_FREE_MAX_AGE_DAYS = 7

MARKET_ZSCORE_WINDOW = 63
MARKET_CACHE_TTL_SECONDS = 3600

USE_COV_SHRINKAGE = True
COV_SHRINKAGE_METHOD = "ledoit_wolf"

USE_BLACK_LITTERMAN = True
BL_RISK_AVERSION = 2.5
BL_TAU = 0.05

TRANSACTION_COST_BPS = 10
SLIPPAGE_BPS = 5

APPLY_TAX_EFFECTS = True
SHORT_TERM_TAX_RATE = 0.35
LONG_TERM_TAX_RATE = 0.15
LONG_TERM_FRACTION = 0.50
DEFAULT_UNREALIZED_GAIN_RATE = 0.20

# Phase 2: transparent market-proxy regimes; no return-based parameter fitting.
REGIME_HORIZONS = (21, 63)
REGIME_ZSCORE_WINDOW = 252 * 3
REGIME_ZSCORE_MIN_PERIODS = 252
REGIME_ZSCORE_CLIP = 3.0
REGIME_SMOOTHING_SPAN = 10
REGIME_TEMPERATURE = 1.0
REGIME_MIN_COMPONENTS = 2
REGIME_MAX_AGE_DAYS = 7
REGIME_YIELD_LAG_DAYS = 1  # Conservative calendar-day availability lag; not vintage data.
REGIME_CONDITIONS_THRESHOLD = 0.5  # Positive=tighter; negative=easier.
REGIME_STRESS_THRESHOLDS = (-0.5, 0.5, 1.5)  # low / normal / elevated / severe.
REGIME_SHRINKAGE_STRENGTH = 20.0  # Pseudo-count of non-overlapping completed outcomes.

# Phase 3 static decision settings. No performance-based tuning.
DECISION_SETTINGS = {
    'regime_horizon': 63,
    'minimum_regime_samples': 5,
    'sample_confidence_target': 12,
    'regime_bl_strength': .5,
    'max_annual_regime_adjustment': .025,
    'max_regime_view_confidence': .25,
    'regime_return_dispersion_scale': .10,
    'covariance_lookback': 252 * 3,
    'minimum_history': 126,
    'robustness_repetitions': 100,
    'bootstrap_block_days': 21,
    'bootstrap_mean_scale': .25,
    'seed': 42,
    'tail_confidence': .95,
    'enable_cvar': True,
    'tactical_blend': .5,
    'tight_conditions_tilt_scale': .75,
    'elevated_stress_tilt_scale': .5,
    'severe_stress_tilt_scale': .25,
    'rebalance_threshold': .01,
    'minimum_trade_confidence': .40,
    'cost_benefit_horizon_years': 3.,
    'agreement_dispersion_scale': .025,
    'agreement_high': .75,
    'agreement_moderate': .40,
    'confidence_high': .75,
    'confidence_moderate': .40,
    'regime_model_options': {},
}
