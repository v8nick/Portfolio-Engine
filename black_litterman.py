from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pypfopt.black_litterman import BlackLittermanModel


def resolve_strategic_prior_weights(tickers: list[str], weights: dict[str, float] | None) -> np.ndarray:
    """Explicit strategic prior, or labeled equal-weight modeling fallback (not market caps)."""
    if weights is None:
        warnings.warn("BL strategic prior unspecified: using equal-weight modeling fallback; not market equilibrium weights.", stacklevel=2)
        return np.repeat(1 / len(tickers), len(tickers))
    if set(weights) != set(tickers):
        raise ValueError("Strategic prior must contain exactly the portfolio tickers.")
    result = np.array([weights[t] for t in tickers], dtype=float)
    if not np.isfinite(result).all() or (result < 0).any() or not np.isclose(result.sum(), 1):
        raise ValueError("Strategic prior weights must be finite, nonnegative, and sum to one.")
    return result


def implied_equilibrium_returns(cov: pd.DataFrame, market_weights: np.ndarray,
                                risk_aversion: float, risk_free_rate: float = 0.0) -> pd.Series:
    """Annual total-return prior: rf + delta * Sigma * strategic weights."""
    weights = np.asarray(market_weights, dtype=float)
    if weights.shape != (len(cov),) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("BL prior weights must be aligned, nonnegative, finite, and sum to one.")
    return pd.Series(risk_free_rate + risk_aversion * (cov.values @ weights), index=cov.index, name="pi")


def build_absolute_view_dict(absolute_views: dict[str, tuple[float, float]] | None = None) -> dict[str, float]:
    # Retained for callers; posterior construction below also retains confidence.
    return {asset: exp_ret for asset, (exp_ret, _conf) in (absolute_views or {}).items()}


def black_litterman_posterior(
    cov: pd.DataFrame, market_weights: np.ndarray, risk_aversion: float, tau: float,
    absolute_views: dict[str, tuple[float, float]] | None = None,
    relative_views: list[tuple[str, str, float, float]] | None = None,
    risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Annual total absolute returns / annual relative return differences.

    Confidence c in [0,1] uses PyPortfolioOpt Idzorek Omega:
    tau * (1-c)/c * P Sigma P'. Zero-confidence views are omitted exactly.
    Relative tuple: (outperformer, underperformer, return_difference, confidence).
    """
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion, risk_free_rate)
    assets = list(cov.index)
    rows, values, confidences = [], [], []
    views = [([asset], [1.0], ret, conf) for asset, (ret, conf) in (absolute_views or {}).items()]
    views += [([a, b], [1.0, -1.0], ret, conf) for a, b, ret, conf in (relative_views or [])]
    for names, coefficients, ret, conf in views:
        if not np.isfinite([ret, conf]).all() or not 0 <= conf <= 1:
            raise ValueError("View return must be finite and confidence must be in [0,1].")
        if any(name not in assets for name in names) or len(set(names)) != len(names):
            raise ValueError("Views must reference known, distinct assets.")
        if conf == 0:
            continue
        row = np.zeros(len(assets))
        for name, coefficient in zip(names, coefficients):
            row[assets.index(name)] = coefficient
        rows.append(row)
        values.append(ret)
        confidences.append(conf)
    if not rows:
        return pi.copy().rename("bl_mu"), pi
    bl = BlackLittermanModel(cov_matrix=cov, pi=pi, P=np.array(rows), Q=np.array(values),
                            tau=tau, omega="idzorek", view_confidences=np.array(confidences),
                            risk_aversion=risk_aversion)
    return bl.bl_returns().rename("bl_mu"), pi


def merge_manual_regime_views(manual_views: dict, regime_views: dict) -> tuple[dict, dict]:
    """Manual fundamental return plus a lower-confidence tactical adjustment.

    For shared assets, tactical delta is weighted by Idzorek precision c/(1-c)
    relative to manual precision. Keep max input confidence, rather than adding
    confidence from correlated sources. A 100% manual view cannot be moved.
    This is a single auditable absolute view per asset; relative views remain
    separate and unchanged in black_litterman_posterior().
    """
    merged, metadata = {}, {}
    for asset in dict.fromkeys(list(manual_views) + list(regime_views)):
        manual = manual_views.get(asset)
        regime = regime_views.get(asset)
        if manual is not None and (not np.isfinite(manual).all() or not 0 <= manual[1] <= 1):
            raise ValueError('Manual returns must be finite and confidence in [0,1].')
        active = regime is not None and regime['confidence'] > 0 and regime['adjustment'] != 0
        if manual is not None and active:
            m_return, m_conf = manual
            r_conf = regime['confidence']
            if m_conf == 1:
                fraction = 0.
            else:
                m_precision = m_conf / (1 - m_conf)
                r_precision = r_conf / (1 - r_conf)
                fraction = r_precision / (m_precision + r_precision)
            base = m_return if m_conf > 0 else regime['prior_expected_return']
            merged[asset] = (base + fraction * regime['adjustment'], max(m_conf, r_conf))
            source = 'combined'
        elif manual is not None:
            merged[asset] = tuple(manual)
            source, fraction = 'manual', 0.
        elif active:
            merged[asset] = (regime['adjusted_bl_view'], regime['confidence'])
            source, fraction = 'regime', 1.
        else:
            continue
        metadata[asset] = {'source': source, 'manual': manual, 'regime': regime,
                           'tactical_precision_fraction': fraction, 'merged_view': merged[asset]}
    return merged, metadata
