from __future__ import annotations
import numpy as np
import pandas as pd

from pypfopt.black_litterman import BlackLittermanModel


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> pd.Series:
    pi = risk_aversion * (cov.values @ market_weights)
    return pd.Series(pi, index=cov.index, name="pi")


def build_absolute_view_dict(
    absolute_views: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    absolute_views = absolute_views or {}
    return {asset: exp_ret for asset, (exp_ret, _conf) in absolute_views.items()}


def black_litterman_posterior(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float,
    tau: float,
    absolute_views: dict[str, tuple[float, float]] | None = None,
    relative_views: list[tuple[str, str, float, float]] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """
    PyPortfolioOpt-backed BL posterior.
    For now, only absolute views are passed into the library.
    Relative views can be added later with explicit P/Q matrices.
    """
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)

    abs_view_dict = build_absolute_view_dict(absolute_views)

    if not abs_view_dict:
        return pi.copy().rename("bl_mu"), pi

    bl = BlackLittermanModel(
        cov_matrix=cov,
        pi=pi,
        absolute_views=abs_view_dict,
        tau=tau,
    )

    posterior_mu = bl.bl_returns()
    posterior_mu.name = "bl_mu"

    return posterior_mu, pi