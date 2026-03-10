# frontier.py

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def efficient_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    n_points: int = 50,
    bounds=(0.0, 1.0),
):
    """
    Compute efficient frontier portfolios.
    Returns volatilities and expected returns along the frontier.
    """

    n_assets = len(mu)
    results_vol = []
    results_ret = []
    weights_list = []

    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    for target in target_returns:

        def portfolio_vol(weights):
            return np.sqrt(weights @ cov.values @ weights)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w: w @ mu.values - target},
        ]

        bounds_list = [bounds] * n_assets
        x0 = np.repeat(1 / n_assets, n_assets)

        result = minimize(
            portfolio_vol,
            x0,
            method="SLSQP",
            bounds=bounds_list,
            constraints=constraints,
        )

        if result.success:
            weights = result.x
            vol = portfolio_vol(weights)

            results_vol.append(vol)
            results_ret.append(target)
            weights_list.append(weights)

    return np.array(results_vol), np.array(results_ret), weights_list