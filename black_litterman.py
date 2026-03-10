from __future__ import annotations
import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float,
) -> pd.Series:
    pi = risk_aversion * (cov.values @ market_weights)
    return pd.Series(pi, index=cov.index, name="pi")


def build_view_matrices(
    assets: list[str],
    absolute_views: dict[str, tuple[float, float]] | None = None,
    relative_views: list[tuple[str, str, float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    absolute_views:
        {"QQQ": (0.14, 0.70)}

    relative_views:
        [
            ("QQQ", "IWM", 0.03, 0.70),   # QQQ expected to outperform IWM by 3%
            ("NVDA", "QQQ", 0.04, 0.55),  # NVDA expected to outperform QQQ by 4%
        ]
    """
    absolute_views = absolute_views or {}
    relative_views = relative_views or []

    rows = []
    q_vals = []
    omega_diag = []

    n_assets = len(assets)

    for asset, (expected_return, confidence) in absolute_views.items():
        if asset not in assets:
            raise ValueError(f"Absolute view asset '{asset}' not found.")

        if not (0 < confidence <= 1):
            raise ValueError(f"Confidence for {asset} must be in (0, 1].")

        row = np.zeros(n_assets)
        row[assets.index(asset)] = 1.0
        rows.append(row)
        q_vals.append(expected_return)
        omega_diag.append((1.0 - confidence) / confidence)

    for asset_long, asset_short, spread, confidence in relative_views:
        if asset_long not in assets or asset_short not in assets:
            raise ValueError(f"Relative view assets '{asset_long}' or '{asset_short}' not found.")

        if not (0 < confidence <= 1):
            raise ValueError(f"Confidence for relative view ({asset_long}, {asset_short}) must be in (0, 1].")

        row = np.zeros(n_assets)
        row[assets.index(asset_long)] = 1.0
        row[assets.index(asset_short)] = -1.0
        rows.append(row)
        q_vals.append(spread)
        omega_diag.append((1.0 - confidence) / confidence)

    if len(rows) == 0:
        return np.empty((0, n_assets)), np.empty((0,)), np.empty((0, 0))

    P = np.vstack(rows)
    Q = np.array(q_vals, dtype=float)
    Omega = np.diag(np.array(omega_diag, dtype=float))
    return P, Q, Omega


def black_litterman_posterior(
    cov: pd.DataFrame,
    market_weights: np.ndarray,
    risk_aversion: float,
    tau: float,
    absolute_views: dict[str, tuple[float, float]] | None = None,
    relative_views: list[tuple[str, str, float, float]] | None = None,
) -> tuple[pd.Series, pd.Series]:
    assets = list(cov.index)
    pi = implied_equilibrium_returns(cov, market_weights, risk_aversion)

    P, Q, Omega = build_view_matrices(
        assets=assets,
        absolute_views=absolute_views,
        relative_views=relative_views,
    )

    if len(Q) == 0:
        return pi.copy(), pi

    sigma = cov.values
    tau_sigma = tau * sigma

    middle = np.linalg.inv(
        np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
    )
    rhs = np.linalg.inv(tau_sigma) @ pi.values + P.T @ np.linalg.inv(Omega) @ Q
    posterior = middle @ rhs

    posterior_mu = pd.Series(posterior, index=assets, name="bl_mu")
    return posterior_mu, pi