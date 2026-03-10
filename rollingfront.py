import numpy as np
import pandas as pd
from mpt import annualize_mean_cov


def rolling_statistics(returns, window, trading_days):

    rolling_mu = []
    rolling_vol = []
    dates = []

    for i in range(window, len(returns)):

        sample = returns.iloc[i-window:i]

        mu, cov = annualize_mean_cov(sample, trading_days)

        port_mu = mu.mean()
        port_vol = np.sqrt(np.mean(np.diag(cov)))

        rolling_mu.append(port_mu)
        rolling_vol.append(port_vol)
        dates.append(returns.index[i])

    return pd.DataFrame({
        "return": rolling_mu,
        "vol": rolling_vol
    }, index=dates)