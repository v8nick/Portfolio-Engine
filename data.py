# data.py

from __future__ import annotations
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """
    Download adjusted close prices for a list of tickers.
    """
    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if "Close" not in df:
        raise ValueError("Could not find 'Close' prices in downloaded data.")

    prices = df["Close"].copy()
    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price series to simple daily returns.
    """
    returns = prices.pct_change().dropna(how="all")
    return returns