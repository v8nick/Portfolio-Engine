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
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices.reindex(columns=tickers)
    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert price series to simple daily returns.
    """
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    return returns

# FRED yields are percentages per annum; price returns are decimal fractions.
TREASURY_SERIES = {"3MO": "DGS3MO", "2Y": "DGS2", "5Y": "DGS5", "10Y": "DGS10", "30Y": "DGS30"}


def download_treasury_yields(start: str, end: str | None = None) -> pd.DataFrame:
    """Free, keyless FRED CSV adapter. Missing maturities remain missing."""
    import io
    import warnings
    import requests
    try:
        response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                                params={"id": ",".join(TREASURY_SERIES.values()), "cosd": start,
                                        "coed": end or pd.Timestamp.now().date().isoformat()}, timeout=20)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), index_col=0, parse_dates=True)
        frame.index = pd.to_datetime(frame.index, errors="raise")
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame = frame.rename(columns={v: k for k, v in TREASURY_SERIES.items()})
        return frame.reindex(columns=TREASURY_SERIES).loc[start:end].dropna(how="all")
    except (requests.RequestException, ValueError, KeyError) as exc:
        warnings.warn(f"Treasury data unavailable: {exc}", stacklevel=2)
        return pd.DataFrame(columns=TREASURY_SERIES, index=pd.DatetimeIndex([]), dtype=float)


def resolve_risk_free_rate(as_of=None, yields: pd.DataFrame | None = None,
                           override: float | None = None) -> dict:
    """Central annual decimal RF interface, with source/date and bounded staleness.

    Uses 3-month Treasury yield, never an ETF price. Fallback is a configurable
    modeling assumption. A supplied frame prevents additional network requests.
    """
    import numpy as np
    from config.shared import RISK_FREE_RATE, RISK_FREE_FALLBACK, RISK_FREE_MAX_AGE_DAYS
    override = RISK_FREE_RATE if override is None else override
    date = pd.Timestamp(as_of or pd.Timestamp.now().normalize()).tz_localize(None)
    if override is not None:
        if not np.isfinite(override):
            raise ValueError("Risk-free override must be finite.")
        return {"rate": float(override), "source": "configuration override", "as_of": None}
    if yields is None:
        yields = download_treasury_yields((date - pd.Timedelta(days=30)).date().isoformat(), date.date().isoformat())
    series = pd.to_numeric(yields.get("3MO", pd.Series(dtype=float)), errors="coerce").copy()
    series.index = pd.to_datetime(series.index).tz_localize(None)
    series = series.loc[series.index <= date].replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if not series.empty and (date - series.index[-1]).days <= RISK_FREE_MAX_AGE_DAYS:
        return {"rate": float(series.iloc[-1]) / 100, "source": "FRED DGS3MO", "as_of": series.index[-1].isoformat()}
    return {"rate": float(RISK_FREE_FALLBACK), "source": "configured fallback assumption (Treasury unavailable/stale)", "as_of": None}


def get_risk_free_rate(**kwargs) -> float:
    """Scalar convenience wrapper; provenance available via resolve_risk_free_rate."""
    return resolve_risk_free_rate(**kwargs)["rate"]
