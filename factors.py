from __future__ import annotations
import io
import zipfile
import requests
import pandas as pd
import statsmodels.api as sm


FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"


def _load_ff3_daily() -> pd.DataFrame:
    resp = requests.get(FF3_URL, timeout=30)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = zf.namelist()[0]
        raw = zf.read(name).decode("utf-8", errors="ignore")

    lines = raw.splitlines()

    start_idx = None
    end_idx = None

    for i, line in enumerate(lines):
        line = line.strip()
        if start_idx is None and line and line[:8].isdigit():
            start_idx = i
        elif start_idx is not None and (not line or line.startswith(" ")):
            end_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not locate Fama-French daily factor table.")

    data_lines = lines[start_idx:end_idx] if end_idx is not None else lines[start_idx:]

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        if not parts[0].isdigit():
            continue
        rows.append(parts[:5])

    ff = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    ff["date"] = pd.to_datetime(ff["date"], format="%Y%m%d")
    ff = ff.set_index("date").astype(float) / 100.0
    return ff


def fama_french_regression(portfolio_returns: pd.Series):
    """
    Run Fama-French 3-factor regression on daily portfolio returns.
    portfolio_returns must be a pd.Series with a DatetimeIndex.
    """
    ff = _load_ff3_daily()

    y = portfolio_returns.copy().dropna()
    y.index = pd.to_datetime(y.index)
    y.name = "portfolio"

    df = pd.concat([y, ff], axis=1, join="inner").dropna()

    excess_portfolio = df["portfolio"] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])

    model = sm.OLS(excess_portfolio, X).fit()
    return model