from __future__ import annotations
import io
import zipfile
import requests
import pandas as pd
import statsmodels.api as sm


FF3_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


def _load_ken_french_csv(url: str, expected_columns: list[str]) -> pd.DataFrame:
    resp = requests.get(url, timeout=30)
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
            continue

        if start_idx is not None:
            if not line:
                end_idx = i
                break
            first = line.split(",")[0].strip()
            if not first.isdigit():
                end_idx = i
                break

    if start_idx is None:
        raise ValueError("Could not locate Ken French daily factor table.")

    data_lines = lines[start_idx:end_idx] if end_idx is not None else lines[start_idx:]

    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < len(expected_columns) + 1:
            continue
        if not parts[0].isdigit():
            continue
        rows.append(parts[: len(expected_columns) + 1])

    df = pd.DataFrame(rows, columns=["date"] + expected_columns)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df.set_index("date").astype(float) / 100.0
    return df


def load_ff3_daily() -> pd.DataFrame:
    return _load_ken_french_csv(
        FF3_URL,
        ["Mkt-RF", "SMB", "HML", "RF"],
    )


def load_ff5_daily() -> pd.DataFrame:
    return _load_ken_french_csv(
        FF5_URL,
        ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"],
    )


def fama_french_regression(
    portfolio_returns: pd.Series,
    model: str = "ff5",
):
    """
    Run Fama-French factor regression on daily portfolio returns.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns with DatetimeIndex.
    model : str
        "ff3" or "ff5"
    """
    y = portfolio_returns.copy().dropna()
    y.index = pd.to_datetime(y.index)
    y.name = "portfolio"

    model = model.lower().strip()

    if model == "ff3":
        factors = load_ff3_daily()
        factor_cols = ["Mkt-RF", "SMB", "HML"]
    elif model == "ff5":
        factors = load_ff5_daily()
        factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    else:
        raise ValueError("model must be 'ff3' or 'ff5'")

    df = pd.concat([y, factors], axis=1, join="inner").dropna()

    excess_portfolio = df["portfolio"] - df["RF"]
    X = sm.add_constant(df[factor_cols])

    fitted = sm.OLS(excess_portfolio, X).fit()
    return fitted


def factor_summary_table(model_fit) -> pd.DataFrame:
    out = pd.DataFrame({
        "coef": model_fit.params,
        "t_stat": model_fit.tvalues,
        "p_value": model_fit.pvalues,
    })
    return out