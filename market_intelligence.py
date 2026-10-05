"""Cross-asset observations and cautious rules; no causal or regime model.

Price horizons use adjusted-price total returns; relative signals are differences
of returns, not changes of price ratios. Yield inputs are annual percent units.
Horizons count available joint observations, not calendar days. Nothing is filled.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from config.shared import MARKET_CACHE_TTL_SECONDS, MARKET_ZSCORE_WINDOW
from data import download_prices, download_treasury_yields, TREASURY_SERIES

HORIZONS = {"1D": 1, "5D": 5, "21D": 21, "63D": 63, "252D": 252}
MARKET_SYMBOLS = {name: name for name in (
    "SPY", "QQQ", "RSP", "IWM", "SMH", "XLY", "XLP", "TLT", "IEF", "SGOV", "BIL",
    "LQD", "HYG", "TIP", "GLD", "DBC", "VNQ")}
MARKET_SYMBOLS.update({"DXY": "DX-Y.NYB", "VIX": "^VIX"})
RELATIVE_PAIRS = {"QQQ_SPY": ("QQQ", "SPY"), "RSP_SPY": ("RSP", "SPY"),
                  "IWM_SPY": ("IWM", "SPY"), "SMH_QQQ": ("SMH", "QQQ"),
                  "XLY_XLP": ("XLY", "XLP"), "HYG_LQD": ("HYG", "LQD"),
                  "TIP_IEF": ("TIP", "IEF")}


def load_market_data(start: str, end: str | None = None,
                     cache_dir: str | Path = "outputs/market_cache",
                     ttl_seconds: int = MARKET_CACHE_TTL_SECONDS) -> dict:
    """Cache successful source downloads as CSV; partial outages never fail snapshot.

    FRED observations may be released after their observation date. This adapter
    is for descriptive analysis, not publication-time/vintage-safe backtesting.
    """
    cache = Path(cache_dir)
    key = hashlib.sha256(f"v1|{start}|{end}".encode()).hexdigest()[:16]
    messages, frames, cached = [], {}, {}
    sources = {"prices": lambda: download_prices(list(MARKET_SYMBOLS.values()), start, end),
               "yields": lambda: download_treasury_yields(start, end)}
    for name, loader in sources.items():
        path = cache / f"{name}_{key}.csv"
        frame, cached[name] = pd.DataFrame(), False
        try:
            if path.exists() and pd.Timestamp.now().timestamp() - path.stat().st_mtime < ttl_seconds:
                frame = pd.read_csv(path, index_col=0, parse_dates=True)
                cached[name] = True
            else:
                frame = loader()
                if not frame.empty:
                    try:
                        cache.mkdir(parents=True, exist_ok=True)
                        frame.to_csv(path)
                    except OSError as exc:
                        messages.append(f"Cache write unavailable: {exc}")
        except Exception as exc:
            # Provider/parsing errors are reported without compromising the other source.
            messages.append(f"{name} source unavailable: {exc}")
        frames[name] = frame
    frames["prices"] = frames["prices"].rename(columns={v: k for k, v in MARKET_SYMBOLS.items()})
    frames.update({"data_quality": messages, "cached": cached,
                   "sources": {"prices": "Yahoo adjusted Close; DXY=DX-Y.NYB; VIX=^VIX",
                               "yields": "FRED daily constant-maturity yields (percent per annum)"}})
    return frames


def yield_change_bp(current: float, previous: float) -> float:
    """Percentage-point yield input: 4.10 - 4.00 = 10 bp."""
    return float((current - previous) * 100)


def curve_spread_bp(short_yield: float, long_yield: float) -> float:
    return yield_change_bp(long_yield, short_yield)


def classify_curve_movement(short_change_bp: float, long_change_bp: float,
                            tolerance_bp: float = 0.01) -> str:
    """Bull=both yields fall; bear=both rise; steepen=long-minus-short increases.

    Opposite-direction moves are twists, flat legs are parallel/unchanged or
    unclassified rather than forcing a misleading bull/bear interpretation.
    """
    if not np.isfinite([short_change_bp, long_change_bp]).all():
        return "unavailable"
    spread_change = long_change_bp - short_change_bp
    if abs(short_change_bp) <= tolerance_bp and abs(long_change_bp) <= tolerance_bp:
        return "unchanged"
    if abs(spread_change) <= tolerance_bp:
        return "parallel_shift"
    shape = "steepener" if spread_change > 0 else "flattener"
    if short_change_bp < -tolerance_bp and long_change_bp < -tolerance_bp:
        return f"bull_{shape}"
    if short_change_bp > tolerance_bp and long_change_bp > tolerance_bp:
        return f"bear_{shape}"
    if short_change_bp * long_change_bp < 0:
        return f"twist_{shape}"
    return f"mixed_{shape}"


def relative_returns(prices: pd.DataFrame, first: str, second: str,
                     periods: int = 1) -> pd.Series:
    if first not in prices or second not in prices:
        return pd.Series(dtype=float)
    joint = prices[[first, second]].dropna()
    returns = joint.pct_change(periods=periods, fill_method=None)
    return (returns[first] - returns[second]).rename(f"{first}_{second}")


def rolling_zscore(series: pd.Series, window: int = MARKET_ZSCORE_WINDOW) -> pd.Series:
    if window < 2:
        raise ValueError("z-score window must be at least two observations.")
    rolling = series.rolling(window, min_periods=window)
    return (series - rolling.mean()) / rolling.std(ddof=1).replace(0, np.nan)


def _number(value) -> float | None:
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def _clean_frame(frame: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    return frame.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")].loc[:as_of].apply(pd.to_numeric, errors="coerce")


def _metric(series: pd.Series, unit: str, as_of: pd.Timestamp, window: int,
            max_age_days: int) -> dict:
    series = series.dropna()
    date = series.index[-1] if len(series) else None
    stale = date is not None and (as_of - date).days > max_age_days
    result = {"value": _number(series.iloc[-1]) if len(series) else None,
              "as_of": date.isoformat() if date is not None else None,
              "stale": stale, "unit": unit, "changes": {}, "zscore": None}
    for label, n in HORIZONS.items():
        change = None
        if len(series) > n and not stale:
            if unit == "yield_percent":
                change = yield_change_bp(series.iloc[-1], series.iloc[-n-1])
            elif unit == "basis_points":
                change = series.iloc[-1] - series.iloc[-n-1]
            else:
                prior = series.iloc[-n-1]
                change = series.iloc[-1] / prior - 1 if prior != 0 else np.nan
        result["changes"][label] = _number(change)
    if len(series) and not stale:
        result["zscore"] = _number(rolling_zscore(series, window).iloc[-1])
    result["change_unit"] = "basis_points" if unit in {"yield_percent", "basis_points"} else "decimal_return"
    return result


def build_market_snapshot(prices: pd.DataFrame, yields: pd.DataFrame,
                          as_of=None, zscore_window: int = MARKET_ZSCORE_WINDOW,
                          max_age_days: int = 7) -> dict:
    """Deterministic, JSON-safe output, with facts, signals, and hypotheses separated."""
    if zscore_window < 2:
        raise ValueError("z-score window must be at least two.")
    available_dates = [f.index.max() for f in (prices, yields) if len(f)]
    date = pd.Timestamp(as_of) if as_of is not None else (max(available_dates) if available_dates else pd.NaT)
    if pd.isna(date):
        date = pd.Timestamp("1970-01-01")  # deterministic empty-input sentinel
    date = pd.Timestamp(date).tz_localize(None).normalize()
    prices, yields = _clean_frame(prices, date), _clean_frame(yields, date)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    price_metrics = {t: _metric(prices.get(t, empty), "index_points" if t in {"DXY", "VIX"} else "adjusted_price", date, zscore_window, max_age_days) for t in MARKET_SYMBOLS}
    rates = {t: _metric(yields.get(t, empty), "yield_percent", date, zscore_window, max_age_days) for t in TREASURY_SERIES}
    joint = yields.reindex(columns=["2Y", "10Y"]).dropna()
    spread = (joint["10Y"] - joint["2Y"]) * 100
    curve = _metric(spread, "basis_points", date, zscore_window, max_age_days)
    classification = "unavailable"
    if len(joint) > 1 and not curve["stale"]:
        classification = classify_curve_movement(yield_change_bp(joint["2Y"].iloc[-1], joint["2Y"].iloc[-2]),
                                                yield_change_bp(joint["10Y"].iloc[-1], joint["10Y"].iloc[-2]))
    curve.update({"spread_2s10s_bp": curve["value"], "daily_classification": classification})
    relative = {}
    for name, (first, second) in RELATIVE_PAIRS.items():
        pair = prices.reindex(columns=[first, second]).dropna()
        pair_date = pair.index[-1] if len(pair) else None
        stale = pair_date is not None and (date - pair_date).days > max_age_days
        values = {label: (_number(relative_returns(pair, first, second, n).iloc[-1]) if len(pair) > n and not stale else None) for label, n in HORIZONS.items()}
        relative[name] = {"returns": values, "unit": "decimal_return_difference", "as_of": pair_date.isoformat() if pair_date is not None else None, "stale": stale}
    quality = []
    observations = []
    for group, metrics in (("prices", price_metrics), ("rates", rates)):
        for name, metric in metrics.items():
            if metric["value"] is None or metric["stale"]:
                quality.append(f"{name}: {'stale' if metric['stale'] else 'missing'}")
            observations.append({"type": "observed_fact", "series": name, "group": group,
                                 "value": metric["value"], "unit": metric["unit"], "as_of": metric["as_of"]})
    signals, interpretation = [], []
    def add(name, value, text):
        signals.append({"type": "derived_signal", "name": name, "value": value, "horizon": "1D", "text": text})
    if classification != "unavailable":
        add("curve", classification, f"2s10s daily movement: {classification.replace('_', ' ')}.")
    for name, metric in relative.items():
        value = metric["returns"]["1D"]
        if value is None:
            continue
        first, second = RELATIVE_PAIRS[name]
        direction = "outperforming" if value > 0 else "underperforming" if value < 0 else "matching"
        add(name, value, f"{first} is {direction} {second} on their latest common observation.")
    breadth = relative["RSP_SPY"]["returns"]["1D"]
    if breadth is not None and breadth < 0:
        interpretation.append({"type": "interpretation", "text": "Equal-weight underperformance is consistent with weaker breadth; it does not identify its cause."})
    credit = relative["HYG_LQD"]["returns"]["1D"]
    if credit is not None:
        interpretation.append({"type": "interpretation", "text": ("Credit ETF relative performance is stable within 10 bp today." if abs(credit) <= .001 else "High-yield versus investment-grade ETF performance suggests a change in risk conditions; duration differences also matter.")})
    for name in ("DXY", "VIX"):
        value = price_metrics[name]["changes"]["1D"]
        if value is not None:
            add(name, value, f"{'The dollar' if name == 'DXY' else 'VIX'} is {'rising' if value > 0 else 'falling' if value < 0 else 'unchanged'}.")
    # Do not compare observations from different dates as though simultaneous.
    spy, ten = price_metrics["SPY"], rates["10Y"]
    if spy["as_of"] == ten["as_of"] and (spy["changes"]["1D"] or 0) > 0 and (ten["changes"]["1D"] or 0) > 0:
        interpretation.append({"type": "interpretation", "text": "Equities are rising alongside higher long-term yields; these observations do not establish causality."})
    if classification == "bear_steepener":
        interpretation.append({"type": "interpretation", "text": "Rising yields with a faster rise at the long end are more consistent with long-end rate pressure than exclusively front-end repricing."})
    snapshot = {"timestamp": date.isoformat(), "rates": rates,
                "equities": {t: price_metrics[t] for t in ("SPY", "QQQ", "RSP", "IWM", "SMH", "XLY", "XLP")},
                "credit": {t: price_metrics[t] for t in ("HYG", "LQD")},
                "fixed_income": {t: price_metrics[t] for t in ("TLT", "IEF", "SGOV", "BIL", "TIP")},
                "other": {t: price_metrics[t] for t in ("GLD", "DBC", "VNQ")},
                "fx": {"DXY": price_metrics["DXY"]}, "volatility": {"VIX": price_metrics["VIX"]},
                "curve": curve, "relative_performance": relative, "observations": observations,
                "signals": signals, "interpretation": interpretation, "data_quality": quality}
    realized = prices.get("SPY", empty).pct_change(fill_method=None).rolling(21, min_periods=21).std() * np.sqrt(252)
    spy_stale = price_metrics["SPY"]["stale"]
    snapshot["volatility"]["spy_realized_21d"] = None if realized.empty or spy_stale else _number(realized.iloc[-1])
    return snapshot


def get_regime_features(snapshot: dict, horizon: str = "21D") -> dict[str, float]:
    """Numeric observed proxies only; missing/stale inputs are omitted, never zero-filled."""
    if horizon not in HORIZONS:
        raise ValueError(f"Unknown horizon: {horizon}")
    features = {}
    def add(name, value):
        if value is not None and np.isfinite(value):
            features[name] = float(value)
    for pair in ("IWM_SPY", "RSP_SPY", "XLY_XLP", "HYG_LQD"):
        add(f"growth_{pair.lower()}_{horizon}", snapshot["relative_performance"][pair]["returns"][horizon])
    add(f"growth_spy_{horizon}", snapshot["equities"]["SPY"]["changes"][horizon])
    add(f"inflation_10y_change_bp_{horizon}", snapshot["rates"]["10Y"]["changes"][horizon])
    add(f"inflation_tip_ief_{horizon}", snapshot["relative_performance"]["TIP_IEF"]["returns"][horizon])
    for asset in ("DBC", "GLD"):
        add(f"inflation_{asset.lower()}_{horizon}", snapshot["other"][asset]["changes"][horizon])
    add(f"conditions_2y_change_bp_{horizon}", snapshot["rates"]["2Y"]["changes"][horizon])
    add(f"conditions_dxy_{horizon}", snapshot["fx"]["DXY"]["changes"][horizon])
    add(f"conditions_hyg_{horizon}", snapshot["credit"]["HYG"]["changes"][horizon])
    add(f"conditions_vix_{horizon}", snapshot["volatility"]["VIX"]["changes"][horizon])
    add("stress_vix_zscore", snapshot["volatility"]["VIX"]["zscore"])
    add("stress_spy_realized_vol_21d", snapshot["volatility"]["spy_realized_21d"])
    for pair, name in (("HYG_LQD", "credit_weakness"), ("RSP_SPY", "breadth_weakness")):
        value = snapshot["relative_performance"][pair]["returns"][horizon]
        add(f"stress_{name}_{horizon}", -value if value is not None else None)
    return features


def main() -> None:
    import argparse
    from dashboard import market_intelligence_table
    parser = argparse.ArgumentParser(description="Market Intelligence — observed cross-asset signals")
    parser.add_argument("--start", default=(pd.Timestamp.now() - pd.Timedelta(days=800)).date().isoformat())
    parser.add_argument("--end", default=None, help="Yahoo end is exclusive; snapshot is truncated to the latest available observation.")
    parser.add_argument("--json", action="store_true", help="Emit structured snapshot and numeric regime features")
    args = parser.parse_args()
    data = load_market_data(args.start, args.end)
    snapshot = build_market_snapshot(data["prices"], data["yields"],
                                     as_of=args.end or pd.Timestamp.now().normalize())
    snapshot["data_quality"].extend(data["data_quality"])
    snapshot["sources"] = data["sources"]
    snapshot["features"] = get_regime_features(snapshot)
    if args.json:
        print(json.dumps(snapshot, indent=2, allow_nan=False))
    else:
        print(f"\nMarket Intelligence — as of {snapshot['timestamp']}")
        print(market_intelligence_table(snapshot).to_string(index=False))
        print(f"\nCurve: {snapshot['curve']['daily_classification']}")
        for item in snapshot["signals"] + snapshot["interpretation"]:
            print(f"[{item['type']}] {item['text']}")
        if snapshot["data_quality"]:
            print("\nData quality: " + "; ".join(snapshot["data_quality"]))


if __name__ == "__main__":
    main()
