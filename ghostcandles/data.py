"""Load daily OHLCV candles for an NSE/BSE ticker (yfinance) or from a CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["Open", "High", "Low", "Close", "Volume"]


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next(c for c in df.columns if c.lower() in ("date", "datetime", "timestamp"))
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df.columns = [c.strip().title() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    return df[REQUIRED].dropna()


def fetch_yahoo(ticker: str, years: int = 6) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, period=f"{years}y", interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Try e.g. RELIANCE.NS or TATAMOTORS.NS")
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker)
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[REQUIRED].dropna()
    return df[df["Volume"] > 0]


def synthetic(n_days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Fake market with volatility clustering and a few planted 'ghost' shocks.

    Only used to test the pipeline offline. Never publish results from this.
    """
    rng = np.random.default_rng(seed)
    vol = np.empty(n_days)
    r = np.empty(n_days)
    vol[0] = 0.015
    for t in range(n_days):
        if t:
            vol[t] = np.sqrt(0.000004 + 0.08 * r[t - 1] ** 2 + 0.9 * vol[t - 1] ** 2)
        r[t] = vol[t] * rng.standard_t(5) / np.sqrt(5 / 3)
    shocks = rng.choice(np.arange(300, n_days), 6, replace=False)
    r[shocks] += rng.choice([-1, 1], 6) * rng.uniform(0.08, 0.14, 6)
    close = 1000 * np.exp(np.cumsum(r))
    open_ = close * np.exp(-r * rng.uniform(0.2, 0.8, n_days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol)))
    volume = rng.lognormal(14, 0.3, n_days) * (1 + 40 * np.abs(r))
    idx = pd.bdate_range("2020-11-02", periods=n_days)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
