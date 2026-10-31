"""Turn candles into a table TabPFN can learn from.

Every feature for day t uses ONLY information available at the close of day t-1.
The target is day t's close-to-close log return. No peeking at the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    c, o, h, l, v = (df[k] for k in ("Close", "Open", "High", "Low", "Volume"))
    ret = np.log(c).diff()
    logv = np.log(v.replace(0, np.nan))

    f = pd.DataFrame(index=df.index)
    for k in (1, 2, 3, 5):
        f[f"ret_lag{k}"] = ret.shift(k)
    f["ret_5d"] = ret.rolling(5).sum().shift(1)
    f["ret_20d"] = ret.rolling(20).sum().shift(1)
    f["vol_5d"] = ret.rolling(5).std().shift(1)
    f["vol_20d"] = ret.rolling(20).std().shift(1)
    f["vol_60d"] = ret.rolling(60).std().shift(1)
    f["range_lag1"] = (np.log(h) - np.log(l)).shift(1)
    f["range_20d"] = (np.log(h) - np.log(l)).rolling(20).mean().shift(1)
    f["body_lag1"] = ((c - o) / (h - l).replace(0, np.nan)).shift(1)
    f["volz_lag1"] = ((logv - logv.rolling(20).mean()) / logv.rolling(20).std()).shift(1)
    f["dist_ma50"] = (np.log(c) - np.log(c.rolling(50).mean())).shift(1)
    f["drawdown_60d"] = (np.log(c) - np.log(c.rolling(60).max())).shift(1)
    f["weekday"] = df.index.dayofweek
    f["month"] = df.index.month

    f["target"] = ret
    # context columns, not fed to the model
    f["close"] = c
    f["volume_ratio"] = (v / v.rolling(20).mean().shift(1))
    return f.dropna()


FEATURES = [
    "ret_lag1", "ret_lag2", "ret_lag3", "ret_lag5", "ret_5d", "ret_20d",
    "vol_5d", "vol_20d", "vol_60d", "range_lag1", "range_20d", "body_lag1",
    "volz_lag1", "dist_ma50", "drawdown_60d", "weekday", "month",
]
