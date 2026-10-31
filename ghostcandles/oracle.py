"""The Oracle: walk-forward probabilistic forecasts of each day's return.

For every test day the oracle only sees the past. It refits every `step` days on a
rolling window of `window` days, then predicts a full distribution for the next block.
A candle that lands far outside its predicted distribution is a ghost: something
the past could not account for.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .features import FEATURES

# Dense quantile grid so we can locate where the actual return fell.
LEVELS = np.array([0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                   0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.9975, 0.999])


@dataclass
class Forecast:
    q: pd.DataFrame          # predicted quantiles, columns = LEVELS
    backend: str
    seconds: float


def _ensure_tabpfn_token() -> None:
    """Get a Prior Labs API key without TabPFN's browser login.

    TabPFN's browser login waits on stdin with select(), which crashes on Windows
    (WinError 10038). So we ask for the key ourselves once, save it to TabPFN's own
    token cache (~/.cache/tabpfn/auth_token), and disable the browser flow.
    """
    import os
    from getpass import getpass
    from pathlib import Path

    os.environ.setdefault("TABPFN_NO_BROWSER", "1")
    token_file = Path.home() / ".cache" / "tabpfn" / "auth_token"
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return
    if token_file.is_file() and token_file.read_text().strip():
        return
    print("\nTabPFN needs your Prior Labs API key (one time only).")
    print("  1. Open https://ux.priorlabs.ai/account and copy your API key")
    print("  2. Make sure TabPFN-3.5 shows 'Accepted' under Licenses")
    key = getpass("  Paste API key (input is hidden), then press Enter: ").strip()
    if not key:
        raise SystemExit("No API key entered.")
    token_file.parent.mkdir(parents=True, exist_ok=True)
    token_file.write_text(key)
    print(f"  Saved to {token_file}. You won't be asked again.\n")


def _tabpfn_model(version: str):
    _ensure_tabpfn_token()
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion

    return TabPFNRegressor.create_default_for_version(ModelVersion(version))


def _predict_block(backend, model, Xtr, ytr, Xte):
    if backend == "tabpfn":
        model.fit(Xtr, ytr)
        out = model.predict(Xte, output_type="quantiles", quantiles=LEVELS.tolist())
        return np.column_stack(out)
    if backend == "fhs":  # filtered historical simulation: test/fallback backend, NOT TabPFN
        vol_tr = Xtr[:, FEATURES.index("vol_20d")]
        z = ytr / vol_tr
        zq = np.quantile(z, LEVELS)
        vol_te = Xte[:, FEATURES.index("vol_20d")]
        return np.outer(vol_te, zq) + np.median(ytr)
    raise ValueError(backend)


def walk_forward(feat: pd.DataFrame, backend: str = "tabpfn", version: str = "v3.5",
                 window: int = 750, step: int = 20, verbose: bool = True) -> Forecast:
    model = _tabpfn_model(version) if backend == "tabpfn" else None
    X = feat[FEATURES].to_numpy(dtype=np.float32)
    y = feat["target"].to_numpy(dtype=np.float32)
    n = len(feat)
    if n <= window + 20:
        raise ValueError(f"Need more than {window + 20} usable days, got {n}. Use --years 6 or a smaller --window.")
    rows, t0 = [], time.time()
    blocks = range(window, n, step)
    for i, start in enumerate(blocks):
        end = min(start + step, n)
        lo = max(0, start - window)
        q = _predict_block(backend, model, X[lo:start], y[lo:start], X[start:end])
        rows.append(np.sort(q, axis=1))  # guard against tiny quantile crossings
        if verbose:
            print(f"\r  oracle [{backend}] block {i + 1}/{len(blocks)}", end="", flush=True)
    if verbose:
        print()
    q = pd.DataFrame(np.vstack(rows), index=feat.index[window:], columns=LEVELS)
    name = f"tabpfn-{version}" if backend == "tabpfn" else "fhs (test backend, not TabPFN)"
    return Forecast(q=q, backend=name, seconds=time.time() - t0)


def gaussian_baseline(feat: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """What a textbook 'returns are normal' model would have expected."""
    vol = feat.loc[index, "vol_60d"].to_numpy()
    return pd.DataFrame(np.outer(vol, norm.ppf(LEVELS)), index=index, columns=LEVELS)


def rank_of_actual(q: pd.DataFrame, actual: pd.Series) -> pd.Series:
    """Where in the predicted distribution did reality land? (0..1, interpolated)."""
    out = []
    for qs, a in zip(q.to_numpy(), actual.to_numpy()):
        if a <= qs[0]:
            out.append(LEVELS[0] / 2)
        elif a >= qs[-1]:
            out.append(1 - (1 - LEVELS[-1]) / 2)
        else:
            out.append(float(np.interp(a, qs, LEVELS)))
    return pd.Series(out, index=q.index)


def pinball(q: pd.DataFrame, actual: pd.Series) -> float:
    """Mean quantile loss across the grid. Lower is better."""
    d = actual.to_numpy()[:, None] - q.to_numpy()
    return float(np.mean(np.maximum(LEVELS * d, (LEVELS - 1) * d)))


def coverage(q: pd.DataFrame, actual: pd.Series, lo: float, hi: float) -> float:
    a = actual.to_numpy()
    return float(np.mean((a >= q[lo].to_numpy()) & (a <= q[hi].to_numpy())))
