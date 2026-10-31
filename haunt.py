#!/usr/bin/env python3
"""Ghost Candles: find the days a stock did something its past said was nearly impossible.

  python haunt.py RELIANCE.NS                       # TabPFN-3.5 + local Ollama narrator
  python haunt.py TATAMOTORS.NS --version v3.5-fast  # faster checkpoint on CPU
  python haunt.py --csv mydata.csv --ticker MYSTOCK
  python haunt.py --synthetic --backend fhs         # offline smoke test (not TabPFN)
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from ghostcandles import data, features, oracle, report
from ghostcandles.narrator import Medium

CLASS_CUTOFF = 0.01   # two-sided odds <= 1% -> ghost
WRAITH_CUTOFF = 0.002  # beyond the oracle's 0.1% / 99.9% quantiles


def fmt_one_in(p: float, capped: bool = False) -> tuple[int, str]:
    n = int(round(1 / max(p, 1e-12)))
    label = f"{n:,}+" if capped else (f"{n:,}" if n < 10**9 else "a billion+")
    return n, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol", nargs="?", default="RELIANCE.NS", help="Yahoo ticker, e.g. RELIANCE.NS")
    ap.add_argument("--csv", help="Load OHLCV from CSV instead of Yahoo")
    ap.add_argument("--ticker", help="Display name (defaults to symbol)")
    ap.add_argument("--synthetic", action="store_true", help="Use fake data (testing only)")
    ap.add_argument("--years", type=int, default=6)
    ap.add_argument("--backend", choices=["tabpfn", "fhs"], default="tabpfn")
    ap.add_argument("--version", default="v3.5", help="TabPFN checkpoint: v3.5, v3.5-fast, v2 ...")
    ap.add_argument("--window", type=int, default=750, help="Training window in trading days")
    ap.add_argument("--step", type=int, default=20, help="Refit every N days")
    ap.add_argument("--llm", default="gemma3:4b", help="Ollama model for the narrator")
    ap.add_argument("--max-ghosts", type=int, default=12)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    ticker = a.ticker or ("SYNTHETIC" if a.synthetic else a.symbol)
    print(f"Summoning candles for {ticker} ...")
    df = data.synthetic() if a.synthetic else data.load_csv(a.csv) if a.csv else data.fetch_yahoo(a.symbol, a.years)
    feat = features.build(df)
    print(f"  {len(df)} candles, {len(feat)} usable days, {len(feat) - a.window} forecast out-of-sample")

    fc = oracle.walk_forward(feat, a.backend, a.version, a.window, a.step)
    q = fc.q
    actual = feat.loc[q.index, "target"]
    gq = oracle.gaussian_baseline(feat, q.index)

    rank = oracle.rank_of_actual(q, actual)
    p_two = 2 * np.minimum(rank, 1 - rank)
    z = actual / feat.loc[q.index, "vol_60d"]
    p_gauss = pd.Series(2 * norm.sf(np.abs(z)), index=q.index)

    is_ghost = p_two <= CLASS_CUTOFF
    gauss_ghost = p_gauss <= CLASS_CUTOFF

    ghosts = []
    order = p_two[is_ghost].sort_values().index[: a.max_ghosts]
    for d in sorted(order):
        capped = (actual[d] <= q.loc[d, oracle.LEVELS[0]]) or (actual[d] >= q.loc[d, oracle.LEVELS[-1]])
        one_in, one_in_label = fmt_one_in(p_two[d], capped)
        g_one_in, g_label = fmt_one_in(p_gauss[d])
        i = feat.index.get_loc(d)
        ghosts.append({
            "ticker": ticker,
            "date": d.strftime("%Y-%m-%d"),
            "weekday": d.strftime("%A"),
            "ret_pct": float(np.expm1(actual[d]) * 100),
            "lo_pct": float(np.expm1(q.loc[d, 0.01]) * 100),
            "hi_pct": float(np.expm1(q.loc[d, 0.99]) * 100),
            "one_in": one_in, "one_in_label": one_in_label,
            "gauss_one_in": g_one_in, "gauss_label": g_label,
            "volume_ratio": float(feat.loc[d, "volume_ratio"]),
            "prev5_pct": float(np.expm1(feat["target"].iloc[max(0, i - 5):i].sum()) * 100),
            "close": float(feat.loc[d, "close"]),
            "klass": "Wraith" if p_two[d] <= WRAITH_CUTOFF else "Poltergeist",
        })

    medium = Medium(a.llm)
    print(f"  medium: {medium.name}")
    for k, g in enumerate(ghosts, 1):
        print(f"\r  channelling ghost {k}/{len(ghosts)}", end="", flush=True)
        g["story"] = medium.tell(g)
    print()

    n = len(actual)
    stats = {
        "Oracle": fc.backend,
        "Narrator": medium.name,
        "Story drafts rejected by fact check": f"{sum(1 for e in medium.log if e['problems'])} of {len(medium.log)}",
        "Out-of-sample days": n,
        "Walk-forward": f"train on previous {a.window} days, refit every {a.step}",
        "Oracle runtime (s)": f"{fc.seconds:.0f}",
        "90% band coverage, oracle (ideal 90%)": f"{oracle.coverage(q, actual, 0.05, 0.95):.1%}",
        "90% band coverage, bell curve": f"{oracle.coverage(gq, actual, 0.05, 0.95):.1%}",
        "98% band coverage, oracle (ideal 98%)": f"{oracle.coverage(q, actual, 0.01, 0.99):.1%}",
        "98% band coverage, bell curve": f"{oracle.coverage(gq, actual, 0.01, 0.99):.1%}",
        "Quantile loss x1e4, oracle (lower is better)": f"{oracle.pinball(q, actual) * 1e4:.3f}",
        "Quantile loss x1e4, bell curve": f"{oracle.pinball(gq, actual) * 1e4:.3f}",
        "Ghosts (odds <= 1 in 100), oracle": int(is_ghost.sum()),
        "Ghosts expected by pure chance": f"{n * CLASS_CUTOFF:.1f}",
        "Ghosts by bell curve": int(gauss_ghost.sum()),
    }

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", ticker).strip("_")
    html_path = out / f"{slug}.html"
    report.render(html_path, ticker=ticker, close=feat.loc[q.index, "close"], q=q, actual=actual,
                  ghosts=ghosts, stats=stats, oracle_name=fc.backend, medium_name=medium.name)
    (out / f"{slug}.json").write_text(json.dumps({"stats": stats, "ghosts": ghosts}, indent=2))

    print("\n".join(f"  {k}: {v}" for k, v in stats.items()))
    print(f"\nHaunted journal written to {html_path}")


if __name__ == "__main__":
    main()
