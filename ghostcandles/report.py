"""Render the haunted journal: one self-contained HTML file, no external assets."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

W, H, PAD = 960, 260, 36

# Ghost cards laid out landscape: facts on the left, story on the right, one card per row.
# Stacks back to portrait on narrow (phone) screens. restyle.py injects this into old journals.
LANDSCAPE_CSS = """<style id="landscape">
.grid{display:grid;grid-template-columns:1fr;gap:16px}
.card{display:grid;grid-template-columns:280px 1fr;grid-template-rows:auto auto 1fr;column-gap:28px;align-items:start;padding:20px 24px}
.card>header,.card>.move,.card>dl{grid-column:1}
.card>.story{grid-column:2;grid-row:1 / span 3;margin:0;font-size:15px;line-height:1.6;padding-left:24px;border-left:1px solid var(--line)}
.card>dl{grid-template-columns:1fr 1fr}
@media (max-width:700px){.card{grid-template-columns:1fr}.card>.story{grid-column:1;grid-row:auto;margin-top:12px;padding-left:0;border-left:0}}
</style>"""


def _x(i, n):
    return PAD + (W - 2 * PAD) * i / max(n - 1, 1)


def _price_svg(close: pd.Series, ghosts: list[dict]) -> str:
    n = len(close)
    lo, hi = float(close.min()), float(close.max())
    y = lambda v: H - PAD - (H - 2 * PAD) * (v - lo) / (hi - lo or 1)
    pts = " ".join(f"{_x(i, n):.1f},{y(v):.1f}" for i, v in enumerate(close.to_numpy()))
    pos = {d: i for i, d in enumerate(close.index.strftime("%Y-%m-%d"))}
    marks = "".join(
        f'<a href="#g{k}"><circle class="ghost-dot {g["klass"].lower()}" cx="{_x(pos[g["date"]], n):.1f}" '
        f'cy="{y(g["close"]):.1f}" r="6"><title>{g["date"]}: {g["ret_pct"]:+.2f}%</title></circle></a>'
        for k, g in enumerate(ghosts) if g["date"] in pos)
    years = sorted({d.year for d in close.index})
    ticks, last_x = "", -1e9
    for yr in years:
        x = _x(int(np.argmax(close.index.year == yr)), n)
        if x - last_x > 60:  # skip a label that would collide with the previous one
            ticks += f'<text class="tick" x="{x:.1f}" y="{H - 10}">{yr}</text>'
            last_x = x
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Price with ghost candles marked">'
            f'<polyline class="price" points="{pts}"/>{marks}{ticks}</svg>')


def _band_svg(q: pd.DataFrame, actual: pd.Series, ghost_dates: set[str]) -> str:
    n = len(actual)
    lo_q, hi_q = q[0.01].to_numpy(), q[0.99].to_numpy()
    a = actual.to_numpy()
    m = float(np.nanmax(np.abs(np.concatenate([a, lo_q, hi_q])))) or 1
    y = lambda v: H / 2 - (H / 2 - PAD / 2) * v / m
    upper = " ".join(f"{_x(i, n):.1f},{y(v):.1f}" for i, v in enumerate(hi_q))
    lower = " ".join(f"{_x(i, n):.1f},{y(v):.1f}" for i, v in reversed(list(enumerate(lo_q))))
    dates = actual.index.strftime("%Y-%m-%d")
    dots = "".join(
        f'<circle class="{"gd" if d in ghost_dates else "rd"}" cx="{_x(i, n):.1f}" cy="{y(v):.1f}" '
        f'r="{4 if d in ghost_dates else 1.4}"/>' for i, (d, v) in enumerate(zip(dates, a)))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Daily returns against the predicted 98% band">'
            f'<polygon class="band" points="{upper} {lower}"/>'
            f'<line class="zero" x1="{PAD}" x2="{W - PAD}" y1="{y(0):.1f}" y2="{y(0):.1f}"/>{dots}</svg>')


def render(path, *, ticker, close, q, actual, ghosts, stats, oracle_name, medium_name):
    cards = "".join(f"""
    <article class="card {g['klass'].lower()}" id="g{k}">
      <header><span class="date">{g['date']} · {g['weekday']}</span>
        <span class="klass">{g['klass']}</span></header>
      <div class="move {'down' if g['ret_pct'] < 0 else 'up'}">{g['ret_pct']:+.2f}%</div>
      <dl>
        <div><dt>TabPFN's 98% range</dt><dd>{g['lo_pct']:+.2f}% … {g['hi_pct']:+.2f}%</dd></div>
        <div><dt>TabPFN's odds</dt><dd>1 in {g['one_in_label']}</dd></div>
        <div><dt>Bell-curve odds</dt><dd>1 in {g['gauss_label']}</dd></div>
        <div><dt>Volume</dt><dd>{g['volume_ratio']:.1f}× normal</dd></div>
      </dl>
      <p class="story">{html.escape(g['story']).replace(chr(10), '<br>')}</p>
    </article>""" for k, g in enumerate(ghosts))

    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>" for k, v in stats.items())
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ghost Candles · {html.escape(ticker)}</title>
<style>
:root{{--bg:#0d0b12;--panel:#17131f;--ink:#e9e4f2;--dim:#9a90ad;--line:#2a2335;--accent:#b8f2a6;--band:#3a2f4d;--down:#ff7a8a;--up:#8fe3b0;--wraith:#d9b8ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Georgia,'Iowan Old Style',serif}}
main{{max-width:1000px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:2.4rem;margin:0 0 4px;letter-spacing:.02em}}h2{{margin:40px 0 8px;font-size:1.3rem}}
.sub{{color:var(--dim);margin:0 0 24px}}svg{{width:100%;height:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}}
.price{{fill:none;stroke:var(--dim);stroke-width:1.4}}.tick{{fill:var(--dim);font:11px system-ui}}
.ghost-dot{{fill:var(--accent);stroke:var(--bg);stroke-width:2;cursor:pointer}}.ghost-dot.wraith{{fill:var(--wraith)}}
.band{{fill:var(--band);opacity:.75}}.zero{{stroke:var(--line)}}.rd{{fill:var(--dim);opacity:.55}}.gd{{fill:var(--accent)}}
.note{{color:var(--dim);font-size:.9rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}}
.card{{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);border-radius:10px;padding:16px}}
.card.wraith{{border-left-color:var(--wraith)}}.card header{{display:flex;justify-content:space-between;font:13px system-ui;color:var(--dim)}}
.klass{{text-transform:uppercase;letter-spacing:.12em}}.move{{font:600 2rem system-ui;margin:4px 0}}.down{{color:var(--down)}}.up{{color:var(--up)}}
dl{{display:grid;grid-template-columns:1fr 1fr;gap:6px 12px;margin:8px 0;font:13px system-ui}}dt{{color:var(--dim)}}dd{{margin:0}}
.story{{font-style:italic;margin:12px 0 0}}table{{border-collapse:collapse;width:100%;font:14px system-ui}}
td{{border-bottom:1px solid var(--line);padding:6px 4px}}td:last-child{{text-align:right}}
footer{{margin-top:48px;color:var(--dim);font:13px system-ui}}
</style>{LANDSCAPE_CSS}</head><body><main>
<h1>Ghost Candles</h1>
<p class="sub">{html.escape(ticker)} · days the past could not explain · oracle: {html.escape(oracle_name)} · medium: {html.escape(medium_name)}</p>
{_price_svg(close, ghosts)}
<p class="note">Each glowing dot is a ghost candle. Click one to read its story.</p>
<h2>What the oracle (TabPFN) expected</h2>
{_band_svg(q, actual, {g['date'] for g in ghosts})}
<p class="note">Shaded: TabPFN's 98% range for each day, forecast using only data from before that day. Dots: what actually happened.</p>
<h2>The ghosts</h2><div class="grid">{cards}</div>
<h2>How honest is the oracle?</h2><table>{rows}</table>
<footer>Ghost Candles is a storytelling tool, not trading advice. Stories are written by a local model that is
told it does not know the cause of any move, so it must not invent one.</footer>
</main></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(doc)
