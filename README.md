# Ghost Candles

**Find the days a stock did something its own past said was nearly impossible, and let a local model tell their ghost stories.**

Built for the DEV Hacktoberfest Open-Source AI Challenge, Week 4: *Ghost in the Machine*.

- **The Oracle:** [TabPFN-3.5](https://github.com/PriorLabs/TabPFN), an open-weight tabular foundation model, runs locally. For every trading day it forecasts a full probability distribution of that day's return, using only data from before that day (walk-forward, refit every 20 days on the previous 750).
- **The Ghosts:** any day whose actual move had odds of 1 in 100 or worse under the Oracle's forecast. Beyond the 0.1% / 99.9% quantiles it becomes a *Wraith*.
- **The Medium:** a local open-weight LLM (default Google Gemma `gemma3:4b` via [Ollama](https://ollama.com)) writes a short ghost story for each one. It only gets the numbers and is told it does **not** know why the move happened, so it can't invent news.
- **The Journal:** one self-contained HTML file with the price chart, the Oracle's 98% band, the ghost cards, and an honesty table (band coverage and quantile loss vs. a bell-curve baseline).

Everything runs on a laptop. No API keys at inference time, and no trading data leaves your machine.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# one-time: accept the TabPFN licence at https://ux.priorlabs.ai (License tab).
# The first run opens a browser to log in; on headless machines set TABPFN_TOKEN instead.

ollama pull gemma3:4b        # optional: without Ollama, stories fall back to a plain template

python haunt.py RELIANCE.NS                          # TabPFN-3.5
python haunt.py TATAMOTORS.NS --version v3.5-fast    # quicker on CPU
python haunt.py --csv my_ohlcv.csv --ticker MYSTOCK  # any Date,Open,High,Low,Close,Volume CSV
```

Output: `out/<TICKER>.html` (open in a browser) and `out/<TICKER>.json`.

Offline smoke test (no TabPFN, no internet; uses fake data and a simple *filtered historical simulation* backend):

```bash
python haunt.py --synthetic --backend fhs
```

## How it works

| Step | File | Notes |
|---|---|---|
| Candles | `ghostcandles/data.py` | yfinance (`.NS` / `.BO`) or CSV |
| Features | `ghostcandles/features.py` | 17 features, all lagged to t-1: recent returns, 5/20/60-day volatility, range, candle body, volume z-score, distance from 50-DMA, 60-day drawdown, weekday, month |
| Oracle | `ghostcandles/oracle.py` | `TabPFNRegressor(...).predict(X, output_type="quantiles")` on a 21-level grid from 0.1% to 99.9%; the actual return's rank inside that grid gives its odds |
| Baseline | `ghostcandles/oracle.py` | Normal distribution with 60-day volatility |
| Medium | `ghostcandles/narrator.py` | Ollama `/api/chat`; template fallback |
| Journal | `ghostcandles/report.py` | Inline SVG, no external assets |

## Limits

- This is a storytelling and anomaly-spotting tool, **not trading advice** and not a trading signal.
- "1 in N" odds are the Oracle's own estimate. Anything beyond its 0.1% quantile is shown as `1,000+` rather than an invented exact number.
- TabPFN on CPU is slow. Use `--version v3.5-fast` or a bigger `--step` if needed.

## Credits and licences

- Code: MIT.
- TabPFN by Prior Labs. The TabPFN-3.5 weights are under a **non-commercial licence** you accept at ux.priorlabs.ai.
- Narrator model: Google Gemma (Gemma Terms of Use) via Ollama. Swap in any Ollama model with `--llm`.
- Market data: Yahoo Finance via yfinance, for personal and research use.
