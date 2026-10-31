"""The Medium: a local open-weight LLM (via Ollama) that tells each ghost's story.

It is given only the numbers. It is told explicitly that it does NOT know why the
move happened, so it must not invent news, earnings, or rumours. The haunting is
the uncertainty itself.
"""
from __future__ import annotations

import json
import urllib.request

SYSTEM = """You are the Medium in a haunted trading journal called Ghost Candles.
You receive the facts about one day when a stock moved in a way a forecasting model
said was nearly impossible. Write a short ghost story (90-130 words) about that day.

Rules you must never break:
- You do NOT know why the move happened. Never invent news, earnings, company events,
  people, rumours or reasons. The mystery IS the story.
- Use only the numbers given. Do not make up any other figures.
- Second person or third person is fine. Atmospheric, not cheesy. No emojis.
- End with one plain line starting with "The record:" that states the facts in one sentence."""


def _facts(g: dict) -> str:
    return (
        f"Ticker: {g['ticker']}\n"
        f"Date: {g['date']} ({g['weekday']})\n"
        f"Actual move: {g['ret_pct']:+.2f}% close-to-close\n"
        f"Model's 98% range for that day: {g['lo_pct']:+.2f}% to {g['hi_pct']:+.2f}%\n"
        f"Model's odds of a move this extreme: about 1 in {g['one_in_label']}\n"
        f"A normal-distribution model's odds: about 1 in {g['gauss_label']}\n"
        f"Volume that day: {g['volume_ratio']:.1f}x the 20-day average\n"
        f"Previous 5 days' return: {g['prev5_pct']:+.2f}%\n"
        f"Ghost class: {g['klass']}"
    )


def template_story(g: dict) -> str:
    direction = "fell" if g["ret_pct"] < 0 else "rose"
    return (
        f"On {g['weekday']}, {g['date']}, {g['ticker']} {direction} {abs(g['ret_pct']):.2f}%. "
        f"Everything the past knew said the day should end between {g['lo_pct']:+.2f}% and "
        f"{g['hi_pct']:+.2f}%. It didn't. Volume ran at {g['volume_ratio']:.1f}x normal, as if "
        f"something was moving through the order book that nobody could see.\n"
        f"The record: a {g['ret_pct']:+.2f}% day the model rated about 1 in {g['one_in_label']}."
    )


class Medium:
    def __init__(self, model: str = "gemma3:4b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host.rstrip("/")
        self.available = self._ping()

    def _ping(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama:{self.model}" if self.available else "template (Ollama not running)"

    def tell(self, g: dict) -> str:
        if not self.available:
            return template_story(g)
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 260},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _facts(g)},
            ],
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["message"]["content"].strip()
