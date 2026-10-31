"""The Medium: a local open-weight LLM (via Ollama) that tells each ghost's story.

It is given only the numbers. It is told explicitly that it does NOT know why the
move happened, so it must not invent news, earnings, or rumours. The haunting is
the uncertainty itself. Every story goes through guard.check(); a story that fails
is sent back with the list of problems, up to `retries` times, before falling back
to a plain template.
"""
from __future__ import annotations

import json
import urllib.request

from .guard import check

SYSTEM = """You are the Medium in a haunted trading journal called Ghost Candles.
You receive the facts about one day when a stock moved in a way a forecasting model
said was nearly impossible. Write a short ghost story (80-120 words) about that day.

Write it like a ghost story, not a market report: the chart as a haunted house,
the candle as something that should not have walked, the order book going cold.
Do not use words like "deviation", "distribution", "statistically" or "parameters".

Rules you must never break:
- You do NOT know why the move happened. Never mention news, earnings, results,
  elections, companies, people, governments or any reason. The mystery IS the story.
- Every number you write must be copied exactly from the facts, written in digits.
  Never write numbers as words. Never round or change a number. If unsure, leave it out.
- No emojis. No headings. No bullet points.
- The LAST line, and only the last line, starts with "The record:" and states the
  move, the date and the model's odds in one plain sentence."""


def _facts(g: dict) -> str:
    return (
        f"Ticker: {g['ticker']}\n"
        f"Date: {g['date']} ({g['weekday']})\n"
        f"Actual move: {g['ret_pct']:+.2f}% close-to-close\n"
        f"Model's 98% range for that day: {g['lo_pct']:+.2f}% to {g['hi_pct']:+.2f}%\n"
        f"Model's odds of a move this extreme: about 1 in {g['one_in_label']}\n"
        f"A bell-curve model's odds: about 1 in {g['gauss_label']}\n"
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
    def __init__(self, model: str = "gemma3:4b", host: str = "http://localhost:11434", retries: int = 3):
        self.model, self.host, self.retries = model, host.rstrip("/"), retries
        self.available = self._ping()
        self.log: list[dict] = []  # every attempt and why it was rejected

    def _ping(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=2)
            return True
        except Exception:
            return False

    @property
    def name(self) -> str:
        return f"ollama:{self.model}" if self.available else "template (Ollama not running)"

    def _chat(self, messages: list[dict]) -> str:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0.8, "num_predict": 300},
            "messages": messages,
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())["message"]["content"].strip()

    def tell(self, g: dict) -> str:
        if not self.available:
            g["story_source"] = "template"
            return template_story(g)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": _facts(g)}]
        for attempt in range(1, self.retries + 1):
            story = self._chat(messages)
            problems = check(story, g)
            self.log.append({"date": g["date"], "attempt": attempt, "problems": problems, "story": story})
            if not problems:
                g["story_source"] = f"{self.model} (attempt {attempt})"
                return story
            messages += [
                {"role": "assistant", "content": story},
                {"role": "user", "content": "Rewrite it. Fix these problems:\n- " + "\n- ".join(problems)
                 + "\nRemember: digits copied exactly from the facts, no causes, 'The record:' only as the last line."},
            ]
        g["story_source"] = "template (model failed the fact check)"
        return template_story(g)
