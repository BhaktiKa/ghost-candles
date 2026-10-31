"""Fact-check a ghost story before it reaches the journal.

Small local models drift: in testing, gemma3:1b turned a +5.26% day into
"nearly eight percent". The guard rejects a story if it
  - uses a number that isn't in the facts it was given,
  - spells a number out in words (which can't be checked),
  - names a cause (news, earnings, elections ...) the model can't know, or
  - doesn't end with the "The record:" line.
"""
from __future__ import annotations

import re

CAUSE_WORDS = [
    "earnings", "results", "quarterly", "profit", "revenue", "election", "vote", "news",
    "announce", "ceo", "chairman", "scandal", "rumor", "rumour", "government", "policy",
    "budget", "merger", "acquisition", "lawsuit", "indict", "regulator", "sebi", "rbi",
    "war", "tariff", "fraud", "guidance", "dividend", "split",
]
NUMBER_WORDS = [
    "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "fifteen", "twenty", "thirty", "fifty", "hundred", "thousand", "million",
    "billion", "half", "twice", "double", "triple", "quarter",
]


def _allowed_numbers(g: dict) -> set[float]:
    vals = [g["ret_pct"], g["lo_pct"], g["hi_pct"], g["volume_ratio"], g["prev5_pct"],
            g["one_in"], g["gauss_one_in"], 98, 20, 5, 100, 1]
    vals += [int(p) for p in g["date"].split("-")]
    out = set()
    for v in vals:
        for x in (v, abs(v)):
            out.update({round(float(x), 2), round(float(x), 1), float(round(x))})
    return out


def check(story: str, g: dict) -> list[str]:
    problems = []
    text = story.lower()
    allowed = _allowed_numbers(g)
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", story):
        num = float(raw.replace(",", ""))
        if not any(abs(num - a) < 0.011 for a in allowed):
            problems.append(f"number not in the facts: {raw}")
    for w in NUMBER_WORDS:
        if re.search(rf"\b{w}\b", text):
            problems.append(f"spelled-out number: '{w}'")
    for w in CAUSE_WORDS:
        if re.search(rf"\b{w}", text):
            problems.append(f"names a possible cause: '{w}'")
    lines = [l.strip() for l in story.strip().splitlines() if l.strip()]
    if len(lines) < 2 or not lines[-1].lower().startswith("the record:"):
        problems.append("the story must come first, then a separate final line starting 'The record:'")
    if story.lower().count("the record:") > 1:
        problems.append("'The record:' appears more than once")
    return problems
