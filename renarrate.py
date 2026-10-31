#!/usr/bin/env python3
"""Rewrite the ghost stories in an existing journal without re-running TabPFN.

  python renarrate.py out/RELIANCE_NS.json --llm gemma3:4b

Updates the stories in out/<TICKER>.json and out/<TICKER>.html in place, and writes
out/<TICKER>_factcheck.json listing every rejected draft and why.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path

from ghostcandles.narrator import Medium


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_path")
    ap.add_argument("--llm", default="gemma3:4b")
    ap.add_argument("--retries", type=int, default=3)
    a = ap.parse_args()

    jp = Path(a.json_path)
    hp = jp.with_suffix(".html")
    data = json.loads(jp.read_text(encoding="utf-8"))
    medium = Medium(a.llm, retries=a.retries)
    if not medium.available:
        raise SystemExit("Ollama is not running. Open the Ollama app (or run `ollama serve`) and try again.")
    print(f"Medium: {medium.name}")

    for k, g in enumerate(data["ghosts"], 1):
        print(f"  ghost {k}/{len(data['ghosts'])} {g['date']} ...", end=" ", flush=True)
        g["story"] = medium.tell(g)
        print(g["story_source"])

    data["stats"]["Narrator"] = medium.name
    rejected = [e for e in medium.log if e["problems"]]
    data["stats"]["Story drafts rejected by fact check"] = f"{len(rejected)} of {len(medium.log)}"
    jp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    jp.with_name(jp.stem + "_factcheck.json").write_text(json.dumps(medium.log, indent=2), encoding="utf-8")

    if hp.exists():
        doc = hp.read_text(encoding="utf-8")
        stories = iter(data["ghosts"])
        doc = re.sub(r'<p class="story">.*?</p>',
                     lambda m: '<p class="story">' + html.escape(next(stories)["story"]).replace("\n", "<br>") + "</p>",
                     doc, flags=re.S)
        doc = re.sub(r"medium: [^<]*</p>", f"medium: {html.escape(medium.name)}</p>", doc, count=1)
        doc = re.sub(r"(<tr><td>Narrator</td><td>)[^<]*", lambda m: m.group(1) + html.escape(medium.name), doc)
        hp.write_text(doc, encoding="utf-8")
        print(f"Updated {hp}")
    print(f"Updated {jp}\nRejected drafts: {len(rejected)} of {len(medium.log)} (details in {jp.stem}_factcheck.json)")


if __name__ == "__main__":
    main()
