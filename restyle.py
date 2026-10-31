#!/usr/bin/env python3
"""Update journals that were already generated: landscape ghost cards + TabPFN labels.

  python restyle.py out/ADANIENT_NS.html out/RELIANCE_NS.html

Only layout and labels change; numbers and stories are untouched. Safe to run twice.
"""
import re
import sys
from pathlib import Path

from ghostcandles.report import LANDSCAPE_CSS

LABELS = [
    ("<dt>Oracle's 98% range</dt>", "<dt>TabPFN's 98% range</dt>"),
    ("<dt>Oracle's odds</dt>", "<dt>TabPFN's odds</dt>"),
    ("<h2>What the oracle expected</h2>", "<h2>What the oracle (TabPFN) expected</h2>"),
    ("Shaded: the oracle's 98% range", "Shaded: TabPFN's 98% range"),
]

files = sys.argv[1:] or [str(x) for x in Path("out").glob("*.html")]
for p in files:
    path = Path(p)
    doc = path.read_text(encoding="utf-8")
    doc = re.sub(r'<style id="landscape">.*?</style>', "", doc, flags=re.S)
    doc = doc.replace("</head>", LANDSCAPE_CSS + "</head>", 1)
    for old, new in LABELS:
        doc = doc.replace(old, new)
    path.write_text(doc, encoding="utf-8")
    ok = 'id="landscape"' in doc and "TabPFN's odds" in doc
    print(f"{'OK ' if ok else 'CHECK'} {path}")
