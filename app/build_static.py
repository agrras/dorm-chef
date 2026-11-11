"""Build a single-file, no-server version of the Compare tab.

    python -m app.build_static          ->  docs/index.html

Precomputes everything the Compare tab needs (judge scores, live rule-check
results, audit) and embeds it in the page, so it can be hosted anywhere that
serves static files (GitHub Pages, Netlify, an S3 bucket) with no API keys.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import replay

REPO = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent / "static" / "index.html"
OUT = REPO / "docs" / "index.html"


def build() -> Path:
    ov = replay.overview()
    payload = {
        "overview": None,
        "samples": {},
        "details": {},
        "audit": replay.audit(),
        "agentExample": replay.agent_example(),
    }
    ov_json = {"experiments": ov["experiments"]}
    payload["overview"] = ov_json
    for ex in ov["experiments"]:
        eid = ex["id"]
        payload["samples"][eid] = replay.sample_index(eid)
        payload["details"][eid] = {
            str(s["id"]): replay.sample_detail(eid, s["id"]) for s in payload["samples"][eid]
        }

    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = SRC.read_text()
    marker = "<script>\nconst $ = "
    if marker not in html:
        raise SystemExit("could not find the main <script> block in static/index.html")
    html = html.replace(marker, f"<script>window.__STATIC__={blob};</script>\n" + marker, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(REPO)} ({p.stat().st_size / 1e6:.1f} MB)")
