"""FastAPI server for dorm-chef.

    uvicorn app.server:app --reload

GET  /              the web UI
GET  /api/config    providers (+ availability), constraint catalog, pantry suggestions
POST /api/cook      run the agent; streams trace events as Server-Sent Events
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent import CookRequest, run_agent
from .providers import build_registry
from . import replay
from .validator import Constraint

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

CUISINES = ["Italian", "Mexican", "Indian", "Chinese", "American", "Korean", "French", "Thai", "Japanese", "Spanish", "British"]
FLAVORS = ["spicy", "tangy", "sweet", "sweet_savory"]
DIETS = [("vegan", "Vegan"), ("vegetarian", "Vegetarian"), ("jain", "Jain"), ("dairy_free", "Dairy-free"), ("gluten_free", "Gluten-free")]

PRESETS = [
    {"name": "Late-night fried rice", "pantry": ["rice", "eggs", "kimchi", "soy sauce", "green onions", "sesame oil", "garlic"], "constraint": {"kind": "none", "value": ""}},
    {"name": "Vegan trap (chicken + tofu)", "pantry": ["chicken", "tofu", "broccoli", "rice", "garlic", "spinach", "tomato"], "constraint": {"kind": "vegan", "value": ""}},
    {"name": "Italian-ish pantry raid", "pantry": ["rice", "chickpeas", "spinach", "yogurt", "garlic", "tomato", "cheese"], "constraint": {"kind": "cuisine", "value": "Italian"}},
    {"name": "No cheese, please", "pantry": ["pasta", "tomato", "garlic", "basil", "mozzarella", "spinach", "mushroom"], "constraint": {"kind": "exclude", "value": "cheese"}},
]


def _suggestions() -> list[str]:
    p = ROOT / "data" / "pantry_vocab.json"
    if not p.exists():
        return []
    raw = json.loads(p.read_text())
    clean = [x for x in raw if "." not in x and "(" not in x and 2 < len(x) < 28]
    return clean


REGISTRY = build_registry()
SUGGESTIONS = _suggestions()

# --- tiny per-IP rate limit so a public deploy can't burn the API budget -----
RATE_MAX = int(os.getenv("DORMCHEF_RATE_MAX", "20"))
RATE_WINDOW = int(os.getenv("DORMCHEF_RATE_WINDOW_S", "600"))
_hits: dict[str, deque[float]] = defaultdict(deque)


def _rate_limit(ip: str, provider_is_real: bool) -> None:
    if not provider_is_real:
        return
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_MAX:
        raise HTTPException(429, f"Rate limit: {RATE_MAX} real-model runs per {RATE_WINDOW // 60} min. Try the Demo provider or wait.")
    q.append(now)


app = FastAPI(title="dorm-chef", version="0.1.0")


class ConstraintIn(BaseModel):
    kind: str = "none"
    value: str = Field("", max_length=40)

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        ok = {"none", "cuisine", "flavor", "low_calorie", "exclude", "weird_combination", *[d for d, _ in DIETS]}
        if v not in ok:
            raise ValueError(f"unknown constraint kind {v!r}")
        return v


class CookIn(BaseModel):
    pantry: list[str] = Field(..., min_length=2, max_length=15)
    constraint: ConstraintIn = ConstraintIn()
    provider: str = "demo"
    max_rounds: int = Field(3, ge=1, le=4)

    @field_validator("pantry")
    @classmethod
    def _pantry(cls, v: list[str]) -> list[str]:
        out: list[str] = []
        for item in v:
            item = " ".join(item.strip().lower().split())
            if not item or len(item) > 40 or any(ch in item for ch in "\n\r{}<>"):
                raise ValueError("invalid ingredient")
            if item not in out:
                out.append(item)
        if len(out) < 2:
            raise ValueError("need at least 2 distinct ingredients")
        return out


@app.get("/api/config")
def config() -> dict:
    providers = [
        {"id": p.id, "label": p.label, "available": p.available(), "real_model": p.is_real_model}
        for p in REGISTRY.values()
    ]
    default = next((p["id"] for p in providers if p["available"] and p["real_model"]), "demo")
    return {
        "providers": providers,
        "default_provider": default,
        "cuisines": CUISINES,
        "flavors": FLAVORS,
        "diets": [{"id": i, "label": l} for i, l in DIETS],
        "presets": PRESETS,
        "suggestions": SUGGESTIONS,
        "limits": {"min_pantry": 2, "max_pantry": 15},
    }


@app.post("/api/cook")
async def cook(body: CookIn, request: Request):
    provider = REGISTRY.get(body.provider)
    if provider is None:
        raise HTTPException(400, f"Unknown provider {body.provider!r}")
    if not provider.available():
        raise HTTPException(400, f"Provider '{provider.label}' isn't configured on this server (missing API key / deployment).")
    if body.constraint.kind in ("cuisine", "flavor", "exclude") and not body.constraint.value.strip():
        raise HTTPException(400, f"Constraint '{body.constraint.kind}' needs a value.")

    ip = request.client.host if request.client else "unknown"
    _rate_limit(ip, provider.is_real_model)

    req = CookRequest(
        pantry=body.pantry,
        constraint=Constraint(body.constraint.kind, body.constraint.value.strip().lower() if body.constraint.kind == "exclude" else body.constraint.value.strip()),
        max_rounds=body.max_rounds,
    )

    async def events():
        async for ev in run_agent(req, provider, judge=provider):
            yield f"data: {json.dumps(ev)}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _replay_guard(fn, *a):
    try:
        return fn(*a)
    except replay.ReplayMissing as e:
        raise HTTPException(503, str(e))
    except KeyError:
        raise HTTPException(404, "Unknown experiment or sample")


@app.get("/api/replay")
def replay_overview() -> dict:
    return _replay_guard(replay.overview)


@app.get("/api/replay/audit")
def replay_audit() -> dict:
    return _replay_guard(replay.audit)


@app.get("/api/replay/agent-example")
def replay_agent_example() -> dict:
    return _replay_guard(replay.agent_example)


@app.get("/api/replay/{exp_id}/samples")
def replay_samples(exp_id: str) -> list:
    return _replay_guard(replay.sample_index, exp_id)


@app.get("/api/replay/{exp_id}/{sample_id}")
def replay_sample(exp_id: str, sample_id: int) -> dict:
    return _replay_guard(replay.sample_detail, exp_id, sample_id)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
