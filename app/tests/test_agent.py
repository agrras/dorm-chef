import asyncio, json
import pytest
from fastapi.testclient import TestClient

from app.agent import CookRequest, run_agent, build_user_prompt
from app.providers import DemoProvider
from app.validator import Constraint


def collect(req, provider=None, judge=None):
    async def go():
        return [e async for e in run_agent(req, provider or DemoProvider(), judge=judge)]
    return asyncio.run(go())


def test_prompt_matches_training_format():
    req = CookRequest(["rice", "eggs"], Constraint("cuisine", "Italian"))
    assert build_user_prompt(req) == (
        "Create a quick, student-friendly Italian-style recipe using these ingredients: rice, eggs. "
        "Keep it under 15 minutes with 3-7 clear steps."
    )
    ex = build_user_prompt(CookRequest(["pasta", "tomato"], Constraint("exclude", "cheese")))
    assert "Do not include cheese." in ex and ex.endswith("3-7 clear steps.")


def test_demo_fails_then_repairs_exclusion():
    req = CookRequest(["pasta", "tomato", "garlic", "mozzarella", "spinach"], Constraint("exclude", "mozzarella"), max_rounds=3)
    evs = collect(req)
    types = [e["type"] for e in evs]
    assert types[0] == "start" and types[-1] == "final"
    assert "revise" in types, "first sloppy draft should trigger a revision"
    fin = evs[-1]
    assert fin["passed"] and fin["attempts"] >= 2
    assert "mozzarella" not in fin["recipe"].lower()


def test_single_round_returns_best_attempt_unpassed():
    req = CookRequest(["pasta", "tomato", "garlic", "mozzarella", "spinach"], Constraint("exclude", "mozzarella"), max_rounds=1)
    fin = collect(req)[-1]
    assert fin["type"] == "final" and fin["passed"] is False and fin["attempts"] == 1


def test_provider_error_becomes_event_not_exception():
    class Boom(DemoProvider):
        async def stream(self, messages, temperature=0.7):
            raise RuntimeError("kaboom")
            yield ""
    evs = collect(CookRequest(["a", "b"]), Boom())
    assert evs[-1]["type"] == "error" and "kaboom" in evs[-1]["message"]


def test_style_judge_failure_feeds_revision():
    class Judge(DemoProvider):
        is_real_model = True
        calls = 0
        async def complete(self, system, user, temperature=0.0, max_tokens=400):
            Judge.calls += 1
            ok = Judge.calls > 1
            return json.dumps({"constraint_followed": ok, "reason": "generic dish" if not ok else "reads Korean"})
    req = CookRequest(["rice", "egg", "cabbage"], Constraint("cuisine", "Korean"), max_rounds=3)
    j = Judge()
    evs = collect(req, DemoProvider(), judge=j)
    assert any(e["type"] == "revise" and "constraint_followed" in e["feedback"] for e in evs)
    assert evs[-1]["passed"] and evs[-1]["verified_by_judge"]


# ---- HTTP layer -----------------------------------------------------------
@pytest.fixture()
def client():
    from app.server import app
    return TestClient(app)


def sse(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ") and l != "data: {}"]


def test_config(client):
    j = client.get("/api/config").json()
    assert any(p["id"] == "demo" and p["available"] for p in j["providers"])
    assert j["default_provider"] in {p["id"] for p in j["providers"]}
    assert len(j["suggestions"]) > 100 and all("." not in s for s in j["suggestions"])


def test_cook_stream_demo(client):
    r = client.post("/api/cook", json={"pantry": ["rice", "eggs", "kimchi", "garlic"], "provider": "demo"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    evs = sse(r)
    assert evs[0]["type"] == "start" and evs[-1]["type"] == "final"


def test_validation_errors(client):
    assert client.post("/api/cook", json={"pantry": ["rice"], "provider": "demo"}).status_code == 422
    assert client.post("/api/cook", json={"pantry": ["rice", "egg"], "provider": "nope"}).status_code == 400
    assert client.post("/api/cook", json={"pantry": ["rice", "egg"], "provider": "demo", "constraint": {"kind": "wat"}}).status_code == 422
    assert client.post("/api/cook", json={"pantry": ["rice", "egg"], "provider": "demo", "constraint": {"kind": "exclude", "value": " "}}).status_code == 400
    assert client.post("/api/cook", json={"pantry": ["rice", "egg"], "provider": "claude"}).status_code in (200, 400)


def test_unconfigured_provider_rejected(client, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    r = client.post("/api/cook", json={"pantry": ["rice", "egg"], "provider": "azure_dpo"})
    assert r.status_code == 400 and "isn't configured" in r.json()["detail"]
