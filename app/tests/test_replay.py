import pytest
from fastapi.testclient import TestClient

from app import replay
from app.validator import constraint_from_tag


@pytest.fixture(scope="module")
def client():
    from app.server import app
    return TestClient(app)


def test_overview_matches_published_headline_numbers(client):
    j = client.get("/api/replay").json()
    exp = {e["id"]: e for e in j["experiments"]}
    assert exp["exp1"]["n"] == 400 and exp["exp2"]["n"] == 200
    m2 = {m["id"]: m["summary"] for m in exp["exp2"]["models"]}
    # README: constrained-sft 0.810, constrained-dpo 0.835
    assert m2["sft_hard"]["all_pass"] == pytest.approx(0.810, abs=0.001)
    assert m2["dpo_hard"]["all_pass"] == pytest.approx(0.835, abs=0.001)
    m1 = {m["id"]: m["summary"] for m in exp["exp1"]["models"]}
    assert m1["base"]["all_pass"] == pytest.approx(0.9825, abs=0.001)


def test_sample_index_and_detail(client):
    idx = client.get("/api/replay/exp2/samples").json()
    assert len(idx) == 200 and {"id", "kind", "passes", "differs"} <= set(idx[0])
    d = client.get("/api/replay/exp2/81").json()
    assert d["kind"] == "vegan" and len(d["outputs"]) == 5
    o = d["outputs"][0]
    assert {"response", "judge", "verifier", "disagree"} <= set(o)


def test_unknown_ids(client):
    assert client.get("/api/replay/nope/samples").status_code == 404
    assert client.get("/api/replay/exp1/99999").status_code == 404


def test_verifier_tracks_judge_on_structural_checks():
    a = replay.audit()["agreement"]
    # guards against regressions in the verifier: measured 0.99 / 0.97 / 0.91
    assert a["num_steps_in_range"]["rate"] >= 0.97
    assert a["cook_time_in_range"]["rate"] >= 0.94
    assert a["ingredient_compliance"]["rate"] >= 0.88


def test_audit_flags_judge_missing_constraint():
    au = replay.audit()
    assert au["constraint_missing_in_judge_reason"]["rows"] > 100
    assert len(au["diet_exclusion_rows"]) == 5


def test_every_saved_tag_parses():
    seen = {s["constraint"] for s in replay._exp("exp2")["samples"]}
    assert all(constraint_from_tag(t).kind != "none" for t in seen if t)
