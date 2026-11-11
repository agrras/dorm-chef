"""Replay data access + the verifier-vs-judge audit.

Reads app/data/replay.json (built by `python -m app.build_replay`) and runs the
rule-based verifier live on every saved generation, so the Compare tab can show
the stored LLM-judge verdict next to an independent rule check.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .validator import DIET_RULES, constraint_from_tag, feedback_for, verify

PATH = Path(__file__).parent / "data" / "replay.json"

# Judge reasons that say the judge did not have / could not apply the constraint tag.
_MISSING_CONSTRAINT = re.compile(
    r"no specific constraint|unspecified|implied constraint|implied|no constraint|no explicit|not specified|"
    r"no particular|no additional constraint|none specified",
    re.I,
)

HARD = ["ingredient_compliance", "num_steps_in_range", "cook_time_in_range", "constraint_followed"]


class ReplayMissing(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load() -> dict:
    if not PATH.exists():
        raise ReplayMissing("app/data/replay.json not found — run `python -m app.build_replay` from the repo root.")
    data = json.loads(PATH.read_text())
    for ex in data["experiments"]:
        ex["_by_id"] = {s["id"]: s for s in ex["samples"]}
        ex["_models"] = {m["id"]: m for m in ex["models"]}
    return data


def _exp(exp_id: str) -> dict:
    for ex in _load()["experiments"]:
        if ex["id"] == exp_id:
            return ex
    raise KeyError(exp_id)


def overview() -> dict:
    return {
        "experiments": [
            {
                "id": ex["id"],
                "label": ex["label"],
                "description": ex["description"],
                "n": len(ex["samples"]),
                "models": ex["models"],
            }
            for ex in _load()["experiments"]
        ]
    }


def sample_index(exp_id: str) -> list[dict]:
    ex = _exp(exp_id)
    out = []
    for s in ex["samples"]:
        c = constraint_from_tag(s["constraint"])
        passes = {m: bool(o["judge"]["all_pass"]) for m, o in s["outputs"].items() if o["judge"]}
        out.append(
            {
                "id": s["id"],
                "ingredients": s["ingredients"],
                "constraint": s["constraint"],
                "kind": c.kind,
                "passes": passes,
                "differs": len(set(passes.values())) > 1,
            }
        )
    return out


def sample_detail(exp_id: str, sample_id: int) -> dict:
    ex = _exp(exp_id)
    s = ex["_by_id"][sample_id]
    c = constraint_from_tag(s["constraint"])
    outputs = []
    for m in ex["models"]:
        o = s["outputs"].get(m["id"])
        if not o:
            continue
        rep = verify(o["response"], s["ingredients"], c)
        judge = o["judge"] or {}
        disagree = [
            ck.name
            for ck in rep.checks
            if ck.passed is not None and ck.name in judge and bool(judge[ck.name]) != ck.passed
        ]
        outputs.append(
            {
                "model": m["id"],
                "label": m["label"],
                "response": o["response"],
                "judge": judge,
                "verifier": rep.to_dict(),
                "disagree": disagree,
            }
        )
    return {
        "id": s["id"],
        "ingredients": s["ingredients"],
        "constraint": s["constraint"],
        "kind": c.kind,
        "prompt": s["prompt"],
        "outputs": outputs,
    }


@lru_cache(maxsize=1)
def audit() -> dict:
    """How well does the rule-based verifier agree with the stored LLM judge, and what does it reveal?"""
    agree: dict[str, dict] = {k: {"n": 0, "agree": 0} for k in HARD}
    rule_rows: dict[str, dict] = {}
    missing = {"rows": 0, "total": 0}

    for ex in _load()["experiments"]:
        for s in ex["samples"]:
            c = constraint_from_tag(s["constraint"])
            rule_checkable = c.kind in DIET_RULES or c.kind == "exclude"
            for mid, o in s["outputs"].items():
                j = o["judge"]
                if not j:
                    continue
                rep = verify(o["response"], s["ingredients"], c)
                for ck in rep.checks:
                    if ck.passed is None or ck.name not in j:
                        continue
                    a = agree[ck.name]
                    a["n"] += 1
                    a["agree"] += int(bool(j[ck.name]) == ck.passed)
                if ex["id"] == "exp2":
                    missing["total"] += 1
                    missing["rows"] += int(bool(_MISSING_CONSTRAINT.search(j.get("judge_reason", ""))))
                    if rule_checkable:
                        cc = next(x for x in rep.checks if x.name == "constraint_followed")
                        r = rule_rows.setdefault(mid, {"n": 0, "judge": 0, "rule": 0})
                        r["n"] += 1
                        r["judge"] += int(bool(j["constraint_followed"]))
                        r["rule"] += int(bool(cc.passed))

    exp2_models = {m["id"]: m["label"] for m in _exp("exp2")["models"]}
    return {
        "agreement": {k: {**v, "rate": round(v["agree"] / v["n"], 3) if v["n"] else None} for k, v in agree.items()},
        "constraint_missing_in_judge_reason": {**missing, "rate": round(missing["rows"] / missing["total"], 3) if missing["total"] else None},
        "diet_exclusion_rows": [
            {
                "model": mid,
                "label": exp2_models.get(mid, mid),
                "n": r["n"],
                "judge_rate": round(r["judge"] / r["n"], 3),
                "rule_rate": round(r["rule"] / r["n"], 3),
            }
            for mid, r in rule_rows.items()
        ],
    }


@lru_cache(maxsize=1)
def agent_example() -> dict:
    """A real saved failure the agent would catch, with the exact message it would send back.

    Picks the first Experiment 2 diet/exclusion prompt where a saved model broke the rule and
    BOTH the rule check and the stored judge flagged it (so the example is not a verifier quirk).
    """
    ex = _exp("exp2")
    for s in ex["samples"]:
        c = constraint_from_tag(s["constraint"])
        if c.kind not in DIET_RULES:
            continue
        for m in ex["models"]:
            o = s["outputs"].get(m["id"])
            if not o or not o["judge"] or m["id"] in ("base",):
                continue
            rep = verify(o["response"], s["ingredients"], c)
            cc = next((x for x in rep.checks if x.name == "constraint_followed"), None)
            if cc and cc.passed is False and not o["judge"]["constraint_followed"] and len(rep.failures) == 1 and len(o["response"]) < 900:
                return {
                    "sample_id": s["id"],
                    "model": m["label"],
                    "prompt": s["prompt"],
                    "response": o["response"],
                    "report": rep.to_dict(),
                    "feedback": feedback_for(rep),
                }
    raise ReplayMissing("no suitable agent example found")
