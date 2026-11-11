"""The dorm-chef agent: draft -> verify -> (judge) -> revise.

The verifier is the project's own evaluation rubric turned into a runtime tool,
so a model that scores 83% all_pass offline can be pushed higher at inference
time by catching and repairing its own failures.

`run_agent` is an async generator of JSON-serialisable events, which the server
streams to the browser as Server-Sent Events so the UI can show the agent
thinking in real time.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import AsyncIterator

from .providers import Provider
from .validator import Constraint, JUDGE_KINDS, Report, Check, feedback_for, verify

MAX_ROUNDS_CAP = 4


@dataclass
class CookRequest:
    pantry: list[str]
    constraint: Constraint = field(default_factory=Constraint)
    max_rounds: int = 3
    temperature: float = 0.7


def build_user_prompt(req: CookRequest) -> str:
    """Same wording as the training/eval prompts in data/."""
    style, extra = req.constraint.prompt_phrase()
    return (
        f"Create a quick, student-friendly {style}recipe using these ingredients: "
        f"{', '.join(req.pantry)}.{extra} Keep it under 15 minutes with 3-7 clear steps."
    )


# Condensed from the judge rubric in the experiment-2 evaluation notebook.
JUDGE_RULES = {
    "cuisine": "The dish must reflect the cuisine's flavor profiles, techniques, or naming conventions. A generic dish that ignores the cuisine does NOT count.",
    "flavor": "The recipe must clearly reflect that flavor profile using the provided ingredients. A neutral dish that happens to use the ingredients does NOT count.",
    "low_calorie": "The ingredient list has pairs of high/low-calorie alternatives for the same role. The recipe must prefer the lower-calorie option in each pair and avoid cream sauces, heavy dressings and fried components when a lighter alternative exists.",
    "weird_combination": "The recipe should use as many provided ingredients as reasonably possible to form a cohesive dish. Skipping ingredients that would make the dish incoherent is expected. Mark false only if it ignores most ingredients without good reason or is clearly inedible.",
}

JUDGE_SYSTEM = (
    "You are a strict evaluator of college-student recipes. Decide ONLY whether the recipe follows the "
    "recipe constraint tag, using this rule:\n{rule}\n\n"
    'Return ONLY JSON: {{"constraint_followed": <true|false>, "reason": "<one short sentence>"}}'
)


def _parse_json(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        m = re.search(r"\{.*\}", cleaned, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


async def judge_style(judge: Provider, req: CookRequest, recipe: str) -> Check:
    rule = JUDGE_RULES[req.constraint.kind]
    user = (
        f"Ingredient list: {', '.join(req.pantry)}\n\nRecipe:\n{recipe}\n\n"
        f"Recipe constraint tag: {req.constraint.tag}\n\nEvaluate and return JSON."
    )
    try:
        raw = await judge.complete(JUDGE_SYSTEM.format(rule=rule), user, temperature=0.0)
    except Exception as e:  # judge failure must never kill the run
        return Check("constraint_followed", None, f"Judge unavailable ({type(e).__name__})", method="judge")
    obj = _parse_json(raw)
    if not obj or "constraint_followed" not in obj:
        return Check("constraint_followed", None, "Judge returned unparseable output", method="judge")
    return Check(
        "constraint_followed",
        bool(obj["constraint_followed"]),
        str(obj.get("reason", ""))[:200] or "judged",
        method="judge",
    )


def _score(report: Report) -> int:
    """Lower is better (number of failed checks)."""
    return len(report.failures)


async def run_agent(
    req: CookRequest,
    provider: Provider,
    judge: Provider | None = None,
) -> AsyncIterator[dict]:
    """Yield trace events. Never raises: provider errors become an `error` event."""
    rounds = max(1, min(req.max_rounds, MAX_ROUNDS_CAP))
    user_prompt = build_user_prompt(req)
    messages: list[dict] = [{"role": "user", "content": user_prompt}]

    yield {
        "type": "start",
        "provider": provider.label,
        "real_model": provider.is_real_model,
        "prompt": user_prompt,
        "max_rounds": rounds,
    }

    best: tuple[int, str, Report] | None = None

    for attempt in range(1, rounds + 1):
        yield {"type": "attempt", "n": attempt}
        draft = ""
        try:
            async for piece in provider.stream(messages, temperature=req.temperature):
                draft += piece
                yield {"type": "delta", "n": attempt, "text": piece}
        except Exception as e:
            yield {"type": "error", "message": f"{provider.label} failed: {type(e).__name__}: {str(e)[:240]}"}
            return
        draft = draft.strip()
        if not draft:
            yield {"type": "error", "message": f"{provider.label} returned an empty response"}
            return

        report = verify(draft, req.pantry, req.constraint)

        # Fuzzy constraints go to the LLM judge when a real model is available.
        if req.constraint.kind in JUDGE_KINDS and judge is not None and judge.is_real_model:
            jc = await judge_style(judge, req, draft)
            report.checks = [c for c in report.checks if c.name != "constraint_followed"] + [jc]

        yield {"type": "check", "n": attempt, "report": report.to_dict()}

        if best is None or _score(report) < best[0]:
            best = (_score(report), draft, report)

        if not report.failures:
            yield {
                "type": "final",
                "recipe": draft,
                "passed": True,
                "attempts": attempt,
                "report": report.to_dict(),
                "verified_by_judge": any(c.method == "judge" and c.passed for c in report.checks),
            }
            return

        if attempt == rounds:
            break

        fb = feedback_for(report)
        yield {"type": "revise", "n": attempt, "feedback": fb}
        messages += [{"role": "assistant", "content": draft}, {"role": "user", "content": fb}]

    assert best is not None
    _, recipe, report = best
    yield {
        "type": "final",
        "recipe": recipe,
        "passed": False,
        "attempts": rounds,
        "report": report.to_dict(),
        "verified_by_judge": False,
    }
