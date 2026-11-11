"""Build app/data/replay.json from the saved evaluation results in results/.

    python -m app.build_replay

Joins each model's stored generations + LLM-judge scores by prompt id, so the
Compare tab can show base / SFT / DPO side by side with no API calls.
Also recomputes each model's aggregate metrics from the rows and checks them
against the repo's *_summary.json files.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "results"
OUT = Path(__file__).resolve().parent / "data" / "replay.json"

E1 = RES / "experiment1"
E2 = RES / "experiment 2 constrained"
H = RES / "trained on harder constrained data"

EXPERIMENTS = [
    {
        "id": "exp1",
        "label": "Experiment 1 · standard prompts",
        "description": "400 held-out ingredient lists. Hard constraints: only given ingredients, 3–7 steps, ≤15 min.",
        "test": None,
        "models": [
            ("base", "Base gpt-4.1-nano", E1 / "base" / "base_eval.jsonl", E1 / "base" / "base_summary.json"),
            ("sft", "SFT", E1 / "sft" / "sft-alpha_eval-2.jsonl", E1 / "sft" / "sft-alpha_summary.json"),
            ("dpo", "DPO", E1 / "dpo" / "dpo-alpha_eval-2.jsonl", E1 / "dpo" / "{ECFG.tag}_summary-4.json"),
        ],
    },
    {
        "id": "exp2",
        "label": "Experiment 2 · constrained prompts",
        "description": "200 held-out prompts with a cuisine / flavor / diet / exclusion rule. Adds a fourth hard constraint: constraint_followed.",
        "test": REPO / "data" / "testing data" / "test-constrained.jsonl",
        "models": [
            ("base", "Base gpt-4.1-nano", E2 / "base" / "base_eval.jsonl", E2 / "base" / "base_summary.json"),
            ("sft", "SFT", E2 / "sft" / "sft_gamma2.0_eval.jsonl", E2 / "sft" / "sft_gamma2.0_summary.json"),
            ("dpo", "DPO", E2 / "dpo" / "dpo_gamma_eval.jsonl", E2 / "dpo" / "dpo_gamma_summary.json"),
            ("sft_hard", "SFT · harder data", H / "sft" / "constrained-sft_eval.jsonl", H / "sft" / "constrained-sft_summary.json"),
            ("dpo_hard", "DPO · harder data", H / "dpo" / "constrained-dpo_eval.jsonl", H / "dpo" / "constrained-dpo_summary.json"),
        ],
    },
]

METRICS = [
    "ingredient_compliance",
    "num_steps_in_range",
    "cook_time_in_range",
    "constraint_followed",
    "all_pass",
    "coherence",
    "student_friendly",
    "llm_score",
]


def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def user_msg(rec: dict) -> str:
    return next(m["content"] for m in rec["messages"] if m["role"] == "user")


def aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["generation_failed"] and r["evaluation"]]
    out = {"n": len(rows)}
    for k in METRICS:
        vals = [r["evaluation"][k] for r in ok if r["evaluation"].get(k) is not None]
        out[k] = round(sum(vals) / len(vals), 4) if vals else None
    return out


def main() -> None:
    result = {"experiments": []}
    for exp in EXPERIMENTS:
        constraints: dict[int, str] = {}
        if exp["test"]:
            for i, rec in enumerate(read_jsonl(exp["test"]), 1):
                constraints[i] = rec.get("constraint", "")

        models, samples = [], {}
        for mid, label, rows_path, sum_path in exp["models"]:
            rows = read_jsonl(rows_path)
            agg = aggregate(rows)
            # cross-check against the published summary
            pub = json.loads(sum_path.read_text())
            diffs = {
                k: (agg[k], pub.get(f"{k}_rate", pub.get("llm_score_avg") if k == "llm_score" else None))
                for k in METRICS
                if agg.get(k) is not None
            }
            bad = {k: v for k, v in diffs.items() if v[1] is not None and abs(v[0] - v[1]) > 0.006}
            status = "matches summary" if not bad else f"MISMATCH {bad}"
            print(f"  {exp['id']}/{mid:9} n={agg['n']:3} all_pass={agg['all_pass']}  [{status}]")
            models.append({"id": mid, "label": label, "summary": agg})

            for r in rows:
                s = samples.setdefault(
                    r["id"],
                    {"id": r["id"], "ingredients": r["ingredients"], "constraint": constraints.get(r["id"], ""), "prompt": r["prompt"]["user"], "outputs": {}},
                )
                if s["prompt"] != r["prompt"]["user"]:
                    raise SystemExit(f"prompt mismatch for id {r['id']} in {exp['id']}/{mid}")
                s["outputs"][mid] = {"response": r["response"], "judge": r["evaluation"]}

        result["experiments"].append(
            {
                "id": exp["id"],
                "label": exp["label"],
                "description": exp["description"],
                "models": models,
                "samples": [samples[k] for k in sorted(samples)],
            }
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, separators=(",", ":")))
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
