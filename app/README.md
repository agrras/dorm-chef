# dorm-chef app — a recipe agent that checks its own work

A web UI + agent loop on top of the CIS 5270 fine-tuning project.

**Agent loop:** draft → verify against the repo's judge rubric → (LLM-judge for style rules) → send failures back → revise (up to N rounds). Every step streams to the browser live over SSE.

| Check | How |
|---|---|
| only provided ingredients (+ water/salt/pepper/oil/butter) | rule-based (`validator.py`) |
| 3–7 numbered steps, ≤ 15 min | rule-based |
| vegan / vegetarian / Jain / dairy-free / gluten-free / "exclude X" | rule-based lexicons |
| cuisine, flavor, low-calorie, weird-combo | LLM judge using the notebook's rubric |

## Compare tab (no API keys needed)

Side-by-side base / SFT / DPO outputs on your saved test prompts (400 for Experiment 1, 200 for Experiment 2, plus the harder-data runs),
with the stored LLM-judge verdicts next to a rule check run live. Data comes from `app/data/replay.json`, built from `results/` by
`python -m app.build_replay` (re-run it if you add results). The Audit panel compares the rule check against the judge over all 2,200 saved generations.

## Run

```bash
pip install -r app/requirements.txt
cp app/.env.example .env   # fill in what you have; or just export vars
uvicorn app.server:app --reload      # http://localhost:8000
python -m pytest app/tests -q
```

With no keys it runs in **Demo** mode (a scripted stand-in, clearly labelled — not a model).
Set `ANTHROPIC_API_KEY` for Claude, and/or `AZURE_OPENAI_API_KEY` + `DORMCHEF_AZURE_{BASE,SFT,DPO}` to put your
fine-tuned gpt-4.1-nano deployments behind the same UI and compare them.

Public deploys: a per-IP rate limit (`DORMCHEF_RATE_MAX` / `DORMCHEF_RATE_WINDOW_S`) protects your API budget.
Never commit keys.
