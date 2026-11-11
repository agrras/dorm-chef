"""Deterministic verifier for dorm-chef recipes.

This is a programmatic port of the *hard constraints* in the project's
LLM-as-judge rubric (see python-notebooks/experiment2 - more constrained/
Evaluation_on_Test_Set_Run_Constrained.ipynb):

    ingredient_compliance   only provided ingredients (+ pantry staples)
    num_steps_in_range      3-7 numbered steps
    cook_time_in_range      <= 15 minutes
    constraint_followed     diet / exclusion rules that can be checked by lexicon

Style constraints (cuisine, flavor, low-calorie, weird-combination) cannot be
checked reliably with string matching, so they are reported as "needs_judge"
and handled by the LLM judge in agent.py when a real provider is configured.

The agent uses these checks as its verifier: draft -> verify -> revise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

MIN_STEPS = 3
MAX_STEPS = 7
MAX_MINUTES = 15

# The judge treats these as always allowed.
STAPLES = {"water", "salt", "pepper", "oil", "cooking oil", "butter", "black pepper"}

# Foods we can detect in a recipe to catch *invented* ingredients. Deliberately
# a list of whole food words (not the 700+ noisy pantry strings in the data).
FOOD_LEXICON = sorted(
    {
        # proteins
        "chicken", "beef", "pork", "turkey", "ham", "bacon", "sausage", "salami", "tuna",
        "salmon", "shrimp", "fish", "egg", "tofu", "tempeh", "lamb", "paneer", "chorizo",
        # dairy
        "cheese", "mozzarella", "cheddar", "parmesan", "feta", "ricotta", "yogurt",
        "milk", "cream", "butter", "ghee",
        # produce
        "tomato", "onion", "garlic", "spinach", "broccoli", "carrot", "potato", "mushroom",
        "cucumber", "lettuce", "cabbage", "kale", "zucchini", "eggplant", "avocado",
        "corn", "peas", "pea", "celery", "cauliflower", "beet", "radish", "ginger",
        "jalapeno", "pepper", "bell pepper", "scallion", "green onion", "basil",
        "cilantro", "parsley", "mint", "dill", "lemon", "lime", "apple", "banana",
        "mango", "pineapple", "orange", "berries", "olives", "pumpkin",
        # grains / carbs
        "rice", "pasta", "noodles", "ramen", "bread", "tortilla", "oats", "quinoa",
        "couscous", "barley", "flour", "crackers", "bagel", "cornstarch",
        # legumes
        "beans", "chickpeas", "lentils", "hummus", "edamame",
        # sauces / flavor
        "soy sauce", "sesame oil", "kimchi", "salsa", "mayo", "mustard", "ketchup",
        "honey", "sugar", "vinegar", "hot sauce", "curry powder", "garam masala",
        "mayonnaise", "cumin", "cinnamon", "paprika", "oregano", "chili", "peanut butter", "coconut milk",
        "sesame", "gochujang", "pesto", "marinara", "nuts", "almonds", "cashews",
    },
    key=lambda s: (-len(s), s),  # longest first so "soy sauce" wins over "sauce"
)

MEAT = {"chicken", "beef", "pork", "turkey", "ham", "bacon", "sausage", "salami", "lamb", "chorizo", "meatball", "steak", "jamon", "prosciutto", "pepperoni"}
SEAFOOD = {"tuna", "salmon", "shrimp", "fish", "sardine", "anchovy", "crab", "clam", "prawn", "fish sauce"}
EGGS = {"egg", "eggs", "mayo", "mayonnaise"}
DAIRY = {"cheese", "mozzarella", "cheddar", "parmesan", "feta", "ricotta", "yogurt", "milk", "cream", "butter", "ghee", "paneer", "whey", "custard"}
ROOT_VEG = {"onion", "garlic", "potato", "carrot", "radish", "beet", "beetroot", "ginger", "shallot"}
GLUTEN = {"bread", "pasta", "ramen", "couscous", "barley", "crackers", "flour tortilla", "tortilla", "seitan", "noodles", "bagel", "soy sauce", "flour", "croutons", "bun", "spaghetti", "oats"}
# Gluten-free substitutes: if the *recipe* names these we should not flag them.
GLUTEN_SAFE_PHRASES = ("gluten-free", "gluten free", "rice noodles", "rice tortilla", "corn tortilla", "tamari")
ANIMAL = MEAT | SEAFOOD | EGGS | DAIRY | {"honey", "gelatin"}

DIET_RULES: dict[str, tuple[str, set[str]]] = {
    "vegan": ("vegan", ANIMAL),
    "vegetarian": ("vegetarian", MEAT | SEAFOOD),
    "jain": ("Jain", MEAT | SEAFOOD | {"egg", "eggs"} | ROOT_VEG),
    "dairy_free": ("dairy-free", DAIRY),
    "gluten_free": ("gluten-free", GLUTEN),
}

# Constraint kinds the verifier can fully check vs. ones that need a judge.
LEXICON_KINDS = set(DIET_RULES) | {"exclude"}
JUDGE_KINDS = {"cuisine", "flavor", "low_calorie", "weird_combination"}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def _singular(w: str) -> str:
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("oes"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w


def _tokens(s: str) -> set[str]:
    return {_singular(t) for t in re.findall(r"[a-z]+", _norm(s))}


def _mentions(text: str, term: str) -> bool:
    """Whole-word (plural-tolerant) mention of `term` in `text`."""
    t = re.escape(term)
    return re.search(rf"\b{t}(?:e?s)?\b", text) is not None


@dataclass
class Constraint:
    """What the user asked for beyond the base recipe rules."""

    kind: str = "none"  # none|cuisine|flavor|vegan|vegetarian|jain|dairy_free|gluten_free|low_calorie|exclude|weird_combination
    value: str = ""  # e.g. "Italian", "spicy", "cheese"

    @property
    def tag(self) -> str:
        """The tag string the repo's judge expects."""
        if self.kind == "cuisine":
            return f"{self.value}-style cuisine"
        if self.kind == "flavor":
            return f"{self.value}_style"
        if self.kind in DIET_RULES:
            return f"{self.kind}_adversarial"
        if self.kind == "exclude":
            return f"exclude_{self.value.replace(' ', '_')}"
        if self.kind in ("low_calorie", "weird_combination"):
            return self.kind
        return ""

    def prompt_phrase(self) -> tuple[str, str]:
        """(style words before 'recipe', sentence after ingredients) — mirrors training prompts."""
        if self.kind == "cuisine":
            return f"{self.value}-style ", ""
        if self.kind == "flavor":
            return f"{self.value.replace('_', '-')}-style ", ""
        if self.kind == "vegan":
            return "vegan ", ""
        if self.kind == "vegetarian":
            return "vegetarian ", ""
        if self.kind == "jain":
            return "Jain-friendly ", ""
        if self.kind == "dairy_free":
            return "dairy-free ", ""
        if self.kind == "gluten_free":
            return "gluten-free ", ""
        if self.kind == "low_calorie":
            return "low-calorie ", ""
        if self.kind == "exclude":
            return "", f" Do not include {self.value}."
        return "", ""


def constraint_from_tag(tag: str) -> Constraint:
    """Inverse of Constraint.tag: parse the tags used in the repo's test data."""
    tag = (tag or "").strip()
    if not tag:
        return Constraint()
    if tag.endswith("-style cuisine"):
        return Constraint("cuisine", tag[: -len("-style cuisine")])
    if tag.endswith("_adversarial"):
        kind = tag[: -len("_adversarial")]
        return Constraint(kind) if kind in DIET_RULES else Constraint()
    if tag.endswith("_style"):
        return Constraint("flavor", tag[: -len("_style")])
    if tag.startswith("exclude_"):
        return Constraint("exclude", tag[len("exclude_"):].replace("_", " "))
    if tag in ("low_calorie", "weird_combination"):
        return Constraint(tag)
    return Constraint()


@dataclass
class Check:
    name: str
    passed: bool | None  # None = could not be checked here (needs judge)
    detail: str = ""
    method: str = "rule"  # rule | judge

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    num_steps: int = 0
    cook_time_min: int | None = None

    @property
    def hard_pass(self) -> bool:
        """All rule-checked constraints pass (judge-only ones don't block)."""
        return all(c.passed is not False for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.passed is False]

    def to_dict(self) -> dict:
        return {
            "checks": [c.to_dict() for c in self.checks],
            "num_steps": self.num_steps,
            "cook_time_min": self.cook_time_min,
            "hard_pass": self.hard_pass,
        }


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
_STEP_RE = re.compile(r"^\s*(?:\*\*)?\s*(\d{1,2})\s*[.)]\s+\S", re.M)


def count_steps(recipe: str) -> int:
    return len(_STEP_RE.findall(recipe))


_RANGE = r"(\d+)(?:\s*(?:-|–|to)\s*(\d+))?"


def stated_minutes(recipe: str) -> int | None:
    """A total time the recipe states explicitly, e.g. a '(12 min)' title or 'Total time: 10-12 minutes'."""
    for head in [l for l in recipe.splitlines() if l.strip()][:3]:
        m = re.search(rf"\(\s*(?:total\s*)?~?{_RANGE}\s*(?:min|mins|minutes)\b", head, re.I)
        if m:
            return int(m.group(2) or m.group(1))
    m = re.search(rf"total[^.\n]*?~?{_RANGE}\s*(?:min|mins|minutes)", recipe, re.I)
    if m:
        return int(m.group(2) or m.group(1))
    return None


def summed_minutes(recipe: str) -> int | None:
    """Rough estimate from the steps: sum of the *lower* bound of each stated duration.

    Lower bounds because steps overlap in real life (rice simmers while you chop), and
    on the saved eval set this tracked the LLM judge best (95% vs 91% for upper bounds).
    """
    total, found = 0, False
    for line in recipe_body(recipe).splitlines():
        if not _STEP_RE.match(line):
            continue
        for a, _b in re.findall(rf"{_RANGE}\s*(?:min|mins|minutes)\b", line, re.I):
            found = True
            total += int(a)
    return total if found else None


def estimate_minutes(recipe: str) -> int | None:
    t = stated_minutes(recipe)
    return t if t is not None else summed_minutes(recipe)


ESTIMATE_SLACK = 2  # an estimate (not a stated total) may exceed the limit by this much before we call it a failure


# Words that mean "this is a suggestion, not part of the recipe". The judge does not
# count these as used ingredients, so neither do we.
_OPTIONAL = r"(?:optional(?:ly)?|if available|if you have|if desired|if you like|if wanted|if preferred|if needed|if on hand|or to taste)"
_OPT_CLAUSE = re.compile(rf"[^,;.()!:]*\b{_OPTIONAL}\b[^,;.()!]*", re.I)
_OIL_PHRASES = re.compile(r"\b(?:extra[- ]virgin |extra virgin )?(?:olive|vegetable|canola|cooking|sunflower) oil\b", re.I)

# Foods that count as "the same ingredient" for compliance (judge: "minor descriptor differences are fine").
GROUPS = [
    {"cheese", "mozzarella", "cheddar", "parmesan", "feta", "ricotta", "gouda", "brie", "halloumi", "gruyere", "swiss", "provolone"},
    {"pasta", "spaghetti", "penne", "macaroni", "fettuccine", "linguine", "fusilli", "noodles", "ramen"},
    {"bread", "baguette", "toast", "ciabatta", "sourdough", "bun", "loaf", "sandwich"},
    {"mayo", "mayonnaise"},
    {"pepper", "bell pepper", "capsicum"},
    {"scallion", "green onion"},
    {"chili", "jalapeno", "chile"},
]


def recipe_body(recipe: str) -> str:
    """The part of a response that is the recipe: ingredient line + numbered steps.

    Drops chatty preambles ("chicken is not vegan, but...") and trailing notes, which the
    judge ignores. Falls back to the whole text if no numbered steps are found.
    """
    keep: list[str] = []
    in_step = False
    for line in recipe.splitlines():
        if _STEP_RE.match(line):
            in_step = True
            keep.append(line)
        elif re.match(r"^\s*(?:\*\*|#+\s*)?\s*ingredients?\b", line, re.I):
            keep.append(line)
        elif in_step and line.strip() and re.match(r"^\s{2,}|^\s*[-*\u2022]\s", line):
            keep.append(line)  # indented continuation / sub-bullet of a step
        elif in_step and not line.strip():
            continue
        elif in_step and re.match(r"^\s*(?:\*+\s*)?(?:note|tip|notes|enjoy|serve)", line, re.I):
            in_step = False
    body = "\n".join(keep)
    return body if count_steps(body) else recipe


def _strip_optional(text: str) -> str:
    return _OPT_CLAUSE.sub(" ", text)


def _group_of(term: str) -> int | None:
    for i, g in enumerate(GROUPS):
        if term in g:
            return i
    return None


def invented_ingredients(recipe: str, pantry: list[str]) -> list[str]:
    """Foods named in the recipe that are neither in the pantry nor a staple."""
    text = _norm(_strip_optional(_OIL_PHRASES.sub("oil", recipe_body(recipe))))
    pantry_tokens: set[str] = set()
    pantry_text = _norm(" ".join(pantry))
    for p in pantry:
        pantry_tokens |= _tokens(p)
    staple_tokens: set[str] = set()
    for st in STAPLES:
        staple_tokens |= _tokens(st)
    pantry_groups = {_group_of(f) for f in FOOD_LEXICON if _mentions(pantry_text, f)} - {None}

    bad: list[str] = []
    consumed = text
    for food in FOOD_LEXICON:
        if not _mentions(consumed, food):
            continue
        ft = _tokens(food)
        # blank out so "soy sauce" is not also matched as e.g. "sauce"
        consumed = re.sub(rf"\b{re.escape(food)}(?:e?s)?\b", " ", consumed)
        if ft <= pantry_tokens or ft <= staple_tokens:
            continue
        if _group_of(food) in pantry_groups:
            continue
        if food == "pepper" and ("black" in text or "salt" in text):
            continue
        bad.append(food)
    return sorted(set(bad))


def forbidden_hits(recipe: str, banned: set[str]) -> list[str]:
    text = _norm(_strip_optional(recipe_body(recipe)))
    safe = any(p in text for p in GLUTEN_SAFE_PHRASES)
    hits = []
    for term in sorted(banned, key=len, reverse=True):
        if not _mentions(text, term):
            continue
        # "gluten-free soy sauce" / "rice noodles" should not trip the gluten rule
        if safe and term in GLUTEN:
            window = re.findall(rf"(?:gluten[- ]free|rice|corn|tamari)\s+\w*\s*{re.escape(term)}", text)
            if window:
                continue
        # "dairy-free yogurt", "non-dairy cream", "plant-based butter"
        if term in DAIRY and re.search(rf"(?:dairy[- ]free|non[- ]dairy|vegan|plant[- ]based)\s+\w*\s*{re.escape(term)}", text):
            continue
        hits.append(term)
    return hits


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def verify(recipe: str, pantry: list[str], constraint: Constraint | None = None) -> Report:
    constraint = constraint or Constraint()
    rep = Report()

    # 1. ingredient compliance
    bad = invented_ingredients(recipe, pantry)
    rep.checks.append(
        Check(
            "ingredient_compliance",
            not bad,
            "Only pantry items + staples used" if not bad else f"Not in pantry: {', '.join(bad)}",
        )
    )

    # 2. steps
    n = count_steps(recipe)
    rep.num_steps = n
    ok = MIN_STEPS <= n <= MAX_STEPS
    rep.checks.append(Check("num_steps_in_range", ok, f"{n} numbered steps (need {MIN_STEPS}-{MAX_STEPS})"))

    # 3. time
    stated = stated_minutes(recipe)
    mins = stated if stated is not None else summed_minutes(recipe)
    rep.cook_time_min = mins
    if mins is None:
        rep.checks.append(Check("cook_time_in_range", None, "No total time stated — add one like (12 min)", method="estimate"))
    elif stated is not None:
        rep.checks.append(Check("cook_time_in_range", mins <= MAX_MINUTES, f"{mins} min stated (limit {MAX_MINUTES})"))
    else:
        rep.checks.append(
            Check("cook_time_in_range", mins <= MAX_MINUTES + ESTIMATE_SLACK, f"~{mins} min estimated from steps (limit {MAX_MINUTES})", method="estimate")
        )

    # 4. constraint
    k = constraint.kind
    if k in DIET_RULES:
        label, banned = DIET_RULES[k]
        hits = forbidden_hits(recipe, banned)
        rep.checks.append(
            Check(
                "constraint_followed",
                not hits,
                f"No {label}-violating items" if not hits else f"Not {label}: uses {', '.join(hits)}",
            )
        )
    elif k == "exclude" and constraint.value:
        hit = _mentions(_norm(_strip_optional(recipe_body(recipe))), _norm(constraint.value))
        rep.checks.append(
            Check(
                "constraint_followed",
                not hit,
                f"'{constraint.value}' avoided" if not hit else f"Recipe still uses '{constraint.value}'",
            )
        )
    elif k in JUDGE_KINDS:
        rep.checks.append(
            Check("constraint_followed", None, f"Style check ({constraint.tag}) is scored by the LLM judge", method="judge")
        )
    return rep


def feedback_for(report: Report) -> str:
    """Turn failed checks into a revision instruction for the model."""
    lines = [f"- {c.name}: {c.detail}" for c in report.failures]
    return (
        "Your recipe failed these checks:\n"
        + "\n".join(lines)
        + "\n\nRewrite the full recipe so every check passes. Keep a title line with the total time, e.g. "
        "**Name** (12 min), then 3-7 numbered steps. Use only the provided ingredients "
        "(water, salt, pepper, cooking oil and butter are always allowed)."
    )
