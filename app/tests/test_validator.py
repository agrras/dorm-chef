from app.validator import Constraint, verify, count_steps, estimate_minutes

GOOD = """**Kimchi Fried Rice** (12 min)

1. Heat cooking oil in a pan and sauté minced garlic.
2. Add rice and kimchi, stir-fry 3-4 minutes.
3. Add soy sauce and sesame oil.
4. Push rice aside, scramble the eggs, then combine.
5. Top with green onions and serve."""
PANTRY = ["rice", "eggs", "kimchi", "soy sauce", "green onions", "sesame oil", "garlic"]


def names(rep, passed):
    return {c.name for c in rep.checks if c.passed is passed}


def test_good_recipe_passes_all():
    rep = verify(GOOD, PANTRY)
    assert rep.hard_pass and not rep.failures
    assert rep.num_steps == 5 and rep.cook_time_min == 12


def test_invented_ingredient_caught():
    bad = GOOD.replace("serve.", "serve with parmesan.")
    rep = verify(bad, PANTRY)
    assert "ingredient_compliance" in names(rep, False)
    assert "parmesan" in rep.failures[0].detail


def test_staples_and_plurals_allowed():
    txt = "**Eggs** (5 min)\n1. Melt butter.\n2. Crack the egg into the pan with salt and pepper.\n3. Cook in water-free pan 3 minutes."
    assert verify(txt, ["egg", "rice"]).hard_pass


def test_pantry_item_superset_matches():
    txt = "**Wrap** (8 min)\n1. Warm the tortilla.\n2. Add chicken.\n3. Roll."
    assert verify(txt, ["flour tortilla", "chicken breast"]).hard_pass


def test_too_many_steps():
    txt = "**x** (10 min)\n" + "\n".join(f"{i}. Stir rice." for i in range(1, 9))
    rep = verify(txt, ["rice", "eggs"])
    assert "num_steps_in_range" in names(rep, False)


def test_too_few_steps():
    rep = verify("**x** (10 min)\n1. Cook rice.\n2. Eat.", ["rice", "eggs"])
    assert "num_steps_in_range" in names(rep, False)


def test_time_over_limit_and_missing_is_unknown():
    over = verify("**x** (25 min)\n1. Boil rice.\n2. Eat.\n3. Wash.", ["rice", "eggs"])
    assert "cook_time_in_range" in names(over, False)
    none = verify("**x**\n1. Boil rice.\n2. Eat.\n3. Wash.", ["rice", "eggs"])
    assert "cook_time_in_range" in names(none, None)  # unknown, not a failure


def test_time_summed_from_steps_when_no_header():
    txt = "**x**\n1. Boil rice 8-10 minutes.\n2. Fry eggs 3 minutes.\n3. Serve."
    assert estimate_minutes(txt) == 11  # lower bounds: 8 + 3


def test_vegan_trap():
    pantry = ["chicken", "tofu", "broccoli", "rice", "garlic"]
    cheat = "**Bowl** (10 min)\n1. Pan-fry the chicken and tofu.\n2. Add broccoli and garlic.\n3. Serve over rice."
    rep = verify(cheat, pantry, Constraint("vegan"))
    assert "constraint_followed" in names(rep, False)
    ok = "**Bowl** (10 min)\n1. Pan-fry the tofu.\n2. Add broccoli and garlic.\n3. Serve over rice."
    assert verify(ok, pantry, Constraint("vegan")).hard_pass


def test_jain_blocks_root_veg():
    pantry = ["onion", "garlic", "tomato", "rice", "peas"]
    txt = "**x** (10 min)\n1. Saute the onion.\n2. Add tomato and peas.\n3. Serve over rice."
    assert "constraint_followed" in names(verify(txt, pantry, Constraint("jain")), False)


def test_gluten_free_ok_with_rice_noodles_and_gf_soy():
    pantry = ["rice noodles", "gluten-free soy sauce", "broccoli"]
    txt = "**x** (10 min)\n1. Boil the rice noodles 4 minutes.\n2. Steam broccoli.\n3. Toss with gluten-free soy sauce."
    rep = verify(txt, pantry, Constraint("gluten_free"))
    assert not rep.failures, [c.detail for c in rep.failures]


def test_dairy_free_yogurt_ok():
    pantry = ["dairy-free yogurt", "rice", "spinach"]
    txt = "**x** (10 min)\n1. Cook rice.\n2. Wilt spinach.\n3. Top with dairy-free yogurt."
    assert not verify(txt, pantry, Constraint("dairy_free")).failures


def test_exclusion():
    pantry = ["pasta", "tomato", "mozzarella", "garlic"]
    txt = "**x** (10 min)\n1. Boil pasta.\n2. Saute garlic and tomato.\n3. Top with mozzarella cheese."
    assert "constraint_followed" in names(verify(txt, pantry, Constraint("exclude", "cheese")), False)


def test_style_constraints_deferred_to_judge():
    rep = verify(GOOD, PANTRY, Constraint("cuisine", "Korean"))
    c = next(c for c in rep.checks if c.name == "constraint_followed")
    assert c.passed is None and c.method == "judge"
    assert rep.hard_pass  # does not block on its own


def test_tags_match_repo_conventions():
    assert Constraint("cuisine", "Italian").tag == "Italian-style cuisine"
    assert Constraint("vegan").tag == "vegan_adversarial"
    assert Constraint("exclude", "soy sauce").tag == "exclude_soy_sauce"
    assert Constraint("flavor", "spicy").tag == "spicy_style"


def test_count_steps_ignores_non_steps():
    assert count_steps("**t**\n*(Note: 2 eggs)*\n1. a\n2. b") == 2
