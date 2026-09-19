"""SAA+'s prompts are the method, so the table that holds them is load-bearing.

Protocol §3 v0.2.9 pre-registers them verbatim from the official repo, because SAA+'s paper
cites a supplementary for them that does not exist (arXiv:2305.10724v1 is the only version and
ends at its references). These are tests of a transcription and of a format, not of our code.

The format part is the unusual one. `SAA/model.py`'s `set_property_text_prompts` does not parse
language -- it indexes whitespace-separated tokens by position. A re-wrapped string, a collapsed
double space or a dropped trailing space changes a threshold to whatever token now sits at that
index, silently, producing a complete and plausible run at parameters nobody chose.
"""
from pathlib import Path

import pytest
import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
PROMPTS = CONFIGS / "methods" / "saa_prompts.yaml"
METHOD = CONFIGS / "methods" / "saa.yaml"


def _prompts():
    with open(PROMPTS) as fh:
        return yaml.safe_load(fh)


def _method():
    with open(METHOD) as fh:
        return yaml.safe_load(fh)


# The five positions SAA/model.py reads, as (index, name, parser).
POSITIONAL_FIELDS = [
    (5, "object_number", int),
    (6, "similar", str),
    (7, "object_prompt", str),
    (12, "k_mask", int),
    (19, "defect_area_threshold", float),
]


@pytest.mark.parametrize("category", sorted(_prompts()["visa_categories"]))
def test_every_property_constraint_still_parses_at_the_positions_the_repo_reads(category):
    """The guard the positional format needs. This would fail on a string that had been
    re-wrapped, re-spaced or tidied -- which is the only way these values can be corrupted,
    because nothing else about them is validated anywhere."""
    text = _prompts()["visa_categories"][category]["property_constraints"]
    tokens = text.split(" ")
    assert len(tokens) > 19, (
        f"{category}: only {len(tokens)} tokens; SAA/model.py reads index 19"
    )
    for index, name, parse in POSITIONAL_FIELDS:
        try:
            parse(tokens[index])
        except ValueError:
            pytest.fail(
                f"{category}: token {index} ({name}) is {tokens[index]!r}, which does not parse "
                f"as {parse.__name__}. The string has been reformatted since it was copied."
            )


def test_the_property_constraints_keep_their_trailing_space():
    """Byte-exactness, at the one byte an editor is most likely to eat. It is not decorative:
    the repo's strings end in a space and a stripped one shifts nothing today but makes the
    stored value differ from the source it claims to be verbatim from."""
    for category, entry in _prompts()["visa_categories"].items():
        text = entry["property_constraints"]
        assert text.endswith(" "), f"{category}: trailing space stripped"
        assert "  " not in text.replace(". ", ". "), f"{category}: unexpected double space"


def test_the_visa_objects_are_the_twelve_the_loader_and_the_other_targets_use():
    """A prompt filed under a key the loader never asks for is a prompt that never runs, and
    the failure mode is a silent fallback rather than an error."""
    prompts = set(_prompts()["visa_categories"])
    with open(CONFIGS / "reproduction" / "adaclip.yaml") as fh:
        canonical = set(yaml.safe_load(fh)["gate_visa"]["per_category"])
    assert prompts == canonical, f"differs: {prompts ^ canonical}"


def test_every_prompt_pair_is_the_repos_two_item_shape():
    """The repo stores [prompt, filtered phrase]. Flattening it would lose which half is which,
    and the filtered phrase is what the property prompt's object token is matched against."""
    for category, entry in _prompts()["visa_categories"].items():
        assert entry["prompt_pairs"], f"{category}: no prompts"
        for pair in entry["prompt_pairs"]:
            assert isinstance(pair, list) and len(pair) == 2, f"{category}: {pair!r}"
            assert all(isinstance(half, str) and half for half in pair), f"{category}: {pair!r}"


def test_no_mvtec_ad2_category_has_an_invented_prompt():
    """Protocol §3 v0.2.9's whole point. The repo publishes nothing for MVTec AD 2, and the one
    forbidden response is to write a plausible prompt ourselves."""
    categories = _prompts()["categories"]
    assert len(categories) == 8
    for name, entry in categories.items():
        assert entry["prompt"] == "NOT_PUBLISHED_IN_REPO", name
        assert entry["property_constraints"] == "NOT_PUBLISHED_IN_REPO", name


def test_the_resolved_coverage_agrees_with_the_table():
    """The count and the table are two records of one fact, and they are edited separately."""
    method = _method()
    assert method["prompt_coverage"] == "0_of_8"
    published = [n for n, e in _prompts()["categories"].items()
                 if e["prompt"] != "NOT_PUBLISHED_IN_REPO"]
    assert not published, f"config says 0_of_8 but the table publishes {published}"


def test_both_files_name_the_branch_that_actually_holds_saa_plus():
    """`master` holds the vanilla SAA demo, whose published VisA Fp is 12.76 against SAA+'s
    27.07. A clone that does not name the branch runs a different method and would not say so."""
    assert _method()["branch"] == "SAA-plus"
    assert _prompts()["source_branch"] == "SAA-plus"
    assert _method()["branch_head_read_for_provenance"] == _prompts()["source_commit"]


def test_the_dead_prompt_table_in_the_source_is_named():
    """`visa_parameters.py` defines `official_prompts` as well, and it is imported nowhere. It is
    the larger and more official-looking of the two tables in that file, so a later reader
    re-deriving these values needs to be told which one the code uses."""
    assert _prompts()["unused_table_in_source"] == "official_prompts"


def test_every_visa_entry_cites_the_lines_it_was_copied_from():
    """Condition 1 of the file's own preamble."""
    for category, entry in _prompts()["visa_categories"].items():
        src = entry["copied_from"]
        assert "visa_parameters.py" in src and "manual_prompts" in src, f"{category}: {src!r}"
        assert "property_prompts" in src, f"{category}: {src!r}"
