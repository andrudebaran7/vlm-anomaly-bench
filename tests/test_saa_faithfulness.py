"""SAA+ has no reproduction gate, so this script is what stands in its place.

Protocol §2 v0.2.17: its paper publishes no image-level number, and §2's own disqualifying test
rules out manufacturing a criterion from what is available. What replaces the gate is a direct
check that our recorded configuration still matches the repo it claims to be verbatim from.

A drift detector is only worth what its negative cases are worth, so most of what follows breaks
something on purpose and asserts that the check notices. The fixture is built from the real
values recorded in configs/methods/saa_prompts.yaml rather than invented, so a test passing here
means the same comparison would pass against the real clone.
"""
import importlib.util
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "saa_faithfulness.py"
PROMPTS = ROOT / "configs" / "methods" / "saa_prompts.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("saa_faithfulness", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def clone(tmp_path):
    """A minimal SAA-plus tree whose prompt values are the ones we actually recorded."""
    recorded = yaml.safe_load(PROMPTS.read_text())["visa_categories"]
    manual = {c: [list(p) for p in e["prompt_pairs"]] for c, e in recorded.items()}
    prop = {c: e["property_constraints"] for c, e in recorded.items()}

    repo = tmp_path / "Segment-Any-Anomaly"
    (repo / "SAA" / "prompts").mkdir(parents=True)
    (repo / "SAA" / "prompts" / "visa_parameters.py").write_text(
        f"manual_prompts = {manual!r}\n\nproperty_prompts = {prop!r}\n")
    (repo / "SAA" / "prompts" / "general_prompts.py").write_text(
        "general_anomaly_description = ['defect on {}', 'damage on {}', 'flaw on {}']\n")
    (repo / "SAA" / "hybrid_prompts.py").write_text(
        "manual_prompts = {'visa_public': None}\nproperty_prompts = {'visa_public': None}\n")
    (repo / "SAA" / "model.py").write_text(
        "\n".join(f"x = p.split(' ')[{i}]" for i in (5, 6, 7, 12, 19)) + "\n")

    subprocess.run(["git", "init", "-q", "-b", "SAA-plus", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-qm", "fixture"], check=True)
    return repo


def _run(clone, capsys, **over):
    rc = _module().main(["--repo", str(clone), *sum(([k, str(v)] for k, v in over.items()), [])])
    return rc, capsys.readouterr().out


def test_a_faithful_clone_passes_every_static_check(clone, capsys):
    rc, out = _run(clone, capsys)
    assert rc == 0, out
    assert out.count("PASS") == 5
    assert "FAIL" not in out


def test_it_says_what_it_did_not_check(clone, capsys):
    """A green tick that implies more than it checked is worse than no tick. The runtime half --
    400x400, K=5, the map stored unmodified -- is owed by the backend and must be named."""
    _, out = _run(clone, capsys)
    assert "STATIC half only" in out
    assert "400x400" in out and "owed by" in out


def test_it_catches_the_wrong_branch(clone, capsys):
    """`master` is the vanilla SAA demo, published VisA Fp 12.76 against SAA+'s 27.07. A clone
    on it runs a different method and nothing else would say so."""
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b", "master"], check=True)
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "FAIL  branch" in out and "12.76" in out


def test_it_catches_a_reformatted_property_constraint(clone, capsys):
    """The failure the positional format makes possible: re-wrapping a string moves a threshold
    to whatever token now sits at index 19, silently. Here a single extra word is inserted."""
    src = clone / "SAA" / "prompts" / "visa_parameters.py"
    src.write_text(src.read_text().replace("the image of candle", "the image of a candle", 1))
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "BYTE-FOR-BYTE" in out and "candle" in out


def test_it_catches_a_changed_prompt(clone, capsys):
    src = clone / "SAA" / "prompts" / "visa_parameters.py"
    src.write_text(src.read_text().replace("bent wire on pcb", "bent wire", 1))
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "prompt pairs differ" in out


def test_it_catches_the_positional_contract_being_dropped(clone, capsys):
    """If model.py stops reading token 19, every recorded area threshold has to be re-read --
    and the strings themselves would still look perfectly fine."""
    (clone / "SAA" / "model.py").write_text("x = p.split(' ')[5]\n")
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "no longer reads token" in out


def test_it_catches_the_dead_table_coming_back_to_life(clone, capsys):
    """`official_prompts` is the larger, more official-looking table in visa_parameters.py and is
    imported nowhere. If the wiring changed to use it, the method's inputs changed."""
    (clone / "SAA" / "hybrid_prompts.py").write_text("official_prompts = {'visa_public': None}\n")
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "dead prompt table" in out or "prompt tables" in out


def test_it_catches_a_changed_fallback_because_that_is_all_of_mvtec_ad2(clone, capsys):
    """Every one of MVTec AD 2's eight categories runs on these three lines (0/8 coverage), so a
    change here changes the whole primary evaluation."""
    (clone / "SAA" / "prompts" / "general_prompts.py").write_text(
        "general_anomaly_description = ['anomaly on {}']\n")
    rc, out = _run(clone, capsys)
    assert rc == 1
    assert "8 of 8 categories" in out


def test_a_missing_clone_exits_2_not_1(clone, capsys, tmp_path):
    """0 PASS, 1 FAIL, 2 could-not-check. A clone that was never made must never be reported as
    a clean bill of health, and `master` has no SAA/ package at all."""
    rc = _module().main(["--repo", str(tmp_path / "never-cloned")])
    assert rc == 2


def test_it_does_not_import_the_repo_it_is_checking(clone, capsys):
    """It parses with ast.literal_eval rather than runpy. A provenance check that executes the
    third-party code it is checking is not a check -- and it would also need that code's
    dependencies, which a CPU box does not have."""
    src = clone / "SAA" / "prompts" / "visa_parameters.py"
    src.write_text("import nonexistent_module_that_would_explode\n" + src.read_text())
    rc, out = _run(clone, capsys)
    assert rc == 0, out
    assert "byte-identical" in out
