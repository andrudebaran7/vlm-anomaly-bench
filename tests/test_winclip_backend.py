"""The parts of the WinCLIP backend CPU can reach: the paper's templates and the patch that installs them.

The backend itself needs anomalib, torch, CLIP weights and a GPU, so it is never constructed here.
What IS testable is the decision protocol §3 v0.2.19 records — anomalib 2.6.0 ships 21 templates
(two of the paper's missing, one listed twice) and the backend replaces them with the paper's 22 —
and the guard that refuses to proceed if the replacement did not yield 154/88 prompts.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from vlmab.methods.winclip_backend import PAPER_TEMPLATES, apply_paper_templates

ROOT = Path(__file__).resolve().parents[1]

#: anomalib 2.6.0's `prompting.TEMPLATES`, as read from the PyPI wheel and observed on Colab
#: 2026-10-03. Reproduced here only so the patch can be exercised without anomalib installed.
ANOMALIB_2_6_0_TEMPLATES = [
    "a cropped photo of the {}.",
    "a close-up photo of a {}.",
    "a close-up photo of the {}.",
    "a bright photo of a {}.",
    "a bright photo of the {}.",
    "a dark photo of the {}.",
    "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of the {}.",
    "a blurry photo of a {}.",
    "a photo of a {}.",
    "a photo of the {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of the {} for visual inspection.",
    "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.",
    "a photo of a {} for anomaly detection.",
]


def _fake_prompting():
    """A stand-in for `anomalib.models.image.winclip.prompting` with the same shape: module-level
    lists that `create_prompt_ensemble` reads at CALL time, which is what makes patching them work."""
    mod = SimpleNamespace(
        NORMAL_STATES=["{}", "flawless {}", "perfect {}", "unblemished {}",
                       "{} without flaw", "{} without defect", "{} without damage"],
        ANOMALOUS_STATES=["damaged {}", "{} with flaw", "{} with defect", "{} with damage"],
        TEMPLATES=list(ANOMALIB_2_6_0_TEMPLATES),
    )

    def create_prompt_ensemble(class_name="object"):
        normal = [t.format(s.format(class_name)) for s in mod.NORMAL_STATES for t in mod.TEMPLATES]
        anomalous = [t.format(s.format(class_name)) for s in mod.ANOMALOUS_STATES for t in mod.TEMPLATES]
        return normal, anomalous

    mod.create_prompt_ensemble = create_prompt_ensemble
    return mod


def test_the_paper_lists_22_distinct_templates_with_one_slot_each():
    assert len(PAPER_TEMPLATES) == 22
    assert len(set(PAPER_TEMPLATES)) == 22
    assert all(t.count("{}") == 1 and t.endswith(".") for t in PAPER_TEMPLATES)


def test_the_paper_templates_come_in_a_and_the_pairs():
    """Figure 6 (c) gives every template once with "a" and once with "the". That structure is what
    anomalib's typo broke ("... jpeg corrupted photo of the" twice, "of a" never), so it is the
    property that would catch the same mistake made again here."""
    def partner(t):
        return t.replace(" of a ", " of the ") if " of a " in t else t.replace(" of the ", " of a ")

    for t in PAPER_TEMPLATES:
        assert partner(t) != t, t
        assert partner(t) in PAPER_TEMPLATES, f"{t!r} has no a/the partner"


def test_the_paper_templates_are_anomalibs_plus_exactly_the_two_it_lacks():
    """Nothing else is changed: the correction is the minimal one the protocol describes."""
    assert set(PAPER_TEMPLATES) - set(ANOMALIB_2_6_0_TEMPLATES) == {
        "a cropped photo of a {}.",
        "a jpeg corrupted photo of a {}.",
    }
    assert set(ANOMALIB_2_6_0_TEMPLATES) <= set(PAPER_TEMPLATES)


def test_the_patch_turns_anomalibs_147_84_into_the_papers_154_88():
    mod = _fake_prompting()
    before = mod.create_prompt_ensemble("vial")
    assert (len(before[0]), len(before[1])) == (147, 84)

    counts = apply_paper_templates(mod)

    assert counts == {"normal": 154, "anomaly": 88}
    normal, anomalous = mod.create_prompt_ensemble("vial")
    assert (len(normal), len(anomalous)) == (154, 88)
    assert len(set(normal)) == 154 and len(set(anomalous)) == 88


def test_the_patch_is_in_place_so_an_alias_of_the_list_sees_it_too():
    """Slice assignment, not rebinding: any code holding a reference to the original list object
    must see the paper's templates, or the patch would be correct in one place and not another."""
    mod = _fake_prompting()
    alias = mod.TEMPLATES
    apply_paper_templates(mod)
    assert alias is mod.TEMPLATES
    assert alias == list(PAPER_TEMPLATES)


def test_the_patch_is_idempotent():
    mod = _fake_prompting()
    apply_paper_templates(mod)
    assert apply_paper_templates(mod) == {"normal": 154, "anomaly": 88}


def test_the_patch_refuses_when_the_state_words_are_not_the_papers():
    """The state words are deliberately left as anomalib ships them, on the strength of a reading
    that they match the paper. If an anomalib build ever changes them, the 154/88 check fails and
    the run stops here instead of scoring a different ensemble under the paper's name."""
    mod = _fake_prompting()
    mod.NORMAL_STATES.append("pristine {}")
    with pytest.raises(RuntimeError, match="154"):
        apply_paper_templates(mod)


def test_the_module_imports_no_gpu_library_at_module_scope():
    """CI has no torch, anomalib or open_clip; every such import must be function-local."""
    src = (ROOT / "src" / "vlmab" / "methods" / "winclip_backend.py").read_text()
    banned = {"torch", "anomalib", "open_clip", "lightning", "torchvision"}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert not banned & set(names), f"module-scope import of {names}"


def test_the_config_records_the_template_decision_the_backend_implements():
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "methods" / "winclip.yaml").read_text())
    assert cfg["templates"] == "paper_22"
    assert cfg["prompt_counts_paper"] == {"normal": 154, "anomaly": 88, "templates": 22}
    assert cfg["pretrained_resolved"] == "laion400m_e31"
    assert cfg["anomalib_version"] == "2.6.0"
