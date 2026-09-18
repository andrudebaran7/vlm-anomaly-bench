"""The pre-registered reproduction targets must stay internally consistent with their source.

These are not tests of our code — they are tests of a transcription. A typo in a published
number silently moves a pre-registered gate, which is exactly the kind of drift protocol §2
exists to prevent.
"""
from pathlib import Path

import yaml

TARGETS = Path(__file__).resolve().parents[1] / "configs" / "reproduction" / "patchcore_ref.yaml"


def _load():
    with open(TARGETS) as fh:
        return yaml.safe_load(fh)


def test_the_per_category_numbers_average_to_the_published_mean():
    """Table S1 prints both the per-category row and its Avg. If our transcription of the 15
    values does not reproduce the printed 99.0, one of the 16 numbers was copied wrong."""
    cfg = _load()
    per_cat = cfg["published_per_category"]
    assert len(per_cat) == 15, "MVTec AD has 15 categories"
    mean = sum(per_cat.values()) / len(per_cat)
    assert abs(mean - cfg["gate"]["published_mean"]) < 0.1, (
        f"per-category mean {mean:.3f} does not reproduce the published "
        f"{cfg['gate']['published_mean']}"
    )


def test_the_gate_matches_the_coreset_ratio_the_repo_actually_runs():
    """`PatchCore-10` means 10% coreset subsampling. A gate taken from that row while the
    method runs a different ratio would compare two different methods."""
    with open(TARGETS.parent.parent / "methods" / "patchcore_ref.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["coreset_sampling_ratio"] == 0.1
    assert "PatchCore-10" in _load()["gate"]["source"]


def test_the_gate_is_mvtec_ad_classic_and_visa_is_only_secondary():
    """Protocol §2 v0.2.12. Swapping these would restore the gate the amendment removed."""
    cfg = _load()
    assert cfg["gate"]["dataset"] == "mvtec_ad"
    assert cfg["secondary"]["dataset"] == "visa"
    assert cfg["gate"]["metric"] == cfg["secondary"]["metric"] == "i_auroc"
    assert cfg["gate"]["tolerance"] == 1.0


def test_every_gate_number_carries_its_source():
    cfg = _load()
    assert "arXiv:2106.08265" in cfg["gate"]["source"]
    assert "arXiv:2207.14315" in cfg["secondary"]["source"]
    assert cfg["secondary"]["caveat"].strip()


def test_the_number_of_seeds_is_pre_registered():
    """Protocol §6 asks for three seeds reported as mean ± std. The gate enforces that count,
    so the count has to be written down before the run rather than read off it."""
    assert _load()["n_seeds"] == 3


# --- WinCLIP -----------------------------------------------------------------------------
# Two gating blocks rather than one gate and one secondary check, because WinCLIP's own paper
# (arXiv:2303.14814, CVPR 2023) postdates the VisA dataset and reports both benchmarks at the
# configuration this repo runs. Protocol §2 v0.2.14.

WINCLIP = TARGETS.parent / "winclip.yaml"


def _winclip():
    with open(WINCLIP) as fh:
        return yaml.safe_load(fh)


def test_winclip_per_category_numbers_average_to_their_published_means():
    """Tables 10 and 16 print the per-category values; Table 1 prints their means. If our
    transcription of the 15 + 12 values does not reproduce 91.8 and 78.1, a cell was copied
    wrong -- and a mistyped gate is one nobody would notice from the verdict."""
    cfg = _winclip()
    for block, n, published in (("gate_mvtec_ad", 15, 91.8), ("gate_visa", 12, 78.1)):
        per_cat = cfg[block]["per_category"]
        assert len(per_cat) == n, f"{block} should hold {n} categories"
        mean = sum(per_cat.values()) / len(per_cat)
        assert abs(mean - published) < 0.1, (
            f"{block}: per-category mean {mean:.3f} does not reproduce the published {published}"
        )
        assert cfg[block]["published_mean"] == published


def test_both_winclip_blocks_gate():
    """The whole point of the WinCLIP entry. Demoting either to a non-gating check would
    silently remove a criterion its own paper supports (protocol §2 v0.2.14)."""
    cfg = _winclip()
    assert cfg["gate_mvtec_ad"]["gates"] is True
    assert cfg["gate_visa"]["gates"] is True
    assert cfg["gate_mvtec_ad"]["dataset"] == "mvtec_ad"
    assert cfg["gate_visa"]["dataset"] == "visa"
    assert cfg["gate_mvtec_ad"]["metric"] == cfg["gate_visa"]["metric"] == "i_auroc"
    assert cfg["gate_mvtec_ad"]["tolerance"] == cfg["gate_visa"]["tolerance"] == 1.0


def test_winclip_targets_are_the_zero_shot_ones_at_the_backbone_the_repo_runs():
    """Table 1 has 0-shot, 1-, 2- and 4-shot blocks for the same method name. Taking a k-shot
    row would gate our zero-shot adapter against WinCLIP+."""
    cfg = _winclip()
    for block in ("gate_mvtec_ad", "gate_visa"):
        assert "0-shot" in cfg[block]["source"]
        assert "arXiv:2303.14814" in cfg[block]["source"]
    with open(WINCLIP.parent.parent / "methods" / "winclip.yaml") as fh:
        method = yaml.safe_load(fh)
    assert method["k_shot"] == 0
    assert method["zero_shot"] is True
    # The paper's default is LAION-400M CLIP with ViT-B/16+ at 240^2 (Section 5, Appendix A).
    assert method["backbone"] == "ViT-B-16-plus-240"
    assert method["scales"] == [2, 3]          # small-scale 2x2, mid-scale 3x3 in ViT patches


def test_the_specific_states_ablation_is_recorded_as_not_the_target():
    """Table 7's VisA 78.9 beats the headline 78.1 and would be the tempting number to chase.
    It comes from per-category defect words, which protocol §3 forbids us to write."""
    cfg = _winclip()
    assert cfg["not_the_target"]["visa_specific_states"] == 78.9
    assert cfg["gate_visa"]["published_mean"] < cfg["not_the_target"]["visa_specific_states"]


def test_winclip_pre_registers_how_the_seed_question_gets_settled():
    """Protocol §6 is conditional ("where any stochasticity exists") and whether this adapter
    has any is unknown. The config has to carry the deciding procedure, not just a count, or
    the answer becomes a choice made after seeing a result."""
    cfg = _winclip()
    assert cfg["n_seeds"] == 3
    text = WINCLIP.read_text()
    assert "bit-identical" in text and "--all-seeds" in text
