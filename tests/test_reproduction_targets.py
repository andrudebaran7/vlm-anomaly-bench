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
