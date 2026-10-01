"""Finding the right target block in a reproduction config, instead of hardcoding its name.

The names differ per method: patchcore_ref has `gate` and `secondary`, winclip has
`gate_mvtec_ad` and `gate_visa`. Both configs state `dataset:` inside each block, so the block
is derivable and a per-method map is unnecessary.
"""
from pathlib import Path

import pytest

from vlmab.eval.targets import block_for

ROOT = Path(__file__).resolve().parents[1]


def test_patchcores_visa_block_is_its_non_gating_secondary():
    assert block_for(ROOT / "configs/reproduction/patchcore_ref.yaml", "visa") == "secondary"


def test_patchcores_classic_block_is_its_gate():
    assert block_for(ROOT / "configs/reproduction/patchcore_ref.yaml", "mvtec_ad") == "gate"


def test_winclip_has_a_gating_block_for_each_dataset():
    cfg = ROOT / "configs/reproduction/winclip.yaml"
    assert block_for(cfg, "mvtec_ad") == "gate_mvtec_ad"
    assert block_for(cfg, "visa") == "gate_visa"


def test_the_not_the_target_block_is_never_returned(tmp_path):
    """WinCLIP's Table 7 ablation is HIGHER than the real target. It declares no dataset, and a
    lookup that returned it would hand the gate the tempting number."""
    cfg = tmp_path / "t.yaml"
    cfg.write_text(
        "method: winclip\n"
        "gate_visa:\n  dataset: visa\n  published_mean: 78.1\n"
        "not_the_target:\n  published_mean: 78.9\n"
    )
    assert block_for(cfg, "visa") == "gate_visa"


def test_a_dataset_with_no_block_is_refused_by_name(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("method: winclip\ngate_visa:\n  dataset: visa\n")
    with pytest.raises(ValueError) as exc:
        block_for(cfg, "mvtec_ad")
    assert "mvtec_ad" in str(exc.value) and "gate_visa" in str(exc.value)


def test_two_blocks_for_one_dataset_are_refused_rather_than_guessed(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("method: m\na:\n  dataset: visa\nb:\n  dataset: visa\n")
    with pytest.raises(ValueError) as exc:
        block_for(cfg, "visa")
    assert "'a'" in str(exc.value) and "'b'" in str(exc.value)
