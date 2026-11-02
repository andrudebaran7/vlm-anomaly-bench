"""scripts/gate_antialias_ablation.py: experiment B, the MVTec AD gate under both forward
resizes on one VM (protocol v0.2.25, 2026-11-02)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _committed(mode):
    from gate_antialias_ablation import REPORTS, parse_gate_report

    return parse_gate_report((ROOT / REPORTS[mode]).read_text())


def test_both_committed_gate_reports_parse_to_their_printed_numbers():
    old, new = _committed("aliased"), _committed("antialiased")
    assert (old["mean"], old["std"]) == ("98.02", "0.07")
    assert (new["mean"], new["std"]) == ("98.22", "0.03")
    assert old["seeds"] == {"seed0": "97.94", "seed1": "98.08", "seed2": "98.02"}
    assert len(old["categories"]) == len(new["categories"]) == 15
    assert old["categories"]["toothbrush"] == "91.48 ± 0.58"
    assert new["categories"]["toothbrush"] == "91.39 ± 0.28"


def test_a_report_compared_with_itself_is_equal_everywhere():
    from gate_antialias_ablation import compare_reports

    new = _committed("antialiased")
    rows = compare_reports(new, new)
    assert len(rows) == 1 + 3 + 15
    assert all(r["equal"] for r in rows)


def test_one_category_off_by_a_hundredth_is_the_only_mismatch():
    from gate_antialias_ablation import compare_reports

    new = _committed("antialiased")
    moved = {**new, "categories": {**new["categories"], "zipper": "97.70 ± 0.11"}}
    off = [r["what"] for r in compare_reports(moved, new) if not r["equal"]]
    assert off == ["zipper"]


def test_a_run_missing_a_category_is_refused_not_compared_short():
    from gate_antialias_ablation import compare_reports

    new = _committed("antialiased")
    short = {**new, "categories": {k: v for k, v in new["categories"].items() if k != "bottle"}}
    with pytest.raises(KeyError, match="bottle"):
        compare_reports(short, new)


def test_a_single_seed_report_is_refused():
    from gate_antialias_ablation import parse_gate_report

    with pytest.raises(ValueError, match="multi-seed"):
        parse_gate_report("Measured mean I-AUROC **98.10** against published **99.0**")


def test_the_modes_and_their_root_are_experiment_a_s_switch_under_their_own_root():
    from gate_antialias_ablation import MODES, RESULTS

    assert MODES == {"aliased": True, "antialiased": False}
    assert RESULTS.parts[:2] == ("results", "antialias_ablation")
