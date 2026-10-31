"""The map-free figure is drawn from the committed reports (spec 2026-10-01 §5, fifth figure)."""
from pathlib import Path

import pytest

from vlmab.viz import figures, tables

REPORTS = Path(__file__).resolve().parents[1] / "results" / "mvtec_ad2" / "patchcore_ref_v0.2.21"


def _committed():
    return {c: (REPORTS / f"{c}.md").read_text() for c in tables.CATEGORY_ORDER}


def test_one_point_per_category_and_condition():
    pts = figures.points(_committed())
    assert len(pts) == 6 * 6 + 7 + 4
    assert sum(p.condition == figures.TRAIN_CONDITION for p in pts) == 8


def test_points_carry_the_facts_the_text_states():
    """§6.A/§6.C: Fabric's AU-PRO@0.05 is below random in four of six conditions, and Wall Plugs
    shift_2 sits at 0.97x (0.98x against the measured 0.0249 until 2026-10-31); `regular` is the best detection condition in no category."""
    pts = figures.points(_committed())
    fabric = [p.au_pro_005_x for p in pts if p.category == "fabric"]
    assert sum(x < 1 for x in fabric) == 4 and len(fabric) == 6
    shift_2 = next(p for p in pts if p.category == "wallplugs" and p.condition == "shift_2")
    assert round(shift_2.au_pro_005_x, 2) == 0.97
    for c in tables.CATEGORY_ORDER:
        mine = [p for p in pts if p.category == c]
        best = max(mine, key=lambda p: p.i_auroc)
        assert best.condition != figures.TRAIN_CONDITION, c


def test_category_means_are_table_twos():
    means = figures.category_means(_committed())
    assert round(means["vial"][0], 3) == 0.888
    assert round(means["can"][1], 2) == 0.61


def test_the_figure_draws(tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "f.pdf"
    figures.draw_per_condition(_committed(), out)
    assert out.stat().st_size > 1000


INPUTS = REPORTS.parents[1] / "figure_inputs"


@pytest.mark.parametrize("draw", ["draw_dataset", "draw_resize", "draw_maps"])
def test_the_image_figures_draw_from_the_preregistered_inputs(tmp_path, draw):
    """Only where results/figure_inputs/ exists: it holds MVTec AD 2 images, gitignored, so CI
    and a fresh clone skip this."""
    pytest.importorskip("matplotlib")
    if not INPUTS.is_dir():
        pytest.skip("no results/figure_inputs (copied from Drive after notebook 7.2/7.3)")
    out = tmp_path / "f.pdf"
    getattr(figures, draw)(INPUTS, out)
    assert out.stat().st_size > 1000
