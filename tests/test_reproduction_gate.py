"""The reproduction gate scores a finished run against the pre-registered targets.

It never runs a method — that is Colab's job. This is the CPU half: read shards, compute
I-AUROC per category, compare, write the verdict. Keeping it separate is what lets the
pass/fail logic be tested at all.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from reproduction_gate import main  # noqa: E402

from vlmab.eval.store import ResultStore  # noqa: E402

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
TARGETS = Path(__file__).resolve().parents[1] / "configs" / "reproduction" / "patchcore_ref.yaml"


def _shard(store, category, i_auroc, dataset="mvtec_ad", method="patchcore_ref", n=20):
    """A shard whose image scores realise `i_auroc` exactly.

    I-AUROC is the fraction of (normal, anomalous) pairs ranked correctly, so with `half`
    of each there are `half**2` pairs and the target is `misses = (1 - auroc) * half**2`
    inverted ones. Demoting an anomaly below every normal costs exactly `half` pairs, and one
    placed below `r` normals costs exactly `r` -- so the count is built, not approximated.
    An earlier version demoted whole anomalies only and silently produced 0.90 when asked for
    0.99, which is precisely the resolution the percentage-scale test needs.
    """
    half = n // 2
    misses = round((1.0 - i_auroc) * half * half)
    full, remainder = divmod(misses, half)
    rows = [{"image_path": f"/n/{i}.png", "label": 0, "image_score": float(i),
             "split": "test", "mask_path": None, "map_path": None} for i in range(half)]
    for i in range(half):
        if i < full:
            score = -1.0 - i                      # below every normal: `half` inverted pairs
        elif i == full and remainder:
            score = (half - remainder) - 0.5      # below exactly `remainder` normals
        else:
            score = float(half + i)               # above every normal
        rows.append({"image_path": f"/a/{i}.png", "label": 1, "image_score": score,
                     "split": "test", "mask_path": None, "map_path": None})
    store.write(dataset, method, category, rows, {"seed": 0, "commit": "abc1234"})


def test_the_fixture_realises_the_auroc_it_is_asked_for(tmp_path):
    """The fixture is doing arithmetic, so it gets its own test: a fixture that misses its
    target would make every tolerance assertion below meaningless."""
    from vlmab.eval.aggregate import image_metrics

    store = ResultStore(tmp_path)
    # With half=10 of each class there are 100 pairs, so the realisable grid is 0.01 apart.
    # Values off that grid (0.985) round to the nearest point and are not a fixture bug.
    for want in (1.0, 0.99, 0.98, 0.9, 0.5):
        _shard(store, f"c{want}", want)
        got = image_metrics(pd.read_parquet(
            store.path_for("mvtec_ad", "patchcore_ref", f"c{want}", 0)))["i_auroc"]
        assert abs(got - want) < 1e-9, f"asked {want}, got {got}"


def _run(store, per_category, **kw):
    for cat, auroc in per_category.items():
        _shard(store, cat, auroc, **kw)


def _all_at(value):
    return {c: value for c in CATEGORIES}


def test_a_perfect_run_passes_and_reports_every_category(tmp_path):
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(1.0))
    out = tmp_path / "report.md"
    assert main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)]) == 0
    text = out.read_text()
    assert "PASS" in text
    for cat in CATEGORIES:
        assert cat in text


def test_a_run_far_below_the_published_mean_fails(tmp_path):
    """0.50 mean against a published 99.0 must fail, and must say so in the exit code so a
    notebook cell cannot print a red verdict and carry on."""
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(0.5))
    out = tmp_path / "report.md"
    assert main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)]) == 1
    assert "FAIL" in out.read_text()


def test_the_comparison_is_on_the_published_percentage_scale(tmp_path):
    """Our metrics are 0..1 and the targets are percentages. Comparing them unconverted would
    make every real run fail by ~98 points, or -- worse, if inverted -- pass on nothing."""
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(0.99))
    out = tmp_path / "report.md"
    assert main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)]) == 0
    text = out.read_text()
    assert "99.0" in text          # the published mean, as printed in the paper
    assert "0.99" not in text.replace("0.990", "")   # never the raw fraction


def test_a_missing_category_refuses_rather_than_averaging_what_is_there(tmp_path):
    """A mean over 14 categories is not the number the paper printed."""
    results = tmp_path / "shards"
    partial = _all_at(1.0)
    partial.pop("zipper")
    _run(ResultStore(results), partial)
    with pytest.raises(ValueError, match="zipper"):
        main(["--results", str(results), "--targets", str(TARGETS),
              "--out", str(tmp_path / "r.md")])


def test_an_unexpected_category_refuses(tmp_path):
    results = tmp_path / "shards"
    extra = _all_at(1.0)
    extra["screw_v2"] = 1.0
    _run(ResultStore(results), extra)
    with pytest.raises(ValueError, match="screw_v2"):
        main(["--results", str(results), "--targets", str(TARGETS),
              "--out", str(tmp_path / "r.md")])


def test_the_wrong_dataset_refuses(tmp_path):
    """Scoring an MVTec AD 2 run against MVTec AD classic's published numbers would compare
    two different benchmarks and print a verdict either way."""
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(1.0), dataset="mvtec_ad2")
    with pytest.raises(ValueError, match="mvtec_ad2"):
        main(["--results", str(results), "--targets", str(TARGETS),
              "--out", str(tmp_path / "r.md")])


def test_the_wrong_method_refuses(tmp_path):
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(1.0), method="winclip")
    with pytest.raises(ValueError, match="winclip"):
        main(["--results", str(results), "--targets", str(TARGETS),
              "--out", str(tmp_path / "r.md")])


def test_pooled_seeds_refuse(tmp_path):
    """Two seeds under one root would score each image twice as independent samples."""
    results = tmp_path / "shards"
    store = ResultStore(results)
    _run(store, _all_at(1.0))
    for cat in CATEGORIES:
        shard = store.path_for("mvtec_ad", "patchcore_ref", cat, 0)
        other = pd.read_parquet(shard)
        other["seed"] = 1
        other.to_parquet(store.path_for("mvtec_ad", "patchcore_ref", cat, 1), index=False)
    with pytest.raises(ValueError, match="pools 2 seeds"):
        main(["--results", str(results), "--targets", str(TARGETS),
              "--out", str(tmp_path / "r.md")])


def test_the_report_records_the_source_and_the_commit(tmp_path):
    """A verdict with no provenance cannot be audited later."""
    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(1.0))
    out = tmp_path / "report.md"
    main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)])
    text = out.read_text()
    assert "arXiv:2106.08265" in text
    assert "abc1234" in text
    assert "seed" in text.lower()


def test_the_secondary_target_is_scored_but_never_gates(tmp_path):
    """VisA is reported with its caveat; a miss there must not produce a failing exit code."""
    results = tmp_path / "shards"
    visa = {f"obj{i}": 0.5 for i in range(12)}
    _run(ResultStore(results), visa, dataset="visa")
    out = tmp_path / "visa.md"
    code = main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out),
                 "--which", "secondary"])
    assert code == 0
    text = out.read_text()
    assert "secondary" in text.lower() and "does not gate" in text.lower()
    assert "92.4" in text


def test_a_mean_that_passes_while_a_category_collapses_is_flagged(tmp_path):
    """The situation the per-category targets exist for. The verdict stays PASS — the gate is
    on the mean (protocol §2) — but a reader who stops at the verdict must still be told."""
    results = tmp_path / "shards"
    scores = _all_at(0.99)
    scores["screw"] = 0.90          # 7 points below its published 97.0
    _run(ResultStore(results), scores)
    out = tmp_path / "r.md"
    assert main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)]) == 0
    text = out.read_text()
    assert "PASS" in text
    assert "screw" in text and "outside" in text
    assert "not an even reproduction" in text


def test_an_even_reproduction_carries_no_warning(tmp_path):
    """The flag must not fire on a clean run, or it becomes noise nobody reads.

    "Even" means each category near *its own* published value, which is not the same as every
    category at one number: the published values span 96.0 to 100.0, so a flat 99.0 everywhere
    is genuinely uneven and the first version of this test asserted the opposite.
    """
    import yaml

    with open(TARGETS) as fh:
        published = yaml.safe_load(fh)["published_per_category"]
    results = tmp_path / "shards"
    # Snap each to the fixture's 0.01 grid; every resulting delta is under half a point.
    _run(ResultStore(results), {c: round(v / 100.0, 2) for c, v in published.items()})
    out = tmp_path / "r.md"
    main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)])
    text = out.read_text()
    assert "outside" not in text and "⚠️" not in text


def test_the_tolerance_boundary_is_inclusive(tmp_path):
    """Protocol §2 says "within ±1.0". A run landing exactly on the boundary passes, and
    pinning that here stops the comparison drifting to `<` on a later edit."""
    results = tmp_path / "shards"
    # Published mean 99.0; 98.0 measured is a delta of exactly -1.00.
    _run(ResultStore(results), _all_at(0.98))
    out = tmp_path / "r.md"
    assert main(["--results", str(results), "--targets", str(TARGETS), "--out", str(out)]) == 0
    assert "-1.00, tolerance ±1.0" in out.read_text()


def test_a_refusal_exits_2_not_1_so_it_is_not_read_as_a_failed_gate(tmp_path):
    """0 PASS, 1 FAIL, 2 refused. A run that could not be scored is not a method that missed
    its target, and a caller that treats them alike would record a FAIL that never happened."""
    import subprocess

    results = tmp_path / "shards"
    _run(ResultStore(results), _all_at(1.0), method="winclip")     # method mismatch
    script = Path(__file__).resolve().parents[1] / "scripts" / "reproduction_gate.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--results", str(results),
         "--targets", str(TARGETS), "--out", str(tmp_path / "r.md")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 2
    assert "refusing to score this run" in proc.stderr
    assert "winclip" in proc.stderr
    assert "Traceback" not in proc.stderr
