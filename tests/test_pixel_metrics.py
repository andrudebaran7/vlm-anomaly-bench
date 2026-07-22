import numpy as np
import pytest
from scipy import ndimage

from vlmab.metrics.pixel_level import _trapezoid, p_auroc, pro_thresholds, seg_f1max


def _one_region(size=100, region=40, top_left=10):
    mask = np.zeros((size, size), dtype=np.uint8)
    s = slice(top_left, top_left + region)
    mask[s, s] = 1
    return mask


def test_p_auroc_perfect_separation():
    mask = _one_region()
    assert p_auroc([mask], [mask.astype(np.float32)]) == 1.0


def test_p_auroc_inverted_prediction_is_zero():
    mask = _one_region()
    assert p_auroc([mask], [1.0 - mask.astype(np.float32)]) == 0.0


def test_p_auroc_pools_across_images():
    # Image `a` is perfectly separated; image `b` has heavy overlapping noise so its
    # own AUROC is well below 1.0. This makes pooled AUROC and the mean of per-image
    # AUROCs demonstrably different numbers, so the test can catch an implementation
    # that wrongly averages per-image scores instead of pooling pixels.
    a, b = _one_region(), _one_region(top_left=50)
    amap_a = a.astype(np.float32)
    rng = np.random.default_rng(0)
    amap_b = b.astype(np.float32) * 0.3 + rng.random(b.shape).astype(np.float32)

    # auroc(a) == 1.0, auroc(b) == 0.7517213169642857
    # -> mean of per-image AUROCs == 0.8758606584821429 (the value this test must NOT match)
    pooled = p_auroc([a, b], [amap_a, amap_b])
    assert pooled == pytest.approx(0.9379303292410714)


def test_seg_f1max_perfect_separation():
    mask = _one_region()
    assert seg_f1max([mask], [mask.astype(np.float32)]) == pytest.approx(1.0)


def test_seg_f1max_no_anomalous_pixels_is_zero():
    empty = np.zeros((20, 20), dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert seg_f1max([empty], [rng.random((20, 20))]) == 0.0


def test_p_auroc_rejects_per_image_shape_mismatch_even_with_matching_total_length():
    # mask/amap pair shapes are swapped between the two images: (10,10)+(20,20) masks
    # vs (20,20)+(10,10) amaps. Concatenated total length is 100+400 == 400+100 == 500
    # either way, so a check on total length alone would pass this silently. Each
    # individual (mask, amap) pair is shape-mismatched, which must be caught.
    masks = [np.zeros((10, 10), dtype=np.uint8), np.zeros((20, 20), dtype=np.uint8)]
    amaps = [np.zeros((20, 20), dtype=np.float32), np.zeros((10, 10), dtype=np.float32)]
    with pytest.raises(ValueError):
        p_auroc(masks, amaps)


def test_seg_f1max_beats_threshold_noise():
    mask = _one_region()
    rng = np.random.default_rng(0)
    amap = mask.astype(np.float32) + 0.4 * rng.random(mask.shape).astype(np.float32)
    assert seg_f1max([mask], [amap]) == pytest.approx(1.0)


from vlmab.metrics.pixel_level import au_pro


def test_au_pro_perfect_prediction_is_one():
    mask = _one_region()
    assert au_pro([mask], [mask.astype(np.float32)]) == pytest.approx(1.0)


def test_au_pro_weights_regions_equally_not_by_area():
    """A big region found and a tiny region missed is PRO 0.5, not ~0.99.

    Pixel-overlap scoring would give 1600/1616 here. PRO must not.
    """
    big = _one_region(size=100, region=40, top_left=10)
    tiny = _one_region(size=100, region=4, top_left=10)
    found = big.astype(np.float32)
    missed = np.zeros((100, 100), dtype=np.float32)
    assert au_pro([big, tiny], [found, missed]) == pytest.approx(0.5, abs=1e-6)


def test_au_pro_constant_map_is_zero():
    mask = _one_region()
    assert au_pro([mask], [np.full(mask.shape, 0.5, dtype=np.float32)]) == 0.0


def test_au_pro_signal_beats_noise():
    mask = _one_region()
    rng = np.random.default_rng(0)
    signal = mask.astype(np.float32) + 0.4 * rng.random(mask.shape).astype(np.float32)
    noise = rng.random(mask.shape).astype(np.float32)
    assert au_pro([mask], [signal]) > 0.8
    assert au_pro([mask], [noise]) < 0.35


def test_au_pro_tighter_fpr_limit_is_not_higher():
    mask = _one_region()
    rng = np.random.default_rng(1)
    amap = mask.astype(np.float32) + 0.4 * rng.random(mask.shape).astype(np.float32)
    assert au_pro([mask], [amap], fpr_limit=0.05) <= au_pro([mask], [amap], fpr_limit=0.3) + 1e-9


def test_au_pro_rejects_length_mismatch():
    mask = _one_region()
    with pytest.raises(ValueError):
        au_pro([mask, mask], [mask.astype(np.float32)])


def _weak_separation_fixture(seed=7, size=128):
    """The case that exposes a score-uniform threshold grid.

    Background U(0, 0.6); anomalous regions get a +U(0.3, 0.7) bump, so the two
    distributions overlap heavily and the whole FPR in (0, 0.05] window lives in a
    narrow slice of score space near 0.6. One bright outlier pixel stretches the
    score range to 1.5, which is exactly what a real anomaly map does. A grid
    uniform in *score* then spends almost all of its points on scores nothing
    reaches and lands only a handful inside the window being integrated.
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[20:40, 20:40] = 1
    mask[80:88, 80:88] = 1
    amap = rng.uniform(0.0, 0.6, size=(size, size)).astype(np.float32)
    bump = rng.uniform(0.3, 0.7, size=(size, size)).astype(np.float32)
    amap[mask > 0] += bump[mask > 0]
    amap[0, 0] = 1.5
    return [mask], [amap]


@pytest.mark.parametrize("fpr_limit, tol", [(0.05, 1e-3), (0.3, 3e-3)])
def test_au_pro_has_converged_at_the_default_threshold_count(fpr_limit, tol):
    """The reported number must not depend on `num_thresholds`.

    This is the gate on threshold *selection*. The naive-reference equivalence test
    shares `pro_thresholds` with production, so it can only catch a wrong sweep, not
    a biased grid; nothing else in this file would notice one.

    Against the old score-uniform `np.linspace(lo, hi, n + 1)[1:]` grid both cases
    fail badly: @0.05 gives 0.7369 (n=200) vs 0.8324 (n=50000), and @0.3 gives 0.8854
    vs 0.9016 -- 9.6 and 1.6 AU-PRO points of discretisation error against a protocol
    tolerance of +-1.0 point (0.01 in these units).

    With the FPR-uniform grid the residuals are 1.2e-05 (@0.05) and 1.7e-03 (@0.3),
    i.e. 0.001 and 0.17 points. The looser tolerance at @0.3 is not slack in the
    implementation: PRO(FPR) is concave with essentially all of its curvature in the
    knee just above FPR 0, and the trapezoid rule under-integrates a concave curve, so
    a grid spread evenly over the six-times-wider [0, 0.3] interval resolves that knee
    six times less finely. The residual is still 6x inside the protocol tolerance and
    ~60x smaller than what the old grid produced. Every other fixture in this file
    converges exactly (residual 0.0) at both limits.
    """
    masks, amaps = _weak_separation_fixture()
    coarse = au_pro(masks, amaps, fpr_limit=fpr_limit)
    fine = au_pro(masks, amaps, fpr_limit=fpr_limit, num_thresholds=50_000)
    assert coarse == pytest.approx(fine, abs=tol)


# --- Naive reference implementation -----------------------------------------------
#
# This is the straightforward, pre-optimisation formulation of AU-PRO: recompute
# `amap >= t` over the full image for every threshold and AND it against every
# region's full-size boolean mask. It is deliberately O(n_regions * n_thresholds *
# image_pixels) -- exactly the cost the real `au_pro` was rewritten to avoid -- and is
# kept here, independent of the production code path, so the optimised version can be
# checked against it. `structure` defaults to 8-connectivity to match the fixed
# production behaviour; the connectivity test below overrides it to demonstrate the
# old 4-connectivity bug.
#
# NOTE: threshold *selection* is deliberately shared with production
# (`pro_thresholds`), because the point of this reference is to validate the
# searchsorted reformulation of the sweep against a full-image recomputation, not to
# second-guess where the thresholds sit. It therefore does NOT independently validate
# threshold selection -- `test_au_pro_has_converged_at_the_default_threshold_count` is
# what covers that.
def _naive_au_pro(
    masks,
    amaps,
    fpr_limit: float = 0.3,
    num_thresholds: int = 200,
    structure=np.ones((3, 3), dtype=int),
) -> float:
    masks = [np.asarray(m) > 0 for m in masks]
    amaps = [np.asarray(a, dtype=np.float32) for a in amaps]

    regions = []
    for i, m in enumerate(masks):
        labelled, n = ndimage.label(m, structure=structure)
        for r in range(1, n + 1):
            regions.append((i, labelled == r))

    n_normal = int(sum((~m).sum() for m in masks))
    if not regions or n_normal == 0:
        return 0.0

    all_values = np.concatenate([a.ravel() for a in amaps])
    lo, hi = float(all_values.min()), float(all_values.max())
    if lo == hi:
        return 0.0

    normal_sorted = np.sort(np.concatenate([a[~m] for m, a in zip(masks, amaps) if (~m).any()]))
    thresholds = pro_thresholds(normal_sorted, lo, fpr_limit, num_thresholds)
    if thresholds.size == 0:
        return 0.0

    pros, fprs = [], []
    for t in thresholds:
        binaries = [a >= t for a in amaps]
        pros.append(
            float(np.mean([
                np.count_nonzero(binaries[i] & reg) / np.count_nonzero(reg)
                for i, reg in regions
            ]))
        )
        fp = sum(int(np.count_nonzero(binaries[i] & ~masks[i])) for i in range(len(masks)))
        fprs.append(fp / n_normal)

    fpr = np.asarray(fprs, dtype=np.float64)
    pro = np.asarray(pros, dtype=np.float64)
    order = np.argsort(fpr, kind="stable")
    fpr, pro = fpr[order], pro[order]

    keep = fpr <= fpr_limit
    x, y = fpr[keep], pro[keep]
    if x.size == 0:
        return 0.0
    if x[0] > 0.0:
        x = np.concatenate([[0.0], x])
        y = np.concatenate([[y[0]], y])
    if x[-1] < fpr_limit:
        x = np.concatenate([x, [fpr_limit]])
        y = np.concatenate([y, [y[-1]]])

    return float(_trapezoid(y, x) / fpr_limit)


def test_au_pro_diagonal_pixels_form_one_region_not_two():
    """Two diagonally-touching anomalous pixels must be ONE region (8-connectivity).

    The fixture is built so that treating them as one region (correct) vs. two
    (the old buggy 4-connectivity default of `ndimage.label`) yields two different
    overall AU-PRO numbers -- 0.75 vs 2/3 -- computed independently below. If
    `structure=np.ones((3, 3))` is ever dropped from `au_pro`, this test fails.

    Construction: one big, perfectly-detected 40x40 region, plus two diagonally
    adjacent single pixels where one has the same anomaly value as the big region and
    the other has none. Every relevant threshold sits at FPR == 0 (the only anomalous
    pixel not fully detected still scores below every positive threshold, and all
    background pixels score exactly 0), so the whole PRO curve is flat and the
    AU-PRO value reduces to a single, easily-hand-checked region-average.
    """
    mask = _one_region(size=100, region=40, top_left=10)
    mask[80, 80] = 1
    mask[81, 81] = 1
    amap = mask.astype(np.float32).copy()
    amap[81, 81] = 0.0

    correct_8conn = _naive_au_pro([mask], [amap], structure=np.ones((3, 3), dtype=int))
    wrong_4conn = _naive_au_pro([mask], [amap], structure=None)

    # Sanity-check the fixture actually distinguishes the two connectivities.
    assert correct_8conn == pytest.approx(0.75)
    assert wrong_4conn == pytest.approx(2 / 3)
    assert correct_8conn != pytest.approx(wrong_4conn)

    got = au_pro([mask], [amap])
    assert got == pytest.approx(correct_8conn)
    assert got != pytest.approx(wrong_4conn)


def _random_au_pro_fixture(rng, n_images, size, n_regions_per_image, region_size_range=(2, 15)):
    """Multiple images, multiple regions per image, varied region sizes."""
    masks, amaps = [], []
    for _ in range(n_images):
        mask = np.zeros((size, size), dtype=np.uint8)
        amap = (rng.random((size, size)).astype(np.float32)) * 0.5
        for _ in range(n_regions_per_image):
            h = int(rng.integers(region_size_range[0], region_size_range[1] + 1))
            w = int(rng.integers(region_size_range[0], region_size_range[1] + 1))
            h, w = min(h, size), min(w, size)
            top = int(rng.integers(0, size - h + 1))
            left = int(rng.integers(0, size - w + 1))
            mask[top:top + h, left:left + w] = 1
            amap[top:top + h, left:left + w] += (
                rng.random((h, w)).astype(np.float32) * 0.5 + 0.5
            )
        masks.append(mask)
        amaps.append(amap.astype(np.float32))
    return masks, amaps


@pytest.mark.parametrize("fpr_limit", [0.3, 0.05])
def test_au_pro_matches_naive_reference(fpr_limit):
    """The optimised au_pro must exactly reproduce the naive O(regions*thresholds*pixels)
    reference (same connectivity, same thresholds) on varied random fixtures.

    This is what makes the performance rewrite trustworthy: it is a reformulation
    proof, not a vibe check on one trivial case.
    """
    rng = np.random.default_rng(20260722)
    fixtures = [
        _random_au_pro_fixture(rng, n_images=1, size=32, n_regions_per_image=1),
        _random_au_pro_fixture(rng, n_images=3, size=48, n_regions_per_image=4),
        _random_au_pro_fixture(
            rng, n_images=5, size=64, n_regions_per_image=6, region_size_range=(1, 4)
        ),
        _random_au_pro_fixture(
            rng, n_images=2, size=96, n_regions_per_image=3, region_size_range=(10, 40)
        ),
        _random_au_pro_fixture(rng, n_images=4, size=40, n_regions_per_image=8, region_size_range=(1, 3)),
    ]
    for masks, amaps in fixtures:
        got = au_pro(masks, amaps, fpr_limit=fpr_limit)
        want = _naive_au_pro(masks, amaps, fpr_limit=fpr_limit)
        assert got == pytest.approx(want, abs=1e-9)
