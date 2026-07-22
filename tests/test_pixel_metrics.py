import numpy as np
import pytest

from vlmab.metrics.pixel_level import p_auroc, seg_f1max


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
