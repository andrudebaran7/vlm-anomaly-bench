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
