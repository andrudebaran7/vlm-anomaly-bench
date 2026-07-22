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
    a, b = _one_region(), _one_region(top_left=50)
    assert p_auroc([a, b], [a.astype(np.float32), b.astype(np.float32)]) == 1.0


def test_seg_f1max_perfect_separation():
    mask = _one_region()
    assert seg_f1max([mask], [mask.astype(np.float32)]) == pytest.approx(1.0)


def test_seg_f1max_no_anomalous_pixels_is_zero():
    empty = np.zeros((20, 20), dtype=np.uint8)
    rng = np.random.default_rng(0)
    assert seg_f1max([empty], [rng.random((20, 20))]) == 0.0


def test_seg_f1max_beats_threshold_noise():
    mask = _one_region()
    rng = np.random.default_rng(0)
    amap = mask.astype(np.float32) + 0.4 * rng.random(mask.shape).astype(np.float32)
    assert seg_f1max([mask], [amap]) == pytest.approx(1.0)
