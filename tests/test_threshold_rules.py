import numpy as np
import pytest

from vlmab.threshold.rules import topk_quantile


def _kth_largest(pooled, alpha):
    """The definition, computed the obvious way on data small enough to pool."""
    k = int(np.ceil(alpha * pooled.size))
    k = max(1, min(k, pooled.size))
    return float(np.sort(np.asarray(pooled, dtype=np.float64).ravel())[-k])


def test_topk_quantile_matches_a_full_sort():
    rng = np.random.default_rng(0)
    chunks = [rng.normal(size=(37, 11)) for _ in range(5)]
    pooled = np.concatenate([c.ravel() for c in chunks])
    for alpha in (1e-3, 1e-2, 0.1, 0.5):
        got = topk_quantile((c for c in chunks), pooled.size, alpha)
        assert got == pytest.approx(_kth_largest(pooled, alpha))


def test_topk_quantile_is_independent_of_how_the_stream_is_chunked():
    rng = np.random.default_rng(1)
    pooled = rng.normal(size=1000)
    one = topk_quantile((pooled,), pooled.size, 0.01)
    many = topk_quantile((pooled[i : i + 7] for i in range(0, 1000, 7)), pooled.size, 0.01)
    assert one == many


def test_topk_quantile_at_the_k_equals_one_edge_is_the_maximum():
    v = np.arange(100.0)
    assert topk_quantile((v,), v.size, 1e-9) == 99.0


def test_topk_quantile_at_alpha_one_is_the_minimum():
    v = np.arange(100.0)
    assert topk_quantile((v,), v.size, 1.0) == 0.0


def test_topk_quantile_exceedance_fraction_is_about_alpha():
    # What the quantile is *for*: thresholding at it flags about alpha of the values, using
    # the same `>=` comparison seg_f1_at uses.
    rng = np.random.default_rng(2)
    v = rng.normal(size=200_000)
    t = topk_quantile((v,), v.size, 1e-3)
    assert np.count_nonzero(v >= t) / v.size == pytest.approx(1e-3, rel=0.05)


def test_topk_quantile_rejects_an_alpha_outside_the_unit_interval():
    v = np.arange(10.0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="alpha"):
            topk_quantile((v,), v.size, bad)


def test_topk_quantile_raises_when_the_stream_is_shorter_than_promised():
    with pytest.raises(ValueError, match="n_total"):
        topk_quantile((np.arange(5.0),), 100, 0.5)
