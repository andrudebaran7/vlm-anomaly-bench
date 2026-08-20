import numpy as np
import pytest

from vlmab.threshold.rules import RULES, Calibration, calibrate, robust_z_stats, thresholds_for, topk_quantile


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


def _factory(maps):
    return lambda: iter(maps)


def test_robust_z_stats_returns_median_and_mad():
    v = np.array([[1.0, 2.0], [3.0, 100.0]])
    med, mad = robust_z_stats(v)
    assert med == 2.5
    assert mad == pytest.approx(1.0)  # |1-2.5|,|2-2.5|,|3-2.5|,|100-2.5| -> median 1.0


def test_global_quantile_calibrates_to_the_pooled_quantile():
    rng = np.random.default_rng(0)
    maps = [rng.normal(size=(50, 50)) for _ in range(4)]
    cal = calibrate("global_quantile", _factory(maps), 0.01)
    pooled = np.concatenate([m.ravel() for m in maps])
    assert cal.rule == "global_quantile"
    assert cal.alpha == 0.01
    assert cal.n_pixels == pooled.size
    assert cal.degenerate_mad == 0
    assert cal.value == pytest.approx(_kth_largest(pooled, 0.01))


def test_global_quantile_hits_its_target_fpr_on_held_out_normals():
    # The claim calibration makes: a threshold fitted on defect-free data flags about alpha
    # of the normal pixels it has never seen.
    rng = np.random.default_rng(1)
    fit = [rng.normal(size=(200, 200)) for _ in range(5)]
    held_out = np.concatenate([rng.normal(size=(200, 200)).ravel() for _ in range(5)])
    cal = calibrate("global_quantile", _factory(fit), 1e-2)
    assert np.count_nonzero(held_out >= cal.value) / held_out.size == pytest.approx(1e-2, rel=0.15)


def test_per_image_robust_z_calibrates_k_in_standardised_space():
    rng = np.random.default_rng(2)
    maps = [rng.normal(size=(60, 60)) for _ in range(4)]
    cal = calibrate("per_image_robust_z", _factory(maps), 0.01)
    zs = []
    for m in maps:
        med, mad = robust_z_stats(m)
        zs.append((m.ravel() - med) / mad)
    assert cal.value == pytest.approx(_kth_largest(np.concatenate(zs), 0.01))


def test_per_image_robust_z_skips_and_counts_a_constant_image():
    rng = np.random.default_rng(3)
    maps = [rng.normal(size=(40, 40)), np.full((40, 40), 7.0)]
    cal = calibrate("per_image_robust_z", _factory(maps), 0.05)
    assert cal.degenerate_mad == 1
    assert cal.n_pixels == 1600  # the constant image contributed nothing


def test_calibrate_refuses_the_transductive_rule():
    with pytest.raises(ValueError, match="apply time"):
        calibrate("transductive_quantile", _factory([np.zeros((4, 4))]), 0.1)


def test_calibrate_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown rule"):
        calibrate("magic", _factory([np.zeros((4, 4))]), 0.1)


def test_global_quantile_applies_one_constant_threshold_to_every_image():
    maps = [np.zeros((5, 5)), np.ones((5, 5))]
    ts, degenerate = thresholds_for("global_quantile", 3.5, _factory(maps), 0.1)
    assert ts == [3.5, 3.5]
    assert degenerate == 0


def test_per_image_robust_z_applies_median_plus_k_mad_per_image():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    med, mad = robust_z_stats(a)
    ts, degenerate = thresholds_for("per_image_robust_z", 2.0, _factory([a]), 0.1)
    assert ts == [pytest.approx(med + 2.0 * mad)]
    assert degenerate == 0


def test_per_image_robust_z_predicts_nothing_on_a_constant_image():
    ts, degenerate = thresholds_for(
        "per_image_robust_z", 2.0, _factory([np.full((4, 4), 5.0)]), 0.1
    )
    assert ts == [np.inf]
    assert degenerate == 1


def test_per_image_robust_z_is_invariant_to_a_per_image_affine_shift_and_global_quantile_is_not():
    # This is the scientific claim B rests on, and the reason it is the designated submission
    # rule: a lighting shift that rescales an image's scores must not change which pixels it
    # flags. If this ever fails, B has lost its reason to exist and the designation in
    # protocol §4 must be revisited.
    rng = np.random.default_rng(4)
    image = rng.normal(size=(50, 50))
    shifted = image * 3.0 + 7.0

    (t_plain,), _ = thresholds_for("per_image_robust_z", 2.5, _factory([image]), 0.1)
    (t_shift,), _ = thresholds_for("per_image_robust_z", 2.5, _factory([shifted]), 0.1)
    assert np.array_equal(image >= t_plain, shifted >= t_shift)

    fixed = float(np.quantile(image, 0.99))
    (a_plain,), _ = thresholds_for("global_quantile", fixed, _factory([image]), 0.1)
    (a_shift,), _ = thresholds_for("global_quantile", fixed, _factory([shifted]), 0.1)
    assert not np.array_equal(image >= a_plain, shifted >= a_shift)


def test_transductive_quantile_cuts_on_the_maps_it_is_applied_to():
    rng = np.random.default_rng(5)
    maps = [rng.normal(size=(80, 80)) for _ in range(3)]
    ts, degenerate = thresholds_for("transductive_quantile", None, _factory(maps), 0.01)
    pooled = np.concatenate([m.ravel() for m in maps])
    assert degenerate == 0
    assert len(ts) == 3 and len(set(ts)) == 1
    assert ts[0] == pytest.approx(_kth_largest(pooled, 0.01))


def test_rules_tuple_names_exactly_the_three_designed_rules():
    assert RULES == ("global_quantile", "per_image_robust_z", "transductive_quantile")
