# Ground-truth-free threshold rule — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the pre-registered, ground-truth-free threshold rule the study needs to report the official MVTec AD 2 SegF1, and the metric that scores it.

**Architecture:** Three candidate rules are three instantiations of one primitive — transform the scores, pool them, cut at a quantile — differing on two axes: the transform (`identity` / `robust_z`) and the calibration source (`validation` / the test split). Both calibration and scoring stream one map at a time, so neither needs the memory guard `pixel_metrics` carries. The fitted parameters land in a YAML artifact that is committed to git, which is what makes "pre-registered" auditable after the fact.

**Tech Stack:** Python ≥3.10, numpy, scipy, scikit-learn, pandas, pyarrow, PyYAML, pytest. **No torch** — this entire plan runs on CPU in the dev/CI environment.

**Spec:** `docs/superpowers/specs/2026-08-20-threshold-rule-design.md`

## Global Constraints

- **`alpha = 1e-3`**, one pre-registered value. Sensitivity is reported over the fixed grid `{1e-4, 1e-3, 1e-2, 1e-1}`. Never derive `alpha` from `test_public` masks.
- **`per_image_robust_z` is the designated submission rule.** Fixed before any result exists.
- **Predicted positive is `amap >= threshold`** — `>=`, everywhere, because it decides the degenerate cases.
- **A threshold is per (method, category).** Maps are on each method's own scale (protocol v0.2.6); nothing is shared across methods.
- **CI installs only** `numpy scipy scikit-learn pandas pyarrow pillow pytest` plus (added in Task 4) `pyyaml`. No module added by this plan may import torch, anomalib, transformers or open_clip at load time.
- **Tests use synthetic arrays and `tests/mvtec_tree.py`.** Vial's 0.77 GB is not in git; the real-data run is a documented manual step (Task 7), never CI.
- **Protocol version after this plan lands: `0.2.11`.**
- Commit style: `feat:` / `test:` / `docs:` / `fix:`, and every commit message ends with the repo's `Co-Authored-By:` trailer.

---

### Task 1: `seg_f1_at` and `normal_pixel_fpr_at`

The official metric, and the diagnostic that reports how far the realised FPR drifts from the targeted `alpha`. Both stream, so a whole category pools at one-map-at-a-time memory.

**Files:**
- Modify: `src/vlmab/metrics/pixel_level.py` (add `_check_pairs`, `_per_image_thresholds`, `seg_f1_at`, `normal_pixel_fpr_at`; refactor `_flatten` to reuse `_check_pairs`)
- Test: `tests/test_pixel_metrics.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `seg_f1_at(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray], threshold: float | Sequence[float]) -> float`
  - `normal_pixel_fpr_at(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray], threshold: float | Sequence[float]) -> float`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_pixel_metrics.py`. Add `seg_f1_at` and `normal_pixel_fpr_at` to the existing import from `vlmab.metrics.pixel_level`.

```python
def _naive_seg_f1_at(masks, amaps, threshold):
    """Pool everything, then count. The obvious implementation seg_f1_at must match."""
    y = np.concatenate([(np.asarray(m) > 0).ravel() for m in masks])
    ts = [threshold] * len(amaps) if np.isscalar(threshold) else list(threshold)
    pred = np.concatenate(
        [(np.asarray(a, dtype=np.float32) >= np.float32(t)).ravel() for a, t in zip(amaps, ts)]
    )
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def test_seg_f1_at_matches_a_naive_pooled_reference():
    rng = np.random.default_rng(0)
    masks = [_one_region(size=40, region=12, top_left=5), _one_region(size=40, region=8, top_left=20)]
    amaps = [m.astype(np.float32) * 0.6 + rng.random(m.shape).astype(np.float32) for m in masks]
    for t in (0.2, 0.5, 0.9, 1.4):
        assert seg_f1_at(masks, amaps, t) == pytest.approx(_naive_seg_f1_at(masks, amaps, t))


def test_seg_f1_at_never_exceeds_the_oracle():
    # The invariant that ties the reportable metric to the oracle one: seg_f1max maximises
    # F1 over every threshold, so no fixed threshold can beat it. If this ever fails, one of
    # the two is computing a different quantity than its name claims.
    rng = np.random.default_rng(1)
    masks = [_one_region(size=30, region=10, top_left=4) for _ in range(3)]
    amaps = [m.astype(np.float32) * 0.4 + rng.random(m.shape).astype(np.float32) for m in masks]
    oracle = seg_f1max(masks, amaps)
    for t in rng.uniform(-0.5, 2.0, size=25):
        assert seg_f1_at(masks, amaps, float(t)) <= oracle + 1e-12


def test_seg_f1_at_pools_pixels_rather_than_averaging_images():
    # The official definition computes precision/recall over the complete pixel set, not per
    # image. Image `b` has no anomalous pixels at all, so its own F1 is 0 and the mean of the
    # two per-image F1s is half the pooled value -- a number this must NOT return.
    a = _one_region(size=20, region=10, top_left=2)
    b = np.zeros((20, 20), dtype=np.uint8)
    amaps = [a.astype(np.float32), np.zeros((20, 20), dtype=np.float32)]
    assert seg_f1_at([a, b], amaps, 0.5) == pytest.approx(1.0)


def test_seg_f1_at_accepts_one_threshold_per_image():
    a = _one_region(size=20, region=10, top_left=2)
    amaps = [a.astype(np.float32), a.astype(np.float32) * 10.0]
    # A single scalar cannot separate both images; the right per-image pair can.
    assert seg_f1_at([a, a], amaps, [0.5, 5.0]) == pytest.approx(1.0)


def test_seg_f1_at_rejects_a_threshold_vector_of_the_wrong_length():
    a = _one_region(size=8, region=3, top_left=1)
    with pytest.raises(ValueError, match="one threshold per image"):
        seg_f1_at([a, a], [a.astype(np.float32)] * 2, [0.5])


def test_seg_f1_at_is_zero_when_the_group_has_no_anomalous_pixels():
    z = np.zeros((8, 8), dtype=np.uint8)
    assert seg_f1_at([z], [np.ones((8, 8), dtype=np.float32)], 0.5) == 0.0


def test_seg_f1_at_is_zero_when_nothing_is_predicted():
    a = _one_region(size=8, region=3, top_left=1)
    assert seg_f1_at([a], [a.astype(np.float32)], np.inf) == 0.0


def test_normal_pixel_fpr_at_counts_only_normal_pixels():
    mask = _one_region(size=10, region=2, top_left=0)  # 4 anomalous of 100
    amap = np.ones((10, 10), dtype=np.float32)         # everything flagged at t=0.5
    assert normal_pixel_fpr_at([mask], [amap], 0.5) == pytest.approx(1.0)
    assert normal_pixel_fpr_at([mask], [amap], np.inf) == 0.0


def test_normal_pixel_fpr_at_raises_when_every_pixel_is_anomalous():
    ones = np.ones((6, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="no normal pixels"):
        normal_pixel_fpr_at([ones], [ones.astype(np.float32)], 0.5)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pixel_metrics.py -k "seg_f1_at or normal_pixel_fpr" -v`
Expected: FAIL at import — `ImportError: cannot import name 'seg_f1_at'`.

- [ ] **Step 3: Write the implementation**

In `src/vlmab/metrics/pixel_level.py`, extract the validation half of `_flatten` into `_check_pairs` and have `_flatten` call it — the streaming metrics need the same checks without the concatenation:

```python
def _check_pairs(masks: Sequence[np.ndarray], amaps: Sequence[np.ndarray]) -> None:
    if len(masks) != len(amaps):
        raise ValueError(f"masks/amaps length mismatch: {len(masks)} vs {len(amaps)}")
    for i, (m, a) in enumerate(zip(masks, amaps)):
        m_shape = np.asarray(m).shape
        a_shape = np.asarray(a).shape
        if m_shape != a_shape:
            raise ValueError(
                f"mask/amap shape mismatch at index {i}: {m_shape} vs {a_shape}"
            )
```

Replace the body of `_flatten` up to the `y = ...` line with a single `_check_pairs(masks, amaps)` call, leaving the two `np.concatenate` lines and the comment above `s` untouched.

Then append:

```python
def _per_image_thresholds(
    amaps: Sequence[np.ndarray], threshold: float | Sequence[float]
) -> list[float]:
    if np.isscalar(threshold):
        return [float(threshold)] * len(amaps)
    ts = [float(t) for t in threshold]
    if len(ts) != len(amaps):
        raise ValueError(
            f"expected one threshold per image: got {len(ts)} thresholds for "
            f"{len(amaps)} maps"
        )
    return ts


def seg_f1_at(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    threshold: float | Sequence[float],
) -> float:
    """Pixel F1 at a *fixed* threshold — the official MVTec AD 2 metric.

    Precision and recall are computed over the complete set of pooled pixels, not averaged
    over individual images (protocol §4, v0.2.10). `threshold` is a scalar, or one value per
    image for a per-image rule.

    Streams: at a fixed threshold nothing has to be sorted, only counted, so the working set
    is one mask plus one map at a time. That is why this can pool a whole category (the
    official definition) while `seg_f1max` cannot and stays per lighting condition -- the
    oracle sorts, this counts. Not interchangeable with `seg_f1max` in a results column:
    that one inspects the ground truth to choose its threshold and is strictly optimistic.

    Predicted positive is `amap >= threshold`, fixed so the degenerate cases are decidable.
    Returns 0.0 when the group has no anomalous pixels (mirroring `seg_f1max`) and when
    nothing is predicted, where precision is undefined and F1 with it.
    """
    _check_pairs(masks, amaps)
    ts = _per_image_thresholds(amaps, threshold)
    tp = fp = fn = 0
    for m, a, t in zip(masks, amaps, ts):
        y = np.asarray(m) > 0
        pred = np.asarray(a, dtype=np.float32) >= np.float32(t)
        tp += int(np.count_nonzero(pred & y))
        fp += int(np.count_nonzero(pred & ~y))
        fn += int(np.count_nonzero(~pred & y))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def normal_pixel_fpr_at(
    masks: Sequence[np.ndarray],
    amaps: Sequence[np.ndarray],
    threshold: float | Sequence[float],
) -> float:
    """Fraction of *normal* pixels flagged at this threshold.

    The realised false-positive rate, against the `alpha` a rule targeted on `validation`.
    The gap between the two is what a lighting shift does to a threshold, and is a reported
    result in its own right. Needs masks, so it is computable on `test_public` only -- never
    on the private splits, where it would be exactly the feedback protocol §7 forbids
    iterating against.
    """
    _check_pairs(masks, amaps)
    ts = _per_image_thresholds(amaps, threshold)
    fp = n_normal = 0
    for m, a, t in zip(masks, amaps, ts):
        normal = np.asarray(m) == 0
        pred = np.asarray(a, dtype=np.float32) >= np.float32(t)
        fp += int(np.count_nonzero(pred & normal))
        n_normal += int(np.count_nonzero(normal))
    if n_normal == 0:
        raise ValueError(
            "normal_pixel_fpr_at has no normal pixels to measure: every pooled pixel in "
            "this group is anomalous"
        )
    return fp / n_normal
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pixel_metrics.py -v`
Expected: PASS, including every pre-existing test in the file (the `_flatten` refactor must not change them).

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/metrics/pixel_level.py tests/test_pixel_metrics.py
git commit -m "feat: seg_f1_at and normal_pixel_fpr_at, the official metric at a fixed threshold"
```

---

### Task 2: the streaming top-k quantile

The primitive every rule cuts with. The `(1 - alpha)` quantile of a pooled set is the k-th largest value with `k = ceil(alpha * N)`; a running top-`k` computes it **exactly** in `O(k)` memory, so calibration never pools a split.

**Files:**
- Create: `src/vlmab/threshold/__init__.py`, `src/vlmab/threshold/rules.py`
- Test: `tests/test_threshold_rules.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `topk_quantile(chunks: Iterable[np.ndarray], n_total: int, alpha: float) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_threshold_rules.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_threshold_rules.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.threshold'`.

- [ ] **Step 3: Write the implementation**

Create `src/vlmab/threshold/__init__.py` (empty, matching `src/vlmab/metrics/__init__.py`).

Create `src/vlmab/threshold/rules.py`:

```python
"""Ground-truth-free threshold rules: one primitive, two axes.

Every rule here has the same shape -- transform the scores, pool them, cut at a quantile --
and differs only in the transform (`identity` or `robust_z`) and the calibration source
(the defect-free `validation` split, or the unlabelled test split itself). See
docs/superpowers/specs/2026-08-20-threshold-rule-design.md.

Pure numpy, no I/O: callers hand in a factory that yields anomaly maps, so this module never
learns where maps are stored.
"""
from typing import Iterable

import numpy as np


def topk_quantile(chunks: Iterable[np.ndarray], n_total: int, alpha: float) -> float:
    """The `(1 - alpha)` quantile of a pooled stream, exactly, in O(k) memory.

    That quantile is the k-th largest value with `k = ceil(alpha * n_total)`, so only the
    largest k values ever need to be resident. For Vial's validation split at alpha=1e-3
    that is ~109,000 float64s (0.4 MB) against 436 MB to pool the split -- which is why
    calibration needs no memory guard.

    `n_total` must be the true pooled size; the caller gets it from a shape-only pass
    (`np.load(..., mmap_mode="r").shape`), the pattern `aggregate.pixel_metrics` already uses.
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]: got {alpha}")
    if n_total <= 0:
        raise ValueError(f"n_total must be positive: got {n_total}")

    k = max(1, min(int(np.ceil(alpha * n_total)), n_total))
    buf = np.empty(0, dtype=np.float64)
    seen = 0
    for chunk in chunks:
        v = np.asarray(chunk, dtype=np.float64).ravel()
        seen += v.size
        buf = np.concatenate([buf, v])
        if buf.size > k:
            buf = np.partition(buf, buf.size - k)[buf.size - k :]
    if seen != n_total:
        raise ValueError(
            f"stream held {seen} values but n_total said {n_total}; the counting pass and "
            "the value pass disagree, so the quantile would be computed at the wrong rank"
        )
    return float(np.partition(buf, buf.size - k)[buf.size - k])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_threshold_rules.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/threshold/ tests/test_threshold_rules.py
git commit -m "feat: exact streaming top-k quantile, the primitive every threshold rule cuts with"
```

---

### Task 3: the three rules — calibrate and apply

The two axes made concrete. Calibration fits one scalar per (method, category); application turns that scalar into a per-image threshold vector `seg_f1_at` can consume directly.

**Files:**
- Modify: `src/vlmab/threshold/rules.py`
- Test: `tests/test_threshold_rules.py`

**Interfaces:**
- Consumes: `topk_quantile` from Task 2.
- Produces:
  - `RULES: tuple[str, ...]` — `("global_quantile", "per_image_robust_z", "transductive_quantile")`
  - `Calibration` — frozen dataclass with fields `rule: str`, `value: float`, `alpha: float`, `n_pixels: int`, `degenerate_mad: int`
  - `robust_z_stats(amap) -> tuple[float, float]` — `(median, mad)`
  - `calibrate(rule: str, maps_factory: Callable[[], Iterable[np.ndarray]], alpha: float) -> Calibration`
  - `thresholds_for(rule: str, value: float | None, maps_factory: Callable[[], Iterable[np.ndarray]], alpha: float) -> tuple[list[float], int]` — per-image thresholds and the degenerate-MAD count

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_threshold_rules.py`. Extend the import to
`from vlmab.threshold.rules import RULES, Calibration, calibrate, robust_z_stats, thresholds_for, topk_quantile`.

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_threshold_rules.py -v`
Expected: FAIL at import — `ImportError: cannot import name 'RULES'`.

- [ ] **Step 3: Write the implementation**

Append to `src/vlmab/threshold/rules.py`, and extend its imports to
`from dataclasses import dataclass` and `from typing import Callable, Iterable`:

```python
#: The three pre-registered candidates (protocol §4, v0.2.11). `per_image_robust_z` is the
#: designated submission rule; the other two are reported on `test_public` for comparison.
RULES = ("global_quantile", "per_image_robust_z", "transductive_quantile")


@dataclass(frozen=True)
class Calibration:
    """What a rule fitted on `validation`, and what it had to skip to fit it."""

    rule: str
    value: float          # the threshold itself for global_quantile; k for per_image_robust_z
    alpha: float
    n_pixels: int         # pixels that actually contributed, after skipping degenerate images
    degenerate_mad: int   # images with MAD == 0, excluded from the fit


def robust_z_stats(amap: np.ndarray) -> tuple[float, float]:
    """`(median, MAD)` of one map. The MAD's breakdown point is 50%, so a defect has to
    cover half the image before it moves these -- which is what lets a per-image threshold
    absorb a lighting shift without absorbing the anomaly along with it."""
    v = np.asarray(amap, dtype=np.float64).ravel()
    med = float(np.median(v))
    return med, float(np.median(np.abs(v - med)))


def calibrate(
    rule: str,
    maps_factory: Callable[[], Iterable[np.ndarray]],
    alpha: float,
) -> Calibration:
    """Fit `rule` on defect-free maps. `maps_factory()` must yield a *fresh* iterator each
    call: this makes two passes, one to count pixels and one to stream values, and the two
    must agree.

    `alpha` is the target false-positive rate on normal pixels -- the only quantity a
    defect-free split can speak to, since it contains no positives to compute F1 against.
    """
    if rule == "transductive_quantile":
        raise ValueError(
            "transductive_quantile fits nothing on validation; it is resolved at apply time "
            "from the test split's own scores -- call thresholds_for() instead"
        )
    if rule == "global_quantile":
        n_pixels = sum(int(np.asarray(a).size) for a in maps_factory())
        if n_pixels == 0:
            raise ValueError("global_quantile got no maps to calibrate on")
        value = topk_quantile(
            (np.asarray(a, dtype=np.float64).ravel() for a in maps_factory()), n_pixels, alpha
        )
        return Calibration(rule, value, alpha, n_pixels, 0)
    if rule == "per_image_robust_z":
        stats: list[tuple[float, float] | None] = []
        n_pixels = 0
        degenerate = 0
        for a in maps_factory():
            med, mad = robust_z_stats(a)
            if mad == 0.0:
                # A constant map has no scale to standardise by. Skipping it is the only
                # defined choice, and it is counted rather than silently dropped.
                degenerate += 1
                stats.append(None)
                continue
            stats.append((med, mad))
            n_pixels += int(np.asarray(a).size)
        if n_pixels == 0:
            raise ValueError(
                f"per_image_robust_z has nothing to calibrate on: all {degenerate} maps are "
                "constant (MAD == 0)"
            )

        def standardised():
            for a, st in zip(maps_factory(), stats):
                if st is None:
                    continue
                med, mad = st
                yield (np.asarray(a, dtype=np.float64).ravel() - med) / mad

        return Calibration(rule, topk_quantile(standardised(), n_pixels, alpha), alpha, n_pixels, degenerate)
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")


def thresholds_for(
    rule: str,
    value: float | None,
    maps_factory: Callable[[], Iterable[np.ndarray]],
    alpha: float,
) -> tuple[list[float], int]:
    """One threshold per map, ready to hand to `seg_f1_at`, plus the degenerate-MAD count.

    `value` is the calibrated scalar (`None` for transductive_quantile, which has none).
    """
    if rule == "global_quantile":
        if value is None:
            raise ValueError("global_quantile needs its calibrated threshold, got None")
        return [float(value)] * sum(1 for _ in maps_factory()), 0
    if rule == "per_image_robust_z":
        if value is None:
            raise ValueError("per_image_robust_z needs its calibrated k, got None")
        thresholds: list[float] = []
        degenerate = 0
        for a in maps_factory():
            med, mad = robust_z_stats(a)
            if mad == 0.0:
                # No scale, so no defensible cut: flag nothing and say so, rather than
                # dividing by zero or letting the image pass with an arbitrary threshold.
                thresholds.append(np.inf)
                degenerate += 1
            else:
                thresholds.append(med + float(value) * mad)
        return thresholds, degenerate
    if rule == "transductive_quantile":
        n_images = 0
        n_pixels = 0
        for a in maps_factory():
            n_images += 1
            n_pixels += int(np.asarray(a).size)
        if n_pixels == 0:
            raise ValueError("transductive_quantile got no maps to cut on")
        t = topk_quantile(
            (np.asarray(a, dtype=np.float64).ravel() for a in maps_factory()), n_pixels, alpha
        )
        return [t] * n_images, 0
    raise ValueError(f"unknown rule {rule!r}; expected one of {RULES}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_threshold_rules.py -v`
Expected: PASS, 20 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/threshold/rules.py tests/test_threshold_rules.py
git commit -m "feat: the three threshold rules, calibrate and apply"
```

---

### Task 4: the calibration artifact

The YAML that gets committed. Committing it is what makes "pre-registered" auditable after the fact — a file in git with a date, not a promise.

**Files:**
- Create: `src/vlmab/threshold/artifact.py`
- Create: `configs/thresholds/.gitkeep`
- Modify: `.github/workflows/ci.yml` (add `pyyaml` to the install line)
- Test: `tests/test_threshold_artifact.py`

**Interfaces:**
- Consumes: `RULES`, `Calibration` from Task 3.
- Produces:
  - `ARTIFACT_PROTOCOL_VERSION: str` — `"0.2.11"`
  - `write_artifact(path, *, dataset, method, alpha, calibrated_on, run_id, categories, designated_for_submission) -> Path` where `categories: dict[str, dict[str, Calibration]]`
  - `load_artifact(path) -> dict` — validates and returns the parsed mapping

- [ ] **Step 1: Add pyyaml to CI**

In `.github/workflows/ci.yml`, change the install line to:

```yaml
      - run: pip install numpy scipy scikit-learn pandas pyarrow pillow pytest pyyaml
```

The dependency list is deliberately minimal (no torch), and PyYAML is a pure-Python, no-build addition. It is needed because the calibration artifact is YAML, matching the `configs/methods/*.yaml` convention already in the repo. `requirements.txt` already lists `pyyaml`; only CI's explicit list was missing it.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_threshold_artifact.py`:

```python
import pytest
import yaml

from vlmab.threshold.artifact import (
    ARTIFACT_PROTOCOL_VERSION,
    load_artifact,
    write_artifact,
)
from vlmab.threshold.rules import Calibration


def _categories():
    return {
        "vial": {
            "global_quantile": Calibration("global_quantile", 12.34, 1e-3, 1000, 0),
            "per_image_robust_z": Calibration("per_image_robust_z", 8.7, 1e-3, 1000, 2),
            "transductive_quantile": Calibration("transductive_quantile", float("nan"), 1e-3, 0, 0),
        }
    }


def _write(tmp_path, **overrides):
    kwargs = dict(
        dataset="mvtec_ad2",
        method="intensity_baseline",
        alpha=1e-3,
        calibrated_on={"split": "validation", "n_images": 41, "lighting": ["regular"]},
        run_id="4880865ba2ed",
        categories=_categories(),
        designated_for_submission="per_image_robust_z",
    )
    kwargs.update(overrides)
    return write_artifact(tmp_path / "a.yaml", **kwargs)


def test_write_artifact_records_everything_a_reader_needs_to_audit_it(tmp_path):
    raw = yaml.safe_load(_write(tmp_path).read_text())
    assert raw["dataset"] == "mvtec_ad2"
    assert raw["method"] == "intensity_baseline"
    assert raw["alpha"] == 1e-3
    assert raw["protocol_version"] == ARTIFACT_PROTOCOL_VERSION
    assert raw["calibrated_on"]["split"] == "validation"
    assert raw["run_id"] == "4880865ba2ed"
    assert raw["designated_for_submission"] == "per_image_robust_z"
    assert raw["categories"]["vial"]["global_quantile"]["threshold"] == 12.34
    assert raw["categories"]["vial"]["per_image_robust_z"]["k"] == 8.7
    assert raw["categories"]["vial"]["per_image_robust_z"]["degenerate_mad"] == 2
    assert raw["categories"]["vial"]["transductive_quantile"] == {}


def test_write_artifact_stamps_the_calibration_date(tmp_path):
    raw = yaml.safe_load(_write(tmp_path).read_text())
    assert len(raw["calibrated_at"]) == 10 and raw["calibrated_at"].count("-") == 2


def test_load_artifact_round_trips_what_was_written(tmp_path):
    loaded = load_artifact(_write(tmp_path))
    assert loaded["categories"]["vial"]["per_image_robust_z"]["k"] == 8.7


def test_load_artifact_rejects_a_stale_protocol_version(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["protocol_version"] = "0.2.10"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="protocol_version"):
        load_artifact(path)


def test_load_artifact_rejects_a_designated_rule_missing_from_a_category(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["categories"]["vial"]["per_image_robust_z"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="designated_for_submission"):
        load_artifact(path)


def test_write_artifact_rejects_a_designated_rule_that_is_not_a_rule(tmp_path):
    with pytest.raises(ValueError, match="designated_for_submission"):
        _write(tmp_path, designated_for_submission="vibes")


def test_write_artifact_rejects_an_alpha_outside_the_unit_interval(tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        _write(tmp_path, alpha=0.0)


def test_load_artifact_rejects_an_unknown_rule_name(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["categories"]["vial"]["magic"] = {"threshold": 1.0}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown rule"):
        load_artifact(path)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_threshold_artifact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.threshold.artifact'`.

- [ ] **Step 4: Write the implementation**

Create `src/vlmab/threshold/artifact.py`:

```python
"""Read and write the committed calibration artifact.

The artifact is the pre-registration. Protocol §4 (v0.2.11) requires a threshold to be fixed
on the defect-free `validation` split and committed *before* any private-split submission, and
a file in git with a date is what makes that claim checkable afterwards -- a submission whose
artifact was committed later than the submission is a violation anyone can detect from the log.
"""
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import yaml

from vlmab.threshold.rules import RULES, Calibration

#: The protocol version this artifact format belongs to. `load_artifact` refuses anything
#: else: a threshold calibrated under different rules is not a threshold this protocol can
#: report, and silently accepting it would put an unpre-registered number in a results table.
ARTIFACT_PROTOCOL_VERSION = "0.2.11"

#: How each rule's fitted scalar is named in the file. transductive_quantile fits nothing.
_VALUE_KEY = {"global_quantile": "threshold", "per_image_robust_z": "k"}


def write_artifact(
    path: Path,
    *,
    dataset: str,
    method: str,
    alpha: float,
    calibrated_on: Mapping[str, Any],
    run_id: str,
    categories: Mapping[str, Mapping[str, Calibration]],
    designated_for_submission: str,
) -> Path:
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1]: got {alpha}")
    if designated_for_submission not in RULES:
        raise ValueError(
            f"designated_for_submission {designated_for_submission!r} is not a rule; "
            f"expected one of {RULES}"
        )

    blocks: dict[str, dict[str, dict[str, Any]]] = {}
    for category, by_rule in categories.items():
        if designated_for_submission not in by_rule:
            raise ValueError(
                f"designated_for_submission {designated_for_submission!r} is missing from "
                f"category {category!r}, so this artifact cannot produce the submission it names"
            )
        block: dict[str, dict[str, Any]] = {}
        for rule, cal in by_rule.items():
            if rule not in RULES:
                raise ValueError(f"unknown rule {rule!r} in category {category!r}")
            key = _VALUE_KEY.get(rule)
            block[rule] = (
                {}
                if key is None
                else {key: float(cal.value), "n_pixels": int(cal.n_pixels),
                      "degenerate_mad": int(cal.degenerate_mad)}
            )
        blocks[category] = block

    document = {
        "dataset": dataset,
        "method": method,
        "protocol_version": ARTIFACT_PROTOCOL_VERSION,
        "alpha": float(alpha),
        "calibrated_on": dict(calibrated_on),
        "run_id": run_id,
        "calibrated_at": date.today().isoformat(),
        "categories": blocks,
        "designated_for_submission": designated_for_submission,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    """Parse and validate. Raises rather than returning a document a caller would have to
    re-check, since every caller is about to put its numbers in a results table."""
    raw = yaml.safe_load(Path(path).read_text())
    version = raw.get("protocol_version")
    if version != ARTIFACT_PROTOCOL_VERSION:
        raise ValueError(
            f"{path} has protocol_version {version!r} but this code implements "
            f"{ARTIFACT_PROTOCOL_VERSION!r}: a threshold calibrated under different rules is "
            "not reportable under these ones"
        )
    designated = raw.get("designated_for_submission")
    if designated not in RULES:
        raise ValueError(
            f"{path} designated_for_submission is {designated!r}; expected one of {RULES}"
        )
    for category, block in raw.get("categories", {}).items():
        for rule in block:
            if rule not in RULES:
                raise ValueError(f"{path} category {category!r} has unknown rule {rule!r}")
        if designated not in block:
            raise ValueError(
                f"{path} category {category!r} is missing its designated_for_submission rule "
                f"{designated!r}"
            )
    return raw
```

Create `configs/thresholds/.gitkeep` (empty file) so the directory exists before the first calibration.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_threshold_artifact.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 6: Commit**

```bash
git add src/vlmab/threshold/artifact.py tests/test_threshold_artifact.py configs/thresholds/.gitkeep .github/workflows/ci.yml
git commit -m "feat: the committed calibration artifact, and pyyaml in CI"
```

---

### Task 5: `threshold_metrics` in aggregation

The reporting path. Deliberately separate from `pixel_metrics`: that one sorts and so must keep its 6 GB guard and stay per lighting condition, while this one counts and can pool a whole category, which is what the official definition asks for.

**Files:**
- Modify: `src/vlmab/eval/aggregate.py`
- Test: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `load_artifact` (Task 4), `thresholds_for`/`RULES` (Task 3), `seg_f1_at`/`normal_pixel_fpr_at` (Task 1), the existing `_load_pair`.
- Produces: `threshold_metrics(df: pd.DataFrame, artifact: Mapping[str, Any], category: str) -> dict[str, float]`, returning `seg_f1_at__<rule>`, `fpr_at__<rule>`, `degenerate_mad__<rule>` for each of `RULES`, plus `n`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_aggregate.py`, importing `threshold_metrics` alongside the existing `aggregate` imports and `from vlmab.threshold.rules import Calibration` / `from vlmab.threshold.artifact import write_artifact, load_artifact`.

```python
def _threshold_shard(tmp_path, n=4, size=16):
    """A shard with real map files on disk and one anomalous square per bad image."""
    import pandas as pd
    from PIL import Image

    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        amap = rng.random((size, size)).astype(np.float32)
        bad = i % 2 == 1
        mask_path = None
        if bad:
            amap[2:6, 2:6] += 5.0
            mask = np.zeros((size, size), dtype=np.uint8)
            mask[2:6, 2:6] = 255
            mask_path = tmp_path / f"{i}_mask.png"
            Image.fromarray(mask, mode="L").save(mask_path)
        map_path = tmp_path / f"{i}.npy"
        np.save(map_path, amap.astype(np.float16))
        rows.append({
            "image_path": f"/fake/{i}.png",
            "label": int(bad),
            "image_score": float(amap.max()),
            "split": "test_public",
            "mask_path": None if mask_path is None else str(mask_path),
            "map_path": str(map_path),
            "meta_lighting": "regular",
        })
    return pd.DataFrame(rows)


def _threshold_artifact(tmp_path, category="vial"):
    path = write_artifact(
        tmp_path / "cal.yaml",
        dataset="mvtec_ad2",
        method="intensity_baseline",
        alpha=1e-3,
        calibrated_on={"split": "validation", "n_images": 4, "lighting": ["regular"]},
        run_id="deadbeef",
        categories={category: {
            "global_quantile": Calibration("global_quantile", 1.0, 1e-3, 100, 0),
            "per_image_robust_z": Calibration("per_image_robust_z", 3.0, 1e-3, 100, 0),
            "transductive_quantile": Calibration("transductive_quantile", float("nan"), 1e-3, 0, 0),
        }},
        designated_for_submission="per_image_robust_z",
    )
    return load_artifact(path)


def test_threshold_metrics_reports_every_rule(tmp_path):
    df = _threshold_shard(tmp_path)
    out = threshold_metrics(df, _threshold_artifact(tmp_path), "vial")
    for rule in ("global_quantile", "per_image_robust_z", "transductive_quantile"):
        assert 0.0 <= out[f"seg_f1_at__{rule}"] <= 1.0
        assert 0.0 <= out[f"fpr_at__{rule}"] <= 1.0
        assert out[f"degenerate_mad__{rule}"] == 0
    assert out["n"] == 4


def test_threshold_metrics_never_beats_the_oracle(tmp_path):
    df = _threshold_shard(tmp_path)
    out = threshold_metrics(df, _threshold_artifact(tmp_path), "vial")
    oracle = pixel_metrics(df)["seg_f1max"]
    for rule in ("global_quantile", "per_image_robust_z", "transductive_quantile"):
        assert out[f"seg_f1_at__{rule}"] <= oracle + 1e-12


def test_threshold_metrics_refuses_unlabelled_rows(tmp_path):
    df = _threshold_shard(tmp_path)
    df.loc[0, "label"] = -1
    with pytest.raises(ValueError, match="unlabelled"):
        threshold_metrics(df, _threshold_artifact(tmp_path), "vial")


def test_threshold_metrics_raises_on_a_category_the_artifact_has_not_calibrated(tmp_path):
    df = _threshold_shard(tmp_path)
    with pytest.raises(KeyError, match="fabric"):
        threshold_metrics(df, _threshold_artifact(tmp_path), "fabric")


def test_threshold_metrics_needs_maps(tmp_path):
    df = _threshold_shard(tmp_path).drop(columns=["map_path"])
    with pytest.raises(ValueError, match="map_path"):
        threshold_metrics(df, _threshold_artifact(tmp_path), "vial")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_aggregate.py -k threshold_metrics -v`
Expected: FAIL at import — `ImportError: cannot import name 'threshold_metrics'`.

- [ ] **Step 3: Write the implementation**

In `src/vlmab/eval/aggregate.py`, extend the imports:

```python
from typing import Any, Mapping

from vlmab.metrics.pixel_level import (
    au_pro, normal_pixel_fpr_at, p_auroc, seg_f1_at, seg_f1max,
)
from vlmab.threshold.rules import RULES, thresholds_for
```

and append:

```python
#: How each rule's fitted scalar is named in the artifact; transductive_quantile fits nothing.
_ARTIFACT_VALUE_KEY = {"global_quantile": "threshold", "per_image_robust_z": "k"}


def threshold_metrics(
    df: pd.DataFrame,
    artifact: Mapping[str, Any],
    category: str,
) -> dict[str, float]:
    """SegF1 at each pre-registered rule, and the FPR each one actually realised.

    Kept out of `pixel_metrics` on purpose. `pixel_metrics` sorts (P-AUROC, SegF1max, AU-PRO),
    so it carries a memory guard and is aggregated per lighting condition. At a fixed threshold
    nothing is sorted, only counted, so this streams one map at a time and can pool a whole
    category -- which is what the official SegF1 definition requires (protocol §4). No
    `max_bytes` parameter, because there is nothing here to guard against.

    Needs ground truth, so it is for `test_public` only. `seg_f1_at` alongside `seg_f1max` in
    a table is fine and is the point; in the same *column* without marking is not (protocol §4).
    """
    if "map_path" not in df.columns:
        raise ValueError(
            "threshold_metrics needs a map_path column: it scores thresholded anomaly maps, "
            "so a run without saved maps has nothing for it to threshold"
        )
    labels = df["label"].to_numpy()
    if (labels == LABEL_UNKNOWN).any():
        raise ValueError(
            f"{int((labels == LABEL_UNKNOWN).sum())} unlabelled rows (label {LABEL_UNKNOWN}): "
            "these come from the private splits, whose ground truth is withheld. SegF1 and the "
            "realised FPR both need masks, and synthesising them would score unknown as normal."
        )
    if category not in artifact["categories"]:
        raise KeyError(
            f"the calibration artifact has no thresholds for category {category!r}; it "
            f"calibrated {sorted(artifact['categories'])}"
        )

    rows = list(df[df["map_path"].notna()].itertuples())
    if not rows:
        return {"n": 0}
    masks, amaps = zip(*(_load_pair(row) for row in rows))
    # One map at a time, re-read per pass: this is the streaming path, so holding the pair
    # list is already the peak and the factory adds nothing to it.
    factory = lambda: iter(amaps)

    block = artifact["categories"][category]
    alpha = float(artifact["alpha"])
    out: dict[str, float] = {}
    for rule in RULES:
        key = _ARTIFACT_VALUE_KEY.get(rule)
        value = None if key is None else float(block[rule][key])
        ts, degenerate = thresholds_for(rule, value, factory, alpha)
        out[f"seg_f1_at__{rule}"] = seg_f1_at(masks, amaps, ts)
        out[f"fpr_at__{rule}"] = normal_pixel_fpr_at(masks, amaps, ts)
        out[f"degenerate_mad__{rule}"] = degenerate
    out["n"] = len(rows)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_aggregate.py -v`
Expected: PASS, including every pre-existing test in the file.

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/eval/aggregate.py tests/test_aggregate.py
git commit -m "feat: threshold_metrics, the official SegF1 pooled over a whole category"
```

---

### Task 6: `scripts/calibrate_threshold.py`

The one new entry point, because it is the step whose output gets committed. Its single most important behaviour is the guard in Step 1's first test: **it refuses to calibrate on anything but the `validation` split.**

Shards are keyed `(dataset, method, category)` with no split in the filename (`ResultStore.path_for`), so a `validation` run and a `test_public` run written to the same `--results` root collide — the second is skipped by `is_done`, and the first is what a naive calibration would read. Two runs therefore need two roots, and this CLI verifies the shard's own `split` column rather than trusting the caller to have done that. Calibrating on test data is the exact failure this whole piece of work exists to prevent, so it is checked, not documented.

**Files:**
- Create: `scripts/calibrate_threshold.py`
- Test: `tests/test_calibrate_threshold.py`

**Interfaces:**
- Consumes: `ResultStore` (`vlmab.eval.store`), `calibrate`/`Calibration`/`RULES` (Task 3), `write_artifact` (Task 4).
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calibrate_threshold.py`:

```python
import numpy as np
import pytest
import yaml

from vlmab.eval.store import ResultStore


def _validation_run(tmp_path, split="validation", n=6, size=12):
    """A shard of defect-free rows with real maps on disk, laid out as the runner lays them out.

    The map directory name matters: `run_id` is not a shard column, so
    `<dataset>__<method>__<run_id>__<category>` is the only place the run is recorded and the
    only place the artifact's provenance can come from.
    """
    maps = tmp_path / "maps" / "mvtec_ad2__intensity_baseline__cafe1234__vial"
    maps.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = maps / f"{i}.npy"
        np.save(path, rng.random((size, size)).astype(np.float16))
        rows.append({
            "image_path": f"/fake/{i}.png",
            "label": 0,
            "image_score": 0.5,
            "split": split,
            "mask_path": None,
            "map_path": str(path),
        })
    results = tmp_path / "shards"
    ResultStore(results).write("mvtec_ad2", "intensity_baseline", "vial", rows, {"seed": 0})
    return results


def _run(argv):
    from calibrate_threshold import main

    return main(argv)


def test_calibrate_refuses_a_shard_that_is_not_the_validation_split(tmp_path):
    # The failure this entire piece of work exists to prevent. Shards are keyed
    # (dataset, method, category) with no split, so pointing this at a test_public run is an
    # easy mistake and a silent one -- unless it raises here.
    results = _validation_run(tmp_path, split="test_public")
    with pytest.raises(ValueError, match="test_public"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_writes_an_artifact_for_every_category_it_found(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "intensity_baseline", "--out", str(out)]) == 0
    raw = yaml.safe_load(out.read_text())
    assert set(raw["categories"]) == {"vial"}
    assert raw["categories"]["vial"]["global_quantile"]["threshold"] > 0
    assert raw["categories"]["vial"]["per_image_robust_z"]["k"] > 0
    assert raw["categories"]["vial"]["transductive_quantile"] == {}


def test_calibrate_defaults_to_the_pre_registered_alpha_and_designated_rule(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    _run(["--results", str(results), "--dataset", "mvtec_ad2",
          "--method", "intensity_baseline", "--out", str(out)])
    raw = yaml.safe_load(out.read_text())
    assert raw["alpha"] == 1e-3
    assert raw["designated_for_submission"] == "per_image_robust_z"


def test_calibrate_records_the_run_id_and_image_count_it_calibrated_from(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    _run(["--results", str(results), "--dataset", "mvtec_ad2",
          "--method", "intensity_baseline", "--out", str(out)])
    raw = yaml.safe_load(out.read_text())
    assert raw["run_id"] == "cafe1234"
    assert raw["calibrated_on"]["n_images"] == 6
    assert raw["calibrated_on"]["split"] == "validation"


def test_calibrate_raises_when_the_method_has_no_shard(tmp_path):
    results = _validation_run(tmp_path)
    with pytest.raises(ValueError, match="winclip"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "winclip", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_raises_when_the_run_saved_no_maps(tmp_path):
    results = _validation_run(tmp_path)
    import pandas as pd

    shard = results / "mvtec_ad2__intensity_baseline__vial.parquet"
    pd.read_parquet(shard).drop(columns=["map_path"]).to_parquet(shard, index=False)
    with pytest.raises(ValueError, match="map_path"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_runs_end_to_end_on_a_synthetic_mvtec_tree(tmp_path):
    """The whole chain CI can run: dataset -> runner -> shard -> calibration artifact.

    The other tests here hand-build a shard, which cannot catch a mismatch between what the
    runner writes and what calibration reads. This goes through `run_evaluation` on the real
    loader over a synthetic tree.

    The method has to produce non-constant maps: `build_category` fills validation images with
    a uniform value, so `intensity_baseline` would return an all-zero deviation map, MAD would
    be 0 on every image and `per_image_robust_z` would have nothing to calibrate on. That is a
    real degenerate path, covered in test_threshold_rules.py; it is not the path this test is
    for.
    """
    from mvtec_tree import build_category
    from vlmab.datasets.mvtec_ad2 import MVTecAD2
    from vlmab.eval.runner import run_evaluation
    from vlmab.methods.base import AnomalyMethod, Prediction

    class NoisyMethod(AnomalyMethod):
        name = "noisy"
        zero_shot = True

        def prepare(self, device="cuda"):
            pass

        def predict(self, image, category):
            h, w = np.asarray(image).shape[:2]
            amap = np.random.default_rng(abs(hash((h, w))) % 2**32).random((h, w)).astype(np.float32)
            return Prediction(image_score=float(amap.max()), anomaly_map=amap)

    root = tmp_path / "data"
    build_category(root, "vial", n_good=3, n_bad=1, size=(12, 10))
    results, maps = tmp_path / "shards", tmp_path / "maps"
    run_evaluation(
        MVTecAD2(root), NoisyMethod(), ResultStore(results), {"seed": 0},
        split="validation", maps_dir=maps, device="cpu",
    )

    out = tmp_path / "cal.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "noisy", "--out", str(out)]) == 0
    raw = yaml.safe_load(out.read_text())
    assert raw["calibrated_on"] == {"split": "validation", "n_images": 3, "lighting": ["regular"]}
    assert np.isfinite(raw["categories"]["vial"]["global_quantile"]["threshold"])
    assert np.isfinite(raw["categories"]["vial"]["per_image_robust_z"]["k"])
    assert raw["run_id"]  # recovered from the runner's map directory name, not a column
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_calibrate_threshold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'calibrate_threshold'`. (`pyproject.toml` puts `scripts` on `pythonpath` for pytest, so the import works once the file exists.)

- [ ] **Step 3: Write the implementation**

Create `scripts/calibrate_threshold.py`:

```python
#!/usr/bin/env python3
"""Fit the pre-registered threshold rules on a defect-free `validation` run.

The output is committed: it *is* the pre-registration required by protocol §4 (v0.2.11).
Run it once per method, after that method's validation run, and commit the YAML before any
private-split submission.

    run_eval.py --split validation --results runs/val --maps-dir runs/val/maps --method M ...
    calibrate_threshold.py --results runs/val --dataset mvtec_ad2 --method M \\
        --out configs/thresholds/mvtec_ad2__M.yaml
    git add configs/thresholds/mvtec_ad2__M.yaml && git commit

The alpha-sensitivity curve protocol §4 requires is produced by running this at each point of
the registered grid {1e-4, 1e-3, 1e-2, 1e-1} into *throwaway* paths, then scoring each one with
`aggregate.threshold_metrics`. Only the alpha=1e-3 artifact is committed: the sweep is a
reported curve, not a choice, and an uncommitted artifact cannot be mistaken for one.
"""
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from vlmab.eval.store import ResultStore
from vlmab.threshold.artifact import write_artifact
from vlmab.threshold.rules import Calibration, calibrate

#: Protocol §4 (v0.2.11). Both are pre-registered, so they are defaults rather than choices
#: a caller is expected to make -- overriding them is what the sensitivity sweep does, and it
#: does not overwrite the committed artifact.
PREREGISTERED_ALPHA = 1e-3
DESIGNATED_RULE = "per_image_robust_z"

#: Fitting a threshold on anything else is calibrating on test data.
CALIBRATION_SPLIT = "validation"


def _run_id_from_maps(map_paths) -> str:
    """Recover the run from the map directory name.

    The runner names it `<dataset>__<method>__<run_id>__<category>` and does **not** write a
    `run_id` shard column, so this directory name is the only record of which run produced
    these maps -- and the artifact is worth much less without it.
    """
    ids = set()
    for name in {Path(p).parent.name for p in map_paths}:
        parts = name.split("__")
        if len(parts) != 4:
            raise ValueError(
                f"map directory {name!r} does not match the runner's "
                "<dataset>__<method>__<run_id>__<category> layout, so the run this "
                "calibration came from cannot be recorded"
            )
        ids.add(parts[2])
    return ",".join(sorted(ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path, help="ResultStore root of the validation run")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", default=PREREGISTERED_ALPHA, type=float)
    parser.add_argument("--designate", default=DESIGNATED_RULE)
    args = parser.parse_args(argv)

    df = ResultStore(args.results).load_all()
    if df.empty:
        raise ValueError(f"no shards under {args.results}")
    df = df[(df["dataset"] == args.dataset) & (df["method"] == args.method)]
    if df.empty:
        raise ValueError(
            f"no shard for dataset={args.dataset!r} method={args.method!r} under {args.results}"
        )

    # Shards are keyed (dataset, method, category) with no split, so a test_public run written
    # to this root is indistinguishable by filename. Check the column, not the caller's word.
    splits = sorted(set(df["split"].unique()))
    if splits != [CALIBRATION_SPLIT]:
        raise ValueError(
            f"refusing to calibrate on split(s) {splits}: a threshold rule must be fitted on "
            f"{CALIBRATION_SPLIT!r}, which is defect-free, never on test data (protocol §4). "
            f"Point --results at a run made with --split {CALIBRATION_SPLIT}."
        )
    if "map_path" not in df.columns or df["map_path"].isna().all():
        raise ValueError(
            f"the run under {args.results} saved no anomaly maps (no usable map_path column); "
            "re-run it with --maps-dir, since calibration reads the maps themselves"
        )

    all_maps = [p for p in df["map_path"] if isinstance(p, str)]
    categories: dict[str, dict[str, Calibration]] = {}
    for category, group in sorted(df.groupby("category"), key=lambda kv: kv[0]):
        paths = [p for p in group["map_path"] if isinstance(p, str)]
        factory = lambda paths=paths: (np.load(p).astype(np.float32) for p in paths)
        categories[category] = {
            "global_quantile": calibrate("global_quantile", factory, args.alpha),
            "per_image_robust_z": calibrate("per_image_robust_z", factory, args.alpha),
            "transductive_quantile": Calibration(
                "transductive_quantile", float("nan"), args.alpha, 0, 0
            ),
        }

    path = write_artifact(
        args.out,
        dataset=args.dataset,
        method=args.method,
        alpha=args.alpha,
        calibrated_on={
            "split": CALIBRATION_SPLIT,
            "n_images": int(len(df)),
            "lighting": sorted(set(df["meta_lighting"])) if "meta_lighting" in df.columns else [],
        },
        run_id=_run_id_from_maps(all_maps),
        categories=categories,
        designated_for_submission=args.designate,
    )
    print(f"wrote {path} — commit it before any submission (protocol §4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_calibrate_threshold.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Run the whole suite**

Run: `pytest -q`
Expected: PASS. The suite was 246 tests before this plan, which adds 49 (9 + 7 + 13 + 8 + 5 + 7),
so expect 295. If the number differs, find out why before continuing — a test that silently did
not run is the failure mode this check exists for. Task 7 puts the real number in `README.md`.

- [ ] **Step 6: Commit**

```bash
git add scripts/calibrate_threshold.py tests/test_calibrate_threshold.py
git commit -m "feat: calibrate_threshold CLI, refusing any split but validation"
```

---

### Task 7: protocol amendment v0.2.11, and the real-data check

The code is worth nothing to this study unless the rule it implements is pre-registered in the frozen protocol. This task also runs the one end-to-end check that uses real data, which CI cannot.

**Files:**
- Modify: `docs/protocol.md` (§4 and Changelog)
- Modify: `docs/next-steps.md` (the ⚠️ section and the ordered list)
- Modify: `README.md` (test count, if it states one)

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Run the end-to-end check on real data**

Vial is the one category on disk and `intensity_baseline` is pure numpy, so the whole chain runs on CPU. This is a **manual step, not CI** — record the output in the commit message.

```bash
python scripts/run_eval.py --method intensity_baseline --root data/mvtec_ad2 \
    --results /tmp/thr-val --maps-dir /tmp/thr-val/maps --split validation \
    --category vial --device cpu
python scripts/calibrate_threshold.py --results /tmp/thr-val --dataset mvtec_ad2 \
    --method intensity_baseline --out /tmp/thr-val/cal.yaml
cat /tmp/thr-val/cal.yaml
```

Expected: a YAML with `calibrated_on.n_images: 41`, `lighting: [regular]`, a finite
`global_quantile.threshold` and a finite `per_image_robust_z.k`.

Then score `test_public` against it, **into a separate results root** (shards carry no split in
their filename, so reusing the root would collide):

```bash
python scripts/run_eval.py --method intensity_baseline --root data/mvtec_ad2 \
    --results /tmp/thr-test --maps-dir /tmp/thr-test/maps --split test_public \
    --category vial --device cpu
python - <<'PY'
from vlmab.eval.aggregate import pixel_metrics, threshold_metrics
from vlmab.eval.store import ResultStore
from vlmab.threshold.artifact import load_artifact

df = ResultStore("/tmp/thr-test").load_all()
art = load_artifact("/tmp/thr-test/../thr-val/cal.yaml")
one = df[df["meta_lighting"] == "regular"]
print("oracle seg_f1max:", pixel_metrics(one)["seg_f1max"])
print(threshold_metrics(df, art, "vial"))
PY
```

Expected: every `seg_f1_at__*` at or below the oracle, and `fpr_at__*` reported. The absolute
numbers will be poor — `intensity_baseline` is the honest floor, not a detector — and that is
fine. What is being verified is that the chain runs on real 1400x1900 maps and that pooling a
whole category through `threshold_metrics` does **not** hit the memory error `pixel_metrics`
raises on the same 140 images.

- [ ] **Step 2: Amend protocol §4**

In `docs/protocol.md`, replace the bullet beginning "**SegF1 needs a threshold, and `seg_f1max` is not it.**" Keep its first three sentences (the oracle explanation is still true and still needed), then replace everything from "Until a ground-truth-free threshold rule exists" to the end of that bullet with:

```markdown
  **The rule, pre-registered 2026-08-20 (v0.2.11).** Every candidate has the same form —
  transform the scores, pool them, cut at a quantile — and they differ on two axes: the
  transform (`identity`, or `robust_z`: per image, `(s - median)/MAD`) and the calibration
  source (the defect-free `validation` split, or the unlabelled test split itself). Three are
  registered:

  | Rule | transform | source | Fitted |
  |---|---|---|---|
  | `global_quantile` | `identity` | `validation` | one threshold per (method, category) |
  | `per_image_robust_z` | `robust_z` | `validation` | one `k` per (method, category) |
  | `transductive_quantile` | `identity` | test split | nothing; resolved at apply time |

  **`alpha = 1e-3`**, one value, justified as a stated deployment operating point — no more
  than one pixel in a thousand flagged on a defect-free part — and deliberately **not** derived
  from the defect-area fraction, which exists only in `test_public`'s masks and whose use would
  be calibrating on test. Sensitivity is reported over the fixed grid
  `{1e-4, 1e-3, 1e-2, 1e-1}`; the grid is registered here so it cannot be chosen after seeing
  the curve.

  **`per_image_robust_z` is designated for the private-split submission**, fixed before any
  result existed. §7 allows one submission per method, so this cannot be revisited after seeing
  public-split numbers. The justification is a priori: what distinguishes the private split is
  `test_private_mixed`'s lighting variation, and it is the only candidate whose calibration is
  invariant to a per-image shift in level and spread; and it is inductive, needing one image at
  a time rather than the whole split, which is the deployment setting this study argues for.
  The other two are computed and reported on `test_public` for comparison; only this one is
  submitted. `transductive_quantile` uses the test *inputs* and never the labels, which is
  ground-truth-free, but it is a different access assumption and is labelled transductive in
  every table.

  **A per-image threshold is not per-image normalisation.** The continuous map stays on the
  method's own scale, unrescaled — v0.2.6 protects the cross-image ranking that P-AUROC, AU-PRO
  and SegF1max depend on, and a per-image *threshold* changes no score.

  **Reporting.** `seg_f1_at` (at a rule) and `seg_f1max` (oracle) may appear in the same table
  and never in the same unmarked column. `seg_f1_at` pools pixels over the complete category,
  as the official definition requires, because at a fixed threshold nothing is sorted; the
  oracle sorts, so it stays per lighting condition under the memory budget above. The realised
  normal-pixel FPR is reported next to the targeted `alpha`, on `test_public` only — on the
  private splits it would be exactly the feedback §7 forbids iterating against.

  The calibration artifact (`configs/thresholds/<dataset>__<method>.yaml`) is committed to git
  before any submission; a submission whose artifact was committed later is a violation
  detectable from the log.
```

- [ ] **Step 3: Add the Changelog entry**

Append to the Changelog in `docs/protocol.md`:

```markdown
- 2026-08-20 — v0.2.11. Pre-registers the ground-truth-free threshold rule §4 v0.2.10 named as a
  hard prerequisite for M4: three candidates on two axes, `alpha = 1e-3` with a fixed sensitivity
  grid, and `per_image_robust_z` designated for the private-split submission with an a priori
  justification recorded before any result existed. Adds `seg_f1_at` (the official SegF1 at a
  fixed threshold, pooled over a whole category) and the realised-FPR diagnostic, and requires
  the calibration artifact to be committed before submission. Clarifies that a per-image
  threshold is not the per-image normalisation v0.2.6 forbids. No existing metric changed.
  Rationale: docs/superpowers/specs/2026-08-20-threshold-rule-design.md
```

- [ ] **Step 4: Update `docs/next-steps.md`**

Replace the whole "⚠️ The one piece of unplanned work, and it gates M4" section with:

```markdown
## The threshold rule — built and pre-registered (2026-08-20)

The repo's one piece of unplanned work is done. Protocol v0.2.11 §4 pre-registers three
ground-truth-free rules on two axes (transform x calibration source), `alpha = 1e-3` with a
fixed sensitivity grid, and **`per_image_robust_z` as the designated private-split submission
rule** — fixed, with an a priori justification, before any result existed. `seg_f1_at` computes
the official SegF1 at a fixed threshold, pooled over a whole category as the official definition
requires; `seg_f1max` stays and stays marked **oracle**.

Calibration runs on the defect-free `validation` split via `scripts/calibrate_threshold.py`,
which refuses any other split, and writes `configs/thresholds/<dataset>__<method>.yaml`. **That
file is committed, and committing it before a submission is what makes the pre-registration
auditable.**

What still gates M4, in order:

1. **Submission packaging** — writing the server's payload (thresholded plus continuous maps in
   its format). Deliberately not built: the format is unverified until first login
   (`docs/datasets-access.md` leaves it as a checkbox), and guessing an upstream API is what
   protocol §3 forbids. Confirm the format on first login, then build it.
2. **Each method's VisA ±1pt gate**, which is the pre-existing ordering above and unchanged.

The paper's C1 and C3 `\todo{withdraw if the threshold rule is not built before submission}`
tripwires can come out once a first calibration has run on a real method — not on
`intensity_baseline`, which is the floor, not a detector.
```

Then, in the ordered list, replace step 7's sentence "Access is **granted** (2026-07-30), but M4
is **blocked on unbuilt work**, not on access: see the threshold rule below." with "Access is
**granted** (2026-07-30) and the threshold rule is built and pre-registered (v0.2.11); what
remains is the submission packaging, which needs the server's own format — confirm it on first
login." Leave the rest of step 7 as it is.

Finally, in "Where to pick up", drop candidate 1 (the threshold-rule spec) and renumber, so
PatchCore's first Colab session becomes the leading candidate.

- [ ] **Step 5: Update the test count in `README.md`**

Run `pytest -q` and put the actual number in, replacing the current "246".

- [ ] **Step 6: Commit**

```bash
git add docs/protocol.md docs/next-steps.md README.md
git commit -m "docs: pre-register the threshold rule in protocol v0.2.11"
```

- [ ] **Step 7: Verify the whole suite one more time**

Run: `pytest -q`
Expected: PASS, all tests. Confirm the number matches what went into `README.md`.

---

## Out of scope

Stated in the spec, repeated here so no task quietly grows to cover it:

- **Submission packaging** — writing the server's payload (thresholded plus continuous maps in its format). Unverified until first login; writing it now would mean guessing an upstream API.
- **Morphological post-processing** of the binary mask. Orthogonal to threshold choice, and adds hyperparameters that would need their own calibration.
- **Removing the `\todo{withdraw}` tripwires** from C1 and C3 in `vlm-anomaly-paper/sections/02-introduction.tex`. That is a commit in the paper repo, after this lands and after a first calibration runs on a real method.
