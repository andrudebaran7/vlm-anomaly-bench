# PatchCore Full-Shot Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the framework so a full-shot anchor fits per category, fix the anomaly-map scaling that per-image normalisation was breaking, and build the PatchCore adapter behind an injectable backend — so the only thing left is wiring the real anomalib call on Colab.

**Architecture:** Two contract corrections come first, both forced by PatchCore being the first method where they matter, both CPU-verifiable. (1) `AnomalyMethod` gains `fit(train_images, category)` — a no-op for zero-shot methods, called by the runner once per category before scoring that category, so a full-shot method can build a per-category memory bank. (2) The anomaly-map contract relaxes from "[0,1]" to "finite float32 on the method's internal, consistent scale", because pixel metrics rank pooled pixels across images and per-image [0,1] normalisation destroys that ranking. The PatchCore adapter then wraps an injected backend (`fit` + `score`); the real anomalib backend is written on Colab against the pinned version, because its inference code cannot be written correctly without anomalib and a GPU present.

**Tech Stack:** Python 3.10+, numpy, pandas + pyarrow, pytest. anomalib is a Colab-only runtime dependency reached through an injected backend, never imported in CI.

## Why the real anomalib backend is not in this plan

PatchCore's substance is the anomalib call: build a coreset memory bank from a category's defect-free training images, then score test images against it. That code cannot be written correctly without anomalib installed (its API differs sharply across the `>=1.1` range) and a GPU to run it. Writing it now would be fabricating an upstream API — the failure the protocol forbids, the same reason the loader waited for the download and the MLLM's model call is injected. This plan builds everything PatchCore needs *around* that call — the full-shot contract, the runner's per-category fit, the map-scaling fix, the adapter orchestration behind an injectable backend — all tested on CPU with a fake backend. The real backend, and the §2 VisA ±1pt reproduction that validates it, are a Colab follow-up.

## Global Constraints

- Compute target is Colab free tier: single T4, 16 GB VRAM, fp16, ~12.7 GB host RAM, ~12h sessions.
  This plan's code runs on CPU; the real anomalib backend is the GPU part.
- **Every test must pass with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`
  installed.** No test and no module imported at load time may require torch, anomalib,
  transformers, open_clip or any YAML library. anomalib is reached only through an injected backend.
- Anomaly maps are HxW **float32, finite, at the input image's native resolution, on the method's
  own consistent scale** — NOT per-image normalised to [0,1] (this plan's contract change).
- Image scores are finite floats; a full-shot method scores a sample only after `fit()` for that
  category. `fit` is called by the runner once per category, before that category's predictions,
  and only when `not method.zero_shot`.
- PatchCore is full-shot: `zero_shot = False`, memory bank built from the `train` split's
  defect-free images.
- Source in `src/vlmab/`, tests in `tests/`.

## File structure

| File | Responsibility |
|---|---|
| `src/vlmab/methods/base.py` | contract: relaxed map docstring, new `fit()` no-op |
| `tests/method_contract.py` | `assert_valid_prediction` drops the [0,1] bound |
| `src/vlmab/methods/baseline.py` | returns a raw-scale map (fixes the degeneracy the [0,1] rule caused) |
| `src/vlmab/eval/runner.py` | calls `fit()` per category for full-shot methods |
| `src/vlmab/methods/patchcore_ref.py` | the adapter, wrapping an injected backend |
| `configs/methods/patchcore_ref.yaml` | pre-registered PatchCore hyperparameters |
| `src/vlmab/methods/registry.py` | register `patchcore_ref` |
| `docs/patchcore-colab-integration.md` | how the real anomalib backend plugs in (Colab) |

Out of scope, with reasons, in the closing section.

---

### Task 1: Relax the map contract and add `fit()` to the method contract

**Files:**
- Modify: `src/vlmab/methods/base.py`
- Modify: `tests/method_contract.py`
- Modify: `docs/protocol.md` (§3 + §4 + Changelog)
- Test: `tests/test_method_contract.py` (created here)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `AnomalyMethod.fit(self, train_images: Iterable[np.ndarray], category: str) -> None` — no-op default
  - `assert_valid_prediction(pred, image)` — now checks finite float32 native-shape map, NOT [0,1]

- [ ] **Step 1: Write the failing test for the contract change**

Create `tests/test_method_contract.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import AnomalyMethod, Prediction


def test_accepts_a_map_outside_zero_one():
    """Raw anomaly scores are not bounded to [0,1]; the contract must accept them."""
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    pred = Prediction(image_score=42.0, anomaly_map=np.full((6, 8), 7.5, dtype=np.float32))
    assert_valid_prediction(pred, img)  # must not raise


def test_still_rejects_a_non_native_shape():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    pred = Prediction(image_score=1.0, anomaly_map=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(AssertionError):
        assert_valid_prediction(pred, img)


def test_still_rejects_a_non_finite_map():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    bad = np.zeros((6, 8), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        assert_valid_prediction(Prediction(image_score=1.0, anomaly_map=bad), img)


def test_still_rejects_a_non_float32_map():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        assert_valid_prediction(Prediction(1.0, np.zeros((6, 8), dtype=np.float64)), img)


def test_fit_is_a_noop_on_the_base_contract():
    """Zero-shot methods inherit fit() and it does nothing."""
    AnomalyMethod().fit(iter([np.zeros((4, 4, 3), dtype=np.uint8)]), "vial")  # must not raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_method_contract.py -q`
Expected: FAIL — `test_accepts_a_map_outside_zero_one` fails on the current `[0,1]` assertion, and
`test_fit_is_a_noop_on_the_base_contract` fails with `AttributeError: 'AnomalyMethod' object has no
attribute 'fit'`.

- [ ] **Step 3: Relax `assert_valid_prediction`**

In `tests/method_contract.py`, update the module docstring and drop the [0,1] line. The whole file
becomes:

```python
"""Reused by every adapter's tests: the one definition of a valid Prediction.

A method that fails this cannot be aggregated — `aggregate._load_pair` requires the map to match
the mask's (native) shape, and the pixel metrics require finite float32 values. The map is on the
method's own consistent scale, NOT per-image normalised to [0,1] (protocol v0.2.6): the pixel
metrics rank pooled pixels across images, so a per-image rescale would destroy that ranking.
"""
import math

import numpy as np

from vlmab.methods.base import Prediction


def assert_valid_prediction(pred: Prediction, image: np.ndarray) -> None:
    assert isinstance(pred, Prediction), type(pred)
    assert isinstance(pred.image_score, float), type(pred.image_score)
    assert math.isfinite(pred.image_score), pred.image_score

    amap = pred.anomaly_map
    assert isinstance(amap, np.ndarray), type(amap)
    assert amap.dtype == np.float32, amap.dtype
    assert amap.shape == image.shape[:2], f"{amap.shape} != native {image.shape[:2]}"
    assert np.isfinite(amap).all(), "anomaly map has non-finite values"
```

- [ ] **Step 4: Update `base.py`**

Replace the module docstring's second line and the `Prediction.anomaly_map` comment, and add
`fit()`. The file becomes:

```python
"""Common method contract: everything is an AnomalyMethod.

predict() returns an image-level score plus a pixel anomaly map at the input resolution, on the
method's own consistent scale (NOT per-image normalised — protocol v0.2.6). Efficiency
instrumentation wraps predict(). Full-shot methods build per-category state in fit().
"""
from dataclasses import dataclass
from typing import Iterable
import numpy as np


class MethodNotRunnable(RuntimeError):
    """Raised by an adapter that cannot run in the current environment (e.g. a model
    wrapper with no weights/GPU/client available). Distinct from a genuine runtime error
    inside predict(), so callers can surface real bugs while handling this cleanly."""


@dataclass
class Prediction:
    image_score: float
    anomaly_map: np.ndarray  # HxW float32, native resolution, method's own consistent scale
    extras: dict | None = None  # tokens/cost for API methods, etc.


class AnomalyMethod:
    """Adapters: winclip.py, anomalyclip.py, adaclip.py, saa.py, mllm.py, patchcore_ref.py."""

    name: str = "base"
    zero_shot: bool = True

    def prepare(self, device: str = "cuda") -> None:
        """Load weights/checkpoints. Called once."""
        raise NotImplementedError

    def fit(self, train_images: Iterable[np.ndarray], category: str) -> None:
        """Build per-category state (e.g. a memory bank) from defect-free training images.

        The runner calls this once per category, before scoring that category, and only when
        `zero_shot` is False. Zero-shot methods do not override it — it is a no-op for them."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError
```

- [ ] **Step 5: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_method_contract.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Amend the protocol**

Append to `docs/protocol.md` §3:

```markdown
- **Full-shot anchors.** A full-shot method (the PatchCore anchor) builds a per-category memory
  bank from that category's defect-free `train` split before scoring its test images. It declares
  `zero_shot = False`; the runner calls `fit(train_images, category)` once per category. This is the
  standard PatchCore regime and is fixed here so the anchor's numbers are comparable across
  categories.
```

Append to §4:

```markdown
- **Anomaly-map scale.** Pixel metrics (P-AUROC, AU-PRO, SegF1) rank pooled pixels across every
  image in a group, so a map's *absolute* scale does not matter but its scale must be *consistent
  across images*. Methods therefore return maps on their own internal scale (e.g. PatchCore's raw
  patch distances), NOT rescaled per image to [0,1]. Per-image [0,1] normalisation would stretch a
  clean image's map to the same range as a defective one and destroy the cross-image ranking the
  metrics depend on; it is used only for visualisation, never for scored maps.
```

Append to the Changelog:

```markdown
- 2026-07-24 — v0.2.6. Two contract changes, both forced by the first full-shot method (PatchCore).
  §3: full-shot anchors fit a per-category memory bank on the train split (`fit()` on the method
  contract, called per category by the runner). §4: anomaly maps are on the method's own consistent
  scale, not per-image normalised to [0,1] — per-image rescaling breaks the cross-image pixel-metric
  ranking. This is an evaluation-affecting change (it alters the pixel numbers a per-image-normalised
  method would have produced), decided before any full-shot or real anomalib number exists.
```

- [ ] **Step 7: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. The MLLM's 0/1 grid maps and the baseline's current [0,1] maps both still satisfy
the relaxed contract (both are finite float32), so nothing breaks yet. Record the count.

- [ ] **Step 8: Commit**

```bash
git add src/vlmab/methods/base.py tests/method_contract.py tests/test_method_contract.py docs/protocol.md
git commit -m "feat: add fit() to the method contract; relax maps to a consistent scale (v0.2.6)"
```

---

### Task 2: IntensityBaseline returns a raw-scale map

The [0,1] rule made the baseline degenerate: `normalise_to_unit` forced every varied image's map
max to 1.0, so `image_score = amap.max()` was a constant 1.0 and I-AUROC came out exactly 0.5 by
ties (confirmed on real Vial). With maps now on a raw scale, the score varies and the floor stops
being an artifact.

**Files:**
- Modify: `src/vlmab/methods/baseline.py`
- Modify: `tests/test_baseline.py`
- Modify: `src/vlmab/methods/postprocess.py` (docstring note only)

**Interfaces:**
- Consumes: the relaxed contract (Task 1).
- Produces: `IntensityBaseline.predict` returns a raw `abs(gray - gray.mean())` map, `image_score`
  its max — both unbounded, both varying per image.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_baseline.py` entirely with:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.baseline import IntensityBaseline


@pytest.fixture
def method():
    m = IntensityBaseline()
    m.prepare(device="cpu")
    return m


def _image(shape=(40, 60), fill=100, blob=None):
    img = np.full((*shape, 3), fill, dtype=np.uint8)
    if blob is not None:
        (y0, y1, x0, x1), value = blob
        img[y0:y1, x0:x1] = value
    return img


def test_prediction_satisfies_the_contract(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    assert_valid_prediction(method.predict(img, "vial"), img)


def test_map_is_native_resolution(method):
    img = _image(shape=(31, 47))
    assert method.predict(img, "vial").anomaly_map.shape == (31, 47)


def test_a_flat_image_scores_below_a_spotted_one(method):
    plain = _image()
    spotted = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(plain, "vial").image_score < method.predict(spotted, "vial").image_score


def test_two_images_with_different_contrast_get_different_scores(method):
    """The bug this fixes: `normalise_to_unit` made every varied image score exactly 1.0, so
    I-AUROC was 0.5 by ties. Raw scores must differ with contrast."""
    faint = _image(blob=((5, 10, 5, 10), 150))
    strong = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(faint, "vial").image_score != method.predict(strong, "vial").image_score


def test_a_strong_blob_scores_above_one(method):
    """Proves the map is raw, not per-image normalised to [0,1]: a 255-on-100 blob deviates by
    ~140, far above 1.0."""
    assert method.predict(_image(blob=((5, 10, 5, 10), 255)), "vial").image_score > 1.0


def test_the_blob_is_the_brightest_region_of_the_map(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    amap = method.predict(img, "vial").anomaly_map
    assert amap[7, 7] > amap[30, 50]


def test_a_perfectly_flat_image_gives_an_all_zero_map(method):
    amap = method.predict(_image(), "vial").anomaly_map
    assert (amap == 0.0).all()


def test_prepare_is_a_noop_and_needs_no_gpu():
    IntensityBaseline().prepare(device="cpu")
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_baseline.py -q`
Expected: FAIL — `test_two_images_with_different_contrast_get_different_scores` and
`test_a_strong_blob_scores_above_one` fail against the current normalised baseline (both scores are
1.0).

- [ ] **Step 3: Implement the raw-scale baseline**

Replace the body of `src/vlmab/methods/baseline.py` below its module docstring with:

```python
import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction


class IntensityBaseline(AnomalyMethod):
    name = "intensity_baseline"
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        """No weights to load."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)
        deviation = np.abs(gray - gray.mean()).astype(np.float32)
        # Raw deviation, NOT normalised to [0,1]: per-image rescaling would make every varied
        # image's score identical (the old bug: I-AUROC 0.5 by ties) and break the cross-image
        # ranking the pixel metrics depend on (protocol v0.2.6).
        return Prediction(image_score=float(deviation.max()), anomaly_map=deviation)
```

(The module docstring at the top of the file is unchanged. The `normalise_to_unit` import is
removed.)

- [ ] **Step 4: Add a guard-rail note to `postprocess.py`**

`normalise_to_unit` is no longer used by any method. It stays as a general utility, but its
docstring must warn against the misuse this plan just removed. In `src/vlmab/methods/postprocess.py`,
change `normalise_to_unit`'s docstring first line to:

```python
    """Min-max `x` into [0,1] as float32. All-equal input -> all zeros (no signal).

    Do NOT use this to scale an anomaly map for scoring: per-image [0,1] normalisation breaks the
    cross-image pixel-metric ranking (protocol v0.2.6). It is for genuinely bounded quantities and
    visualisation only.
    """
```

- [ ] **Step 5: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_baseline.py -q`
Expected: PASS (8 passed)

- [ ] **Step 6: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

- [ ] **Step 7: Commit**

```bash
git add src/vlmab/methods/baseline.py tests/test_baseline.py src/vlmab/methods/postprocess.py
git commit -m "fix: IntensityBaseline returns raw-scale maps, ending the constant-score degeneracy"
```

---

### Task 3: Runner fits full-shot methods per category

**Files:**
- Modify: `src/vlmab/eval/runner.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: `AnomalyMethod.fit` (Task 1); `dataset.samples(split, category)` and
  `dataset.load_image` (existing).
- Produces: `run_evaluation(..., fit_split: str = "train")` — before scoring a category, a full-shot
  method (`not method.zero_shot`) is fitted on that category's `fit_split` images.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
def test_runner_fits_a_full_shot_method_once_per_category_before_predicting(tmp_path, fake_dataset):
    """A full-shot method must see a category's train images (via fit) before it scores that
    category's test images."""
    events = []

    class _FullShot(AnomalyMethod):
        name = "fullshot"
        zero_shot = False

        def prepare(self, device="cuda"):
            events.append(("prepare", None))

        def fit(self, train_images, category):
            events.append(("fit", category, len(list(train_images))))

        def predict(self, image, category):
            events.append(("predict", category))
            return Prediction(image_score=0.5, anomaly_map=np.zeros((8, 8), dtype=np.float32))

    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _FullShot(), store, {"seed": 0}, categories=["alpha"])

    assert events[0] == ("prepare", None)
    assert events[1] == ("fit", "alpha", 3)          # FakeDataset yields 3 train images per category
    assert all(e[0] == "predict" for e in events[2:])  # every fit precedes its predicts


def test_runner_does_not_fit_a_zero_shot_method(tmp_path, fake_dataset, counting_method):
    """counting_method is zero_shot; its fit() must never be called."""
    calls = []
    original = counting_method.fit

    def _spy(train_images, category):
        calls.append(category)
        return original(train_images, category)

    counting_method.fit = _spy
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
    assert calls == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -k "full_shot or zero_shot" -q`
Expected: FAIL — `test_runner_fits_a_full_shot_method_once_per_category_before_predicting` fails
because no `("fit", ...)` event is recorded.

- [ ] **Step 3: Implement**

In `src/vlmab/eval/runner.py`, add `fit_split: str = "train"` to the `run_evaluation` signature
(next to `split`). Then, inside the `for category in todo:` loop, immediately after the line
`for category in todo:` and before the `rows` list is created, insert:

```python
        if not method.zero_shot:
            # Full-shot anchor: build this category's memory bank from its defect-free train
            # images before scoring it (protocol §3). Passed as a lazy generator so the backend
            # streams them rather than holding a category of 2.66 MP images in memory at once.
            train_images = (
                dataset.load_image(s) for s in dataset.samples(fit_split, category)
            )
            method.fit(train_images, category)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Existing runner tests use zero-shot methods, so `fit` is skipped for them and they
are unaffected. Record the count.

- [ ] **Step 6: Commit**

```bash
git add src/vlmab/eval/runner.py tests/test_runner.py
git commit -m "feat: runner fits full-shot methods per category before scoring"
```

---

### Task 4: PatchCore adapter over an injectable backend

**Files:**
- Modify: `src/vlmab/methods/patchcore_ref.py` (replaces the stub)
- Modify: `configs/methods/patchcore_ref.yaml`
- Create: `tests/test_patchcore_ref.py`

**Interfaces:**
- Consumes: `AnomalyMethod`, `Prediction`, `MethodNotRunnable` (base); `upsample_to` (postprocess);
  `assert_valid_prediction` (Task 1).
- Produces:
  - A backend protocol: an object with `fit(train_images: Iterable[np.ndarray]) -> None` and
    `score(image: np.ndarray) -> tuple[float, np.ndarray]` (image score, anomaly map at any
    resolution).
  - `PatchCoreRef(backend=None)` with `name = "patchcore_ref"`, `zero_shot = False`.

- [ ] **Step 1: Pre-register the hyperparameters**

Replace `configs/methods/patchcore_ref.yaml` entirely with:

```yaml
name: patchcore_ref
zero_shot: false
# Reference anchor (Roth et al., CVPR 2022). Provenance: anomalib implementation (protocol §3
# priority 2). The exact anomalib version is pinned and recorded here on first Colab run.
source: anomalib
anomalib_version: ">=1.1"        # exact pin resolved at integration; record the resolved version
backbone: wide_resnet50_2         # canonical PatchCore backbone
layers: [layer2, layer3]          # mid-level feature layers
coreset_sampling_ratio: 0.1       # 10% greedy coreset
num_neighbors: 9
fit_split: train                  # memory bank from the defect-free train split
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_patchcore_ref.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.patchcore_ref import PatchCoreRef


class _FakeBackend:
    """Stands in for the real anomalib-backed backend. Records fit, returns a canned score+map."""

    def __init__(self, coarse=(7, 7)):
        self.fitted_images = None
        self._coarse = coarse

    def fit(self, train_images):
        self.fitted_images = list(train_images)

    def score(self, image):
        # A raw (unbounded) image score plus a coarse map with one hot cell — like anomalib.
        m = np.zeros(self._coarse, dtype=np.float32)
        m[1, 2] = 4.2
        return 4.2, m


def test_is_full_shot():
    assert PatchCoreRef().zero_shot is False


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        PatchCoreRef().prepare(device="cpu")


def test_fit_delegates_the_training_images_to_the_backend():
    backend = _FakeBackend()
    m = PatchCoreRef(backend=backend)
    m.prepare(device="cpu")
    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    m.fit(iter(imgs), "vial")
    assert len(backend.fitted_images) == 3


def test_predict_returns_a_native_resolution_raw_map():
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    m.fit(iter([img]), "vial")
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)              # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(4.2)   # raw score passed through, not rescaled
    assert pred.anomaly_map[15, 25] == pytest.approx(4.2)  # the hot cell, upsampled to native
    assert pred.anomaly_map[0, 0] == 0.0


def test_predict_before_fit_is_a_real_error_not_a_not_runnable():
    """Predicting a category that was never fitted is an orchestration bug — it must surface,
    not be swallowed as 'cannot run here'."""
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    with pytest.raises(RuntimeError) as exc:
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
    assert not isinstance(exc.value, MethodNotRunnable)


def test_predict_for_a_different_category_than_fitted_is_an_error():
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    m.fit(iter([np.zeros((8, 8, 3), dtype=np.uint8)]), "vial")
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "cable")
```

- [ ] **Step 3: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_patchcore_ref.py -q`
Expected: FAIL — `ImportError: cannot import name 'PatchCoreRef' from 'vlmab.methods.patchcore_ref'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/patchcore_ref.py` entirely with:

```python
"""PatchCore reference anchor (Roth et al., CVPR 2022) — the full-shot ceiling.

PatchCore's substance is the anomalib call: build a coreset memory bank from a category's
defect-free training images, then score test patches against it. That code needs anomalib and a
GPU, so it lives behind an injected backend and is written on Colab (docs/patchcore-colab-
integration.md). Everything here — the full-shot orchestration, the native-resolution map, the
predict-before-fit guard — is tested with a fake backend.

The backend is any object with:
    fit(train_images: Iterable[np.ndarray]) -> None
    score(image: np.ndarray) -> tuple[float, np.ndarray]   # (raw image score, raw anomaly map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution,
never per-image normalised.
"""
from typing import Iterable

import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class PatchCoreRef(AnomalyMethod):
    name = "patchcore_ref"
    zero_shot = False

    def __init__(self, backend=None):
        self._backend = backend
        self._fitted_category = None

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real anomalib backend
        is Colab-only (it needs anomalib and a GPU), so this reports that cleanly."""
        if self._backend is None:  # pragma: no cover - needs anomalib + GPU
            raise MethodNotRunnable(
                "PatchCoreRef needs an anomalib backend; build it in a GPU session with anomalib "
                "installed and inject it (docs/patchcore-colab-integration.md)"
            )

    def fit(self, train_images: Iterable[np.ndarray], category: str) -> None:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("PatchCoreRef has no backend; inject one or build it on GPU")
        self._backend.fit(train_images)
        self._fitted_category = category

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("PatchCoreRef has no backend; inject one or build it on GPU")
        if self._fitted_category != category:
            raise RuntimeError(
                f"predict on {category!r} but the memory bank was fitted on "
                f"{self._fitted_category!r}; the runner must fit() this category first"
            )
        raw_score, raw_map = self._backend.score(image)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"fitted_category": category},
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_patchcore_ref.py -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/patchcore_ref.py configs/methods/patchcore_ref.yaml tests/test_patchcore_ref.py
git commit -m "feat: PatchCore adapter over an injectable backend, with full-shot orchestration"
```

---

### Task 5: Register PatchCore and document the Colab backend

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `tests/test_registry.py`
- Create: `docs/patchcore-colab-integration.md`

**Interfaces:**
- Consumes: `PatchCoreRef` (Task 4).
- Produces: `build_method("patchcore_ref")` returns a `PatchCoreRef`; `available()` includes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_builds_the_patchcore_anchor():
    from vlmab.methods.patchcore_ref import PatchCoreRef

    assert isinstance(build_method("patchcore_ref"), PatchCoreRef)
    assert "patchcore_ref" in available()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -k patchcore -q`
Expected: FAIL — `KeyError: "unknown method 'patchcore_ref'..."`

- [ ] **Step 3: Register it**

In `src/vlmab/methods/registry.py`, add the import and the entry. The `_REGISTRY` dict becomes:

```python
from vlmab.methods.base import AnomalyMethod
from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.mllm import QwenMLLM
from vlmab.methods.patchcore_ref import PatchCoreRef

_REGISTRY: dict[str, Callable[[], AnomalyMethod]] = {
    "intensity_baseline": IntensityBaseline,
    "mllm_qwen": QwenMLLM,          # CPU-usable only with an injected client; prepare() gates the rest
    "patchcore_ref": PatchCoreRef,  # CPU-usable only with an injected backend; prepare() gates the rest
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Document the Colab backend contract**

Create `docs/patchcore-colab-integration.md`:

```markdown
# PatchCore anomalib backend — Colab integration

The `PatchCoreRef` adapter (`src/vlmab/methods/patchcore_ref.py`) wraps an injected backend so its
full-shot orchestration is tested on CPU. The real backend runs anomalib on a GPU and is built in a
Colab session, not in this repository's tests — anomalib's API differs across the pinned range and
cannot be exercised without a GPU.

## The backend contract

Inject any object with these two methods into `PatchCoreRef(backend=...)`:

    fit(train_images: Iterable[np.ndarray]) -> None
        Build the coreset memory bank from a category's defect-free training images (HxWx3 uint8,
        as the loader yields them). Called once per category by the runner.

    score(image: np.ndarray) -> tuple[float, np.ndarray]
        Return (raw image-level score, raw anomaly map). The map may be at any resolution; the
        adapter upsamples it to the image's native resolution. Neither value is per-image
        normalised (protocol v0.2.6) — return anomalib's raw scores.

## Hyperparameters

Frozen in `configs/methods/patchcore_ref.yaml`: `wide_resnet50_2`, layers `layer2`+`layer3`,
coreset ratio 0.1, 9 neighbours. Record the resolved anomalib version there on the first run.

## Validation gate (protocol §2)

Before any MVTec AD 2 number is reported, the backend must reproduce PatchCore's published VisA
image-AUROC within ±1.0 point, run through this same adapter and the project's own metrics. A
failure to reproduce flags the anchor in every table.

## Running it

Once the backend is built and injected, everything else is already wired:
`scripts/run_eval.py --method patchcore_ref ...` fits per category and scores through the runner,
exactly as `intensity_baseline` does today. Registered with no backend, `patchcore_ref` reports
cleanly via `MethodNotRunnable` (a `prepare()` on CPU exits 1), so the CLI never pretends it ran.
```

- [ ] **Step 6: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/registry.py tests/test_registry.py docs/patchcore-colab-integration.md
git commit -m "feat: register PatchCore anchor; document the Colab anomalib backend contract"
```

---

## Out of scope for this plan, and why

- **The real anomalib PatchCore backend** — the `fit`/`score` implementation against anomalib on a
  GPU. It cannot be written correctly without anomalib installed and a GPU to run it, and the
  protocol forbids fabricating an upstream API. Written on Colab against the pinned version, then
  validated against the §2 VisA ±1pt reproduction gate (docs/patchcore-colab-integration.md).
- **The four remaining GPU wrappers** (WinCLIP, AnomalyCLIP, AdaCLIP, SAA+) — each its own
  per-method plan on Colab. They are all zero-shot, so none needs the full-shot `fit` path this
  plan added; they slot into the existing framework.
- **The Colab notebook and the evaluation-server submission writer** — as before, the notebook is
  orchestration meaningful once a GPU backend runs, and the submission writer waits on the server
  registration (docs/datasets-access.md).
