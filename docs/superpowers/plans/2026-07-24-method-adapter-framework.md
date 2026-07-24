# Method Adapter Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the adapter framework every method plugs into — the map post-processing that satisfies the native-resolution contract, a reusable contract test, a dependency-free Baseline that exercises the pipeline end to end, per-sample latency and `extras` capture in the runner, and the full Qwen2.5-VL-3B parser — so that adding a model wrapper is wiring one inference call, not designing an adapter.

**Architecture:** Everything here is verifiable on CPU without model weights. Shared post-processing (upsample-to-native, normalise) is pure numpy. Adapters implement the existing `AnomalyMethod` contract. The MLLM adapter takes its model call as an injectable boundary, so the substance — prompt formatting, response parsing, grid-to-map, parse-failure policy — is tested with canned strings and never needs a GPU. The five upstream model wrappers (WinCLIP, AnomalyCLIP, AdaCLIP, SAA+, PatchCore) are out of scope and get per-method plans written against their pinned repos on Colab, because their exact inference code cannot be written correctly without the repo and a GPU in front of you.

**Tech Stack:** Python 3.10+, numpy, pandas + pyarrow, pytest. Torch/transformers/open_clip/anomalib stay out of CI: every adapter reaches them through a lazy import or an injected boundary.

## Why the model wrappers are not in this plan

Five of the six adapters wrap an external repository at a pinned commit and run real weights on a GPU. Writing their `predict()` inference bodies now would mean inventing upstream API calls that cannot be run or checked on this machine — the exact failure mode the pre-registered protocol exists to prevent, and the one that already bit this project on split names and the evaluation resolution. They follow the same precedent as the MVTec AD 2 loader (which waited for the download) and the submission writer (which waits for the server): built when the thing they depend on is actually present. This plan builds the frame they slot into. **First up next: PatchCore via anomalib** — the most stable API, the anchor that calibrates every table, and half of the §2 VisA reproduction gate that closes M2.

## Global Constraints

- Compute target is Colab free tier: single T4, 16 GB VRAM, fp16 only, ~12.7 GB host RAM, sessions
  disconnect at ~12 hours. This plan's code runs on CPU; the deferred wrappers are the GPU part.
- **Every test must pass with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`
  installed.** No test and no module imported at load time may require torch, transformers,
  open_clip or anomalib. Model libraries are reached only through a lazy import inside `prepare()`
  or through an injected callable.
- Anomaly maps are returned at the **input image's native resolution**, HxW float32 in [0,1]
  (`docs/protocol.md` §4; `aggregate._load_pair` hard-fails on any shape mismatch).
- Image-level scores are finite floats. A method that cannot produce a score for a sample records
  a defined no-information value and flags it, never NaN and never a silent 0 (protocol §6: failures
  are reported, not dropped).
- MLLM baseline is `Qwen/Qwen2.5-VL-3B-Instruct`, fp16, ONE fixed prompt and rubric for every
  dataset (protocol §3; the prompt lives in `configs/methods/mllm_qwen.yaml` and is not edited here).
- Source in `src/vlmab/`, tests in `tests/`.

## The contract every adapter implements (already in `src/vlmab/methods/base.py`)

```python
@dataclass
class Prediction:
    image_score: float
    anomaly_map: np.ndarray  # HxW float32 in [0,1]
    extras: dict | None = None

class AnomalyMethod:
    name: str = "base"
    zero_shot: bool = True
    def prepare(self, device: str = "cuda") -> None: ...
    def predict(self, image: np.ndarray, category: str) -> Prediction: ...
```

`image` is HxWx3 uint8 at native resolution, as `MVTecAD2.load_image` returns it. The runner saves
`prediction.anomaly_map` and, after Task 3, records latency and `extras`.

## File structure

| File | Responsibility |
|---|---|
| `src/vlmab/methods/postprocess.py` | `upsample_to`, `normalise_to_unit` — the shared native-res + [0,1] plumbing |
| `src/vlmab/methods/baseline.py` | `IntensityBaseline` — dependency-free floor that exercises the pipeline |
| `src/vlmab/methods/mllm.py` | `QwenMLLM` — injectable model boundary, response parser, grid-to-map |
| `src/vlmab/methods/registry.py` | name → adapter, so `run_eval.py` can instantiate a method |
| `tests/method_contract.py` | `assert_valid_prediction` — reused by every adapter test |
| `scripts/run_eval.py` | wire the runner: (method, dataset) → shards end to end |

Out of scope, with reasons, in the closing section.

---

### Task 1: Map post-processing — native resolution and unit range

**Files:**
- Create: `src/vlmab/methods/postprocess.py`
- Create: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `upsample_to(coarse: np.ndarray, size: tuple[int, int]) -> np.ndarray` — nearest-block upsample
    a `HcxWc` map to `size=(H, W)`, returning float32. Exact for any target size.
  - `normalise_to_unit(x: np.ndarray) -> np.ndarray` — min-max to [0,1] as float32; an all-equal
    input maps to all-zeros (no signal); raises `ValueError` on any non-finite value.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_postprocess.py`:

```python
import numpy as np
import pytest

from vlmab.methods.postprocess import normalise_to_unit, upsample_to


def test_upsample_hits_the_exact_target_size():
    out = upsample_to(np.zeros((7, 7), dtype=np.float32), (1400, 1900))
    assert out.shape == (1400, 1900)
    assert out.dtype == np.float32


def test_upsample_is_nearest_block_not_interpolated():
    coarse = np.array([[0.0, 1.0]], dtype=np.float32)  # 1x2
    out = upsample_to(coarse, (1, 4))
    # Left half maps to the 0 cell, right half to the 1 cell; no in-between values.
    assert out.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_upsample_preserves_a_single_hot_cell_as_a_block():
    coarse = np.zeros((7, 7), dtype=np.float32)
    coarse[1, 2] = 1.0  # cell "B3"
    out = upsample_to(coarse, (70, 70))
    assert set(np.unique(out)) == {0.0, 1.0}
    assert out[15, 25] == 1.0                 # inside the B3 block (rows 10-19, cols 20-29)
    assert out[0, 0] == 0.0


def test_upsample_handles_a_target_not_divisible_by_the_grid():
    coarse = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = upsample_to(coarse, (10, 10))       # 10 is not a multiple of 3
    assert out.shape == (10, 10)
    assert out.min() == 0.0 and out.max() == 8.0  # no new values invented


def test_normalise_maps_min_to_zero_and_max_to_one():
    out = normalise_to_unit(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert out.dtype == np.float32


def test_normalise_a_constant_array_is_all_zero_not_nan():
    out = normalise_to_unit(np.full((4, 4), 3.0, dtype=np.float32))
    assert (out == 0.0).all()


def test_normalise_rejects_non_finite_values():
    with pytest.raises(ValueError):
        normalise_to_unit(np.array([0.0, np.nan, 1.0], dtype=np.float32))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.methods.postprocess'`

- [ ] **Step 3: Implement**

Create `src/vlmab/methods/postprocess.py`:

```python
"""Map post-processing shared by every adapter.

Two jobs, both mandated by the protocol: bring an anomaly map to the input image's *native*
resolution (§4 evaluates pixel metrics there against unmodified masks, so a coarse or resized
map has to be expanded, never the mask shrunk), and bring scores into [0,1].
"""
import numpy as np


def upsample_to(coarse: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-block upsample `coarse` to `size=(H, W)`, exact for any target.

    Each output pixel takes the value of the coarse cell it falls in, so no value the method
    did not emit is ever invented — the right behaviour for a hard localisation claim like the
    MLLM's 7x7 grid. Smooth interpolation is a per-method choice left to the wrappers that
    produce dense feature maps (they have torch's interpolate for it); this default stays
    numpy-only and faithful.
    """
    coarse = np.asarray(coarse, dtype=np.float32)
    gh, gw = coarse.shape
    h, w = size
    ys = (np.arange(h) * gh) // h
    xs = (np.arange(w) * gw) // w
    return coarse[ys][:, xs]


def normalise_to_unit(x: np.ndarray) -> np.ndarray:
    """Min-max `x` into [0,1] as float32. All-equal input -> all zeros (no anomaly signal)."""
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("map contains non-finite values; a NaN/inf map is a method bug")
    lo = float(x.min())
    hi = float(x.max())
    if hi == lo:
        return np.zeros_like(x)
    return ((x - lo) / (hi - lo)).astype(np.float32)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/methods/postprocess.py tests/test_postprocess.py
git commit -m "feat: map post-processing — native-resolution upsample and unit normalisation"
```

---

### Task 2: Contract harness and the Baseline adapter

A dependency-free adapter that passes the same contract every real method must, so the whole
pipeline — method → runner → shard → aggregate — can be exercised in CI, and so results have an
honest floor to sit above.

**Files:**
- Create: `tests/method_contract.py`
- Modify: `src/vlmab/methods/baseline.py` (created here; not currently present)
- Create: `tests/test_baseline.py`

**Interfaces:**
- Consumes: `Prediction`, `AnomalyMethod` from `vlmab.methods.base`; `normalise_to_unit` (Task 1).
- Produces:
  - `tests/method_contract.py::assert_valid_prediction(pred, image)` — raises `AssertionError`
    unless `pred` is a `Prediction` with a finite float score and an `image.shape[:2]` float32
    map in [0,1].
  - `IntensityBaseline(AnomalyMethod)` with `name = "intensity_baseline"`, `zero_shot = True`.

- [ ] **Step 1: Write the contract harness**

Create `tests/method_contract.py`:

```python
"""Reused by every adapter's tests: the one definition of a valid Prediction.

A method that fails this cannot be aggregated — `aggregate._load_pair` requires the map to match
the mask's (native) shape, and every metric requires values in [0,1].
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
    assert float(amap.min()) >= 0.0 and float(amap.max()) <= 1.0, (amap.min(), amap.max())
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_baseline.py`:

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


def test_a_uniform_image_scores_lower_than_one_with_a_bright_blob(method):
    plain = _image()
    spotted = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(plain, "vial").image_score < method.predict(spotted, "vial").image_score


def test_the_blob_is_the_brightest_region_of_the_map(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    amap = method.predict(img, "vial").anomaly_map
    assert amap[7, 7] > amap[30, 50]  # inside the blob vs. the plain background


def test_a_perfectly_flat_image_gives_an_all_zero_map(method):
    amap = method.predict(_image(), "vial").anomaly_map
    assert (amap == 0.0).all()


def test_prepare_is_a_noop_and_needs_no_gpu():
    IntensityBaseline().prepare(device="cpu")  # must not raise
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_baseline.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.methods.baseline'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/baseline.py` entirely with:

```python
"""IntensityBaseline: the honest floor.

Not a serious detector — it flags pixels whose brightness deviates from the image mean, which
catches nothing subtle. It exists so the full pipeline (method -> runner -> shard -> aggregate)
runs in CI against a real `AnomalyMethod`, and so every results table has a trivial baseline to
sit above: a zero-shot/VLM method that cannot beat "the odd bright pixel" is telling you
something. Pure numpy, no weights, no GPU.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction
from vlmab.methods.postprocess import normalise_to_unit


class IntensityBaseline(AnomalyMethod):
    name = "intensity_baseline"
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        """No weights to load."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)
        deviation = np.abs(gray - gray.mean())
        amap = normalise_to_unit(deviation)
        return Prediction(image_score=float(amap.max()), anomaly_map=amap)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_baseline.py -q`
Expected: PASS (6 passed)

- [ ] **Step 6: Commit**

```bash
git add tests/method_contract.py src/vlmab/methods/baseline.py tests/test_baseline.py
git commit -m "feat: adapter contract harness and dependency-free IntensityBaseline"
```

---

### Task 3: Runner records latency and method extras

The efficiency module exists but nothing calls it, and the runner drops `Prediction.extras` — so
the §4 efficiency table and the MLLM's tokens/cost have nowhere to come from. This wires per-sample
latency into result rows and carries `extras` through. VRAM stays out: it is a method-level number
measured once in the M5 efficiency pass, not per sample.

**Files:**
- Modify: `src/vlmab/eval/runner.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: `run_evaluation` as it stands, `Prediction.extras`.
- Produces: result rows gain `latency_ms` (float) and `extras_<key>` columns for each key a method
  puts in `Prediction.extras` (absent when `extras` is None).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_runner.py`:

```python
def test_runner_records_a_positive_latency_per_row(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha"))
    assert "latency_ms" in df.columns
    assert (df["latency_ms"] >= 0).all()
    assert df["latency_ms"].notna().all()


def test_runner_carries_prediction_extras_into_prefixed_columns(tmp_path, fake_dataset):
    """API methods report tokens+cost via extras (protocol §4); the shard must keep them."""
    class _ExtrasMethod(AnomalyMethod):
        name = "extras"
        def prepare(self, device="cuda"):
            pass
        def predict(self, image, category):
            return Prediction(image_score=0.5, anomaly_map=np.zeros((8, 8), dtype=np.float32),
                              extras={"tokens": 123, "parse_ok": True})

    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _ExtrasMethod(), store, {"seed": 0}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "extras", "alpha"))
    assert list(df["extras_tokens"]) == [123, 123, 123]
    assert list(df["extras_parse_ok"]) == [True, True, True]


def test_runner_omits_extras_columns_when_a_method_returns_none(tmp_path, fake_dataset,
                                                                counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha"))
    assert not [c for c in df.columns if c.startswith("extras_")]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -k "latency or extras" -q`
Expected: FAIL — `latency_ms` not in columns / `KeyError: 'extras_tokens'`

- [ ] **Step 3: Implement**

In `src/vlmab/eval/runner.py`, add `import time` to the import block at the top of the file
(next to the existing `import hashlib`/`import re`). Then, in the per-sample loop, replace:

```python
        for sample in dataset.samples(split, category):
            image = dataset.load_image(sample)
            prediction = method.predict(image, category)

            row: dict[str, Any] = {
                "image_path": str(sample.image_path),
                "label": int(sample.label),
                "image_score": float(prediction.image_score),
                "split": split,
                # Where the ground truth lives, so aggregation can compute pixel metrics from
                # the shard alone instead of re-walking the dataset for a second time.
                "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
            }
```

with:

```python
        for sample in dataset.samples(split, category):
            image = dataset.load_image(sample)
            start = time.perf_counter()
            prediction = method.predict(image, category)
            latency_ms = (time.perf_counter() - start) * 1000.0

            row: dict[str, Any] = {
                "image_path": str(sample.image_path),
                "label": int(sample.label),
                "image_score": float(prediction.image_score),
                "split": split,
                # Where the ground truth lives, so aggregation can compute pixel metrics from
                # the shard alone instead of re-walking the dataset for a second time.
                "mask_path": str(sample.mask_path) if sample.mask_path is not None else None,
                # Per-sample wall time. Recorded on every run and tagged with the GPU in `meta`,
                # but only *reported* from the fixed reference machine (protocol §5).
                "latency_ms": latency_ms,
            }
            # API methods report tokens/cost here (protocol §4); flatten under an extras_ prefix.
            if prediction.extras is not None:
                row.update({f"extras_{k}": v for k, v in prediction.extras.items()})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS — everything from before plus the new tests. Nothing may fail or error.

- [ ] **Step 6: Commit**

```bash
git add src/vlmab/eval/runner.py tests/test_runner.py
git commit -m "feat: runner records per-sample latency and carries method extras into rows"
```

---

### Task 4: Qwen2.5-VL-3B MLLM adapter

The MLLM's substance is not the model call — it is turning a free-text response into a score and a
localisation map, and deciding what happens when the model does not answer in the required form.
All of that is tested here with canned strings; the model itself is an injected callable, wired to
real Qwen weights only in `prepare()` on Colab.

**Files:**
- Modify: `src/vlmab/methods/mllm.py` (replaces the stub)
- Create: `tests/test_mllm.py`
- Modify: `docs/protocol.md` (§3 note + Changelog)

**Interfaces:**
- Consumes: `Prediction`, `AnomalyMethod`; `upsample_to` (Task 1); `assert_valid_prediction` (Task 2).
- Produces:
  - `parse_mllm_response(text: str) -> tuple[float, list[str], bool]` — `(score, cells, parse_ok)`
  - `cells_to_grid(cells: list[str]) -> np.ndarray` — a 7x7 float32 grid, 1.0 at each valid cell
  - `PROMPT: str` — the one fixed prompt, `{category}` the only substituted token
  - `QwenMLLM(model_client=None)` implementing the contract, where
    `model_client: Callable[[np.ndarray, str], str]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mllm.py`:

```python
import json
from pathlib import Path

import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.mllm import PROMPT, QwenMLLM, cells_to_grid, parse_mllm_response


def _client(response):
    return lambda image, prompt: response


def test_parses_a_clean_response():
    text = json.dumps({"anomaly_probability": 0.82, "cells": ["B3", "B4"], "reason": "scratch"})
    score, cells, ok = parse_mllm_response(text)
    assert ok is True
    assert score == pytest.approx(0.82)
    assert cells == ["B3", "B4"]


def test_finds_json_embedded_in_prose():
    text = 'Sure! Here is my assessment:\n{"anomaly_probability": 0.1, "cells": []}\nHope that helps.'
    score, cells, ok = parse_mllm_response(text)
    assert ok is True and score == pytest.approx(0.1) and cells == []


def test_clamps_an_out_of_range_probability():
    score, _, ok = parse_mllm_response('{"anomaly_probability": 1.7, "cells": []}')
    assert ok is True and score == 1.0


def test_drops_invalid_cell_labels_but_keeps_valid_ones():
    _, cells, ok = parse_mllm_response('{"anomaly_probability": 0.5, "cells": ["B3", "Z9", "h2", "A1"]}')
    assert ok is True and cells == ["B3", "A1"]


def test_unparseable_response_is_flagged_not_guessed():
    score, cells, ok = parse_mllm_response("the image looks fine to me, no JSON here")
    assert ok is False
    assert score == 0.5   # no-information score for AUROC, not a silent 0
    assert cells == []


def test_missing_probability_field_is_a_parse_failure():
    score, cells, ok = parse_mllm_response('{"cells": ["A1"]}')
    assert ok is False and score == 0.5 and cells == []


def test_cells_to_grid_sets_the_named_cells():
    grid = cells_to_grid(["A1", "G7"])
    assert grid.shape == (7, 7)
    assert grid[0, 0] == 1.0 and grid[6, 6] == 1.0
    assert grid.sum() == 2.0


def test_cells_to_grid_empty_is_all_zero():
    assert cells_to_grid([]).sum() == 0.0


def test_predict_scores_and_localises_a_defect():
    text = json.dumps({"anomaly_probability": 0.9, "cells": ["D4"]})
    m = QwenMLLM(model_client=_client(text))
    m.prepare(device="cpu")
    image = np.zeros((70, 70, 3), dtype=np.uint8)
    pred = m.predict(image, "vial")
    assert_valid_prediction(pred, image)
    assert pred.image_score == pytest.approx(0.9)
    assert pred.anomaly_map[35, 35] == 1.0     # centre falls in the D4 block
    assert pred.anomaly_map[0, 0] == 0.0
    assert pred.extras["parse_ok"] is True


def test_predict_on_an_unparseable_response_flags_it_and_stays_valid():
    m = QwenMLLM(model_client=_client("no json"))
    m.prepare(device="cpu")
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    pred = m.predict(image, "vial")
    assert_valid_prediction(pred, image)
    assert pred.image_score == 0.5
    assert (pred.anomaly_map == 0.0).all()
    assert pred.extras["parse_ok"] is False


def test_predict_puts_the_category_in_the_prompt():
    seen = {}
    def client(image, prompt):
        seen["prompt"] = prompt
        return '{"anomaly_probability": 0.0, "cells": []}'
    m = QwenMLLM(model_client=client)
    m.prepare(device="cpu")
    m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "fruit_jelly")
    assert "fruit_jelly" in seen["prompt"]


def test_predict_without_a_client_or_prepare_fails_loudly():
    m = QwenMLLM()  # no injected client, prepare() not called
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")


def test_adapter_prompt_stays_in_sync_with_the_published_config():
    """§3 requires the prompt be published in config for auditability; the adapter embeds its
    own runtime copy to avoid a YAML dependency in CI. This fails if the two drift apart."""
    config = (Path(__file__).resolve().parents[1] / "configs/methods/mllm_qwen.yaml").read_text()
    for line in PROMPT.splitlines():
        assert line.strip() in config, f"prompt line not found in config: {line!r}"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_mllm.py -q`
Expected: FAIL — `ImportError: cannot import name 'QwenMLLM' from 'vlmab.methods.mllm'`

- [ ] **Step 3: Implement**

Replace `src/vlmab/methods/mllm.py` entirely with:

```python
"""Qwen2.5-VL-3B baseline: a generalist MLLM prompted as an inspector (protocol §3).

The model call is the least interesting part and the only part that needs a GPU, so it is an
injected boundary: `model_client(image, prompt) -> str`. Everything that decides the numbers —
formatting the fixed prompt, parsing the response, turning a 7x7 cell grid into a native-
resolution map, and what to do when the model does not answer in the required form — lives here
and is tested without weights. `prepare()` builds the real client on Colab.

The prompt is embedded here as the single runtime copy, so no YAML parser is pulled into the
test environment for it. `configs/methods/mllm_qwen.yaml` stays the published, auditable copy
protocol §3 requires; `test_adapter_prompt_stays_in_sync_with_the_published_config` fails if the
two drift apart. `{category}` is the only substituted token and every other brace is literal, so
substitution is a plain `.replace`, not `str.format`.

Parse-failure policy (protocol §3, dated amendment): a response we cannot parse carries no
information, so its image score is 0.5 — the no-information point for AUROC — never a silent 0
that would read as "confidently normal", and never NaN. The failure is recorded in
`Prediction.extras["parse_ok"] = False` so the parse-failure *rate* is itself a reported number,
which is exactly the MLLM failure taxonomy the paper's analysis calls for.
"""
import json
import re
from typing import Callable

import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction
from vlmab.methods.postprocess import upsample_to

_GRID = 7
_CELL = re.compile(r"^([A-G])([1-7])$")
_NO_INFO_SCORE = 0.5

PROMPT = (
    "You are an industrial quality inspector. Look at this image of a {category}.\n"
    "1) Is there any defect or anomaly? Answer with a probability between 0.00 and 1.00.\n"
    "2) If yes, describe its location using a 7x7 grid (rows A-G, columns 1-7).\n"
    'Respond as JSON: {"anomaly_probability": <float>, "cells": ["B3", ...], "reason": "<short>"}'
)


def cells_to_grid(cells: list[str]) -> np.ndarray:
    """A 7x7 float32 grid, 1.0 at each valid `<row A-G><col 1-7>` cell, 0 elsewhere."""
    grid = np.zeros((_GRID, _GRID), dtype=np.float32)
    for cell in cells:
        m = _CELL.match(cell)
        if m:
            grid[ord(m.group(1)) - ord("A"), int(m.group(2)) - 1] = 1.0
    return grid


def parse_mllm_response(text: str) -> tuple[float, list[str], bool]:
    """`(score, cells, parse_ok)` from a model response.

    Tolerant of prose around the JSON and of a probability out of range; strict about the one
    field that carries the score. `parse_ok=False` returns the no-information score and no cells.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            prob = obj["anomaly_probability"]
            score = float(np.clip(float(prob), 0.0, 1.0))
            raw = obj.get("cells", []) or []
            cells = [c for c in raw if isinstance(c, str) and _CELL.match(c)]
            return score, cells, True
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return _NO_INFO_SCORE, [], False


class QwenMLLM(AnomalyMethod):
    name = "mllm_qwen"
    zero_shot = True

    def __init__(self, model_client: Callable[[np.ndarray, str], str] | None = None):
        self._client = model_client

    def prepare(self, device: str = "cuda") -> None:
        """With an injected client there is nothing to load. Otherwise the real Qwen client is
        built here — a Colab-only path that imports transformers lazily, never in CI."""
        if self._client is None:  # pragma: no cover - needs a GPU and transformers
            raise RuntimeError(
                "no model_client injected and real-client construction is Colab-only; "
                "inject a callable for CPU use or build the Qwen client in a GPU session"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._client is None:
            raise RuntimeError("QwenMLLM has no model_client; inject one or call prepare() on GPU")
        response = self._client(image, PROMPT.replace("{category}", category))
        score, cells, parse_ok = parse_mllm_response(response)
        amap = upsample_to(cells_to_grid(cells), image.shape[:2])
        return Prediction(
            image_score=score,
            anomaly_map=amap,
            extras={"parse_ok": parse_ok, "n_cells": len(cells), "raw": response[:500]},
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_mllm.py -q`
Expected: PASS (13 passed)

- [ ] **Step 5: Record the parse-failure policy in the protocol**

Append to `docs/protocol.md` §3:

```markdown
- **MLLM response parsing.** The Qwen baseline returns free text. A response is parsed for the
  required JSON (`anomaly_probability` plus a `cells` list on the 7x7 grid); prose around the JSON
  and an out-of-range probability are tolerated, invalid cell labels are dropped. A response that
  cannot be parsed carries no information, so its image score is 0.5 (the no-information point for
  AUROC), never a silent 0 and never NaN, and the failure is recorded per sample
  (`extras.parse_ok = False`) so the parse-failure rate is a reported number. This policy is fixed
  for every dataset and category, like the prompt itself.
```

Append to the Changelog:

```markdown
- 2026-07-24 — v0.2.5. Records the MLLM response-parsing and parse-failure policy in §3 (score 0.5
  and a per-sample flag on an unparseable response, so parse failures are reported rather than read
  as confident normals). Fixes it before any MLLM number is produced. No other evaluation rule changed.
```

- [ ] **Step 6: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS.

```bash
git add src/vlmab/methods/mllm.py tests/test_mllm.py docs/protocol.md
git commit -m "feat: Qwen2.5-VL-3B adapter — injectable model, robust parser, grid-to-map"
```

---

### Task 5: Method registry and the run_eval CLI

Wire the pieces into a command that runs a named method over a dataset, so the pipeline is usable
end to end rather than only from tests. The registry lists only adapters that run without a GPU —
the deferred wrappers register themselves when their plans land.

**Files:**
- Create: `src/vlmab/methods/registry.py`
- Modify: `scripts/run_eval.py` (replaces the stub)
- Create: `tests/test_registry.py`
- Create: `tests/test_run_eval.py`

**Interfaces:**
- Consumes: `IntensityBaseline` (Task 2), `QwenMLLM` (Task 4), `MVTecAD2`, `ResultStore`,
  `run_evaluation`, `run_meta`.
- Produces:
  - `registry.build_method(name: str) -> AnomalyMethod` — raises `KeyError` with the known names
    on an unknown one.
  - `registry.available() -> list[str]`
  - `scripts/run_eval.py::main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_registry.py`:

```python
import pytest

from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.registry import available, build_method


def test_builds_a_known_method():
    assert isinstance(build_method("intensity_baseline"), IntensityBaseline)


def test_unknown_method_lists_the_known_ones():
    with pytest.raises(KeyError) as exc:
        build_method("winclip")           # a deferred wrapper: not registered yet
    assert "intensity_baseline" in str(exc.value)


def test_available_is_sorted_and_nonempty():
    names = available()
    assert names == sorted(names) and "intensity_baseline" in names
```

Create `tests/test_run_eval.py`:

```python
import pandas as pd

from mvtec_tree import build_category
from run_eval import main
from vlmab.eval.store import ResultStore


def test_run_eval_writes_shards_for_a_real_method(tmp_path):
    build_category(tmp_path, "vial")
    results = tmp_path / "results"
    code = main(["--method", "intensity_baseline", "--root", str(tmp_path),
                 "--split", "test_public", "--results", str(results), "--seed", "0"])
    assert code == 0
    df = ResultStore(results).load_all()
    assert len(df) > 0
    assert set(df["method"]) == {"intensity_baseline"}
    assert "latency_ms" in df.columns


def test_run_eval_rejects_an_unknown_method(tmp_path, capsys):
    build_category(tmp_path, "vial")
    code = main(["--method", "nope", "--root", str(tmp_path), "--results", str(tmp_path / "r")])
    assert code == 1
    assert "intensity_baseline" in capsys.readouterr().out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py tests/test_run_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.methods.registry'` /
`No module named 'run_eval'`

- [ ] **Step 3: Implement the registry**

Create `src/vlmab/methods/registry.py`:

```python
"""Name -> adapter, for the CLI and the Colab notebook.

Only GPU-free adapters are registered here today. Each deferred model wrapper (WinCLIP,
AnomalyCLIP, AdaCLIP, SAA+, PatchCore) adds its own entry when its plan lands, so an unknown
name fails with the list of what actually runs rather than a promise.
"""
from typing import Callable

from vlmab.methods.base import AnomalyMethod
from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.mllm import QwenMLLM

_REGISTRY: dict[str, Callable[[], AnomalyMethod]] = {
    "intensity_baseline": IntensityBaseline,
    "mllm_qwen": QwenMLLM,   # CPU-usable only with an injected client; prepare() gates the rest
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_method(name: str) -> AnomalyMethod:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    return _REGISTRY[name]()
```

- [ ] **Step 4: Implement the CLI**

Replace `scripts/run_eval.py` entirely with:

```python
#!/usr/bin/env python
"""Run one method over a dataset, writing per-category result shards.

    python scripts/run_eval.py --method intensity_baseline --root data/mvtec_ad2 \
        --split test_public --results results/shards --seed 0
"""
import argparse
import sys
from pathlib import Path
from typing import Sequence

from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.provenance import run_meta
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.methods.registry import available, build_method


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--split", default="test_public")
    parser.add_argument("--category", default=None)
    parser.add_argument("--maps-dir", default=None, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    try:
        method = build_method(args.method)
    except KeyError as exc:
        print(exc.args[0])
        print(f"available methods: {available()}")
        return 1

    method.prepare(device=args.device)
    dataset = MVTecAD2(args.root)
    store = ResultStore(args.results)
    meta = run_meta({"method": args.method, "split": args.split}, seed=args.seed)

    written = run_evaluation(
        dataset, method, store, meta,
        categories=[args.category] if args.category else None,
        split=args.split,
        maps_dir=args.maps_dir,
        device=args.device,
    )
    print(f"{args.method}: wrote {len(written)} shard(s) to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py tests/test_run_eval.py -q`
Expected: PASS (5 passed)

Note: `test_run_eval` builds a synthetic tree and runs `intensity_baseline` with `device="cuda"`
passed but never touched (the baseline's `prepare`/`predict` ignore the device), so it runs on CPU.

- [ ] **Step 6: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS — every test to date plus the new ones.

- [ ] **Step 7: Verify the CLI against the real Vial category**

The pipeline now runs a real method end to end. Do it for real, on the data already on disk:

```bash
/tmp/vlmab-venv/bin/python scripts/run_eval.py --method intensity_baseline \
    --root data/mvtec_ad2 --split test_public --category vial \
    --results /tmp/vlmab-run/shards --maps-dir /tmp/vlmab-run/maps --seed 0
```

Expected: exit 0, `wrote 1 shard(s)`. Then confirm the shard aggregates and that
`intensity_baseline` scores near chance on Vial (it is a floor, not a detector):

```bash
/tmp/vlmab-venv/bin/python -c "
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore('/tmp/vlmab-run/shards').load_all()
out = aggregate(df, by='meta_lighting')
print(out[['meta_lighting', 'i_auroc', 'au_pro_030', 'n']].to_string(index=False))
"
```

Record the output in your report. A near-0.5 I-AUROC is the expected, correct result — it confirms
the whole path (loader → baseline → runner → shard → per-condition aggregation) works and that the
floor behaves like a floor. `/tmp` is fine here: these outputs are a few MB, not the dataset.

- [ ] **Step 8: Commit**

```bash
git add src/vlmab/methods/registry.py scripts/run_eval.py tests/test_registry.py tests/test_run_eval.py
git commit -m "feat: method registry and run_eval CLI — pipeline runs a named method end to end"
```

---

## Out of scope for this plan, and why

- **The five model wrappers** — WinCLIP, AnomalyCLIP, AdaCLIP, SAA+, PatchCore. Each wraps an
  external repo at a pinned commit and runs real weights on a GPU; its `predict()` inference body
  cannot be written correctly without the repo and the hardware present, and inventing it is the
  fabrication the protocol forbids. Each gets a per-method plan written on Colab, starting with
  **PatchCore via anomalib** (stable API, the anchor, half the §2 VisA reproduction gate). Their
  configs already exist under `configs/methods/`; each registers itself in `registry.py` and passes
  the same `assert_valid_prediction` contract plus a GPU-gated smoke test.
- **The Colab notebook.** Orchestration only, meaningful once at least one GPU wrapper runs; it
  needs the per-category download URLs behind the MVTec account.
- **The evaluation-server submission writer** and the official metric-name mapping — still behind
  the pending registration (`docs/datasets-access.md`).
- **Per-method VRAM and the efficiency table.** Latency per sample lands here (Task 3); peak VRAM
  and the frozen efficiency table are the M5 pass on the fixed reference instance (protocol §5).
