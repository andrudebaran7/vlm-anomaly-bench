# SAA+ Training-Free Adapter Implementation Plan

> **Split execution.** Tasks 1–4 are CPU-verifiable and run by subagents (TDD / docs). The Colab
> phases (A–D) need two official repos, GroundingDINO + SAM weights and a GPU — none in the dev
> environment — so they run **interactively in a Colab GPU session**. SAA+ has no packaged API: its
> backend is derived by reading the official repo's inference entry point at a pinned commit, so the
> Colab phases give the integration recipe with VERIFY steps, not verified packaged code. Fabricating
> an unpackaged repo's internals is exactly what the protocol forbids. Steps use checkbox (`- [ ]`)
> syntax.
>
> **Run this plan after** `docs/superpowers/plans/2026-07-29-adaclip-zeroshot-adapter.md`: that plan
> re-points the registry's unknown-method example to `"saa"`, and Task 3 here moves it again.
>
> For the CPU tasks only: REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the SAA+ training-free adapter over an injectable backend (CPU-testable), pre-register both the per-category prompt table (a dated protocol §3 amendment) and the anomaly-map provenance decision, add a map-granularity diagnostic and a measured cost probe, and — on Colab — wrap the official repo and clear the §2 VisA reproduction gate before any MVTec AD 2 number.

**Architecture:** SAA+ (Cao et al., *Segment Any Anomaly without Training*) is a training-free cascade: GroundingDINO proposes regions from defect language prompts, SAM refines them into masks. No method-trained weights — two frozen foundation models, so **no auxiliary-training overlap audit is needed**. Three things make it unlike the other adapters: its prompts are per-object by design (a §3 amendment), its output is region masks rather than a dense field (we take the repo's own final map verbatim), and it is the heaviest and slowest run in the study (a measured cost probe replaces the untested "may not fit 12h" assumption). The backend seam takes a category — substantively, since the category selects the domain prompts.

**Tech Stack:** Python 3.10+, numpy, pytest (CPU tasks); Colab T4 GPU, the official `caoyunkang/Segment-Any-Anomaly` repo at a pinned commit, GroundingDINO + SAM checkpoints, torch (Colab phases).

**Spec:** `docs/superpowers/specs/2026-07-29-adaclip-saa-design.md`

## The three decisions that govern this plan

**1. Per-category prompts — a dated §3 amendment (v0.2.9).** Protocol §3 requires prompts taken
verbatim from the original papers with *no per-category prompt engineering*. SAA+'s central
contribution — hybrid prompt regularization — is per-object domain knowledge: defect language
expressions and object-specific area/size constraints. Running SAA+ without them is not SAA+; running
it with them violates the letter of §3. **Decision: amend §3 to exempt SAA+ explicitly**, on the
condition that its per-object prompts are taken **verbatim from the official repo at the pinned
commit**, never tuned by us and never adjusted after seeing a result, and that the full prompt table is
committed **before any scoring run**.

**2. The anomaly map is not dense.** SAA+ produces region masks with confidence scores, not a
continuous per-pixel field, and our pixel metrics sweep thresholds. **Decision: take the repo's own
final anomaly map verbatim.** Compositing our own would be a re-implementation (§3 priority 3, flagged
in every table) and would risk failing the gate on our composition rather than on the method. A
granularity diagnostic records how few distinct levels the map has, feeding a stated limitation in the
paper's §5.2.

**3. Compute.** The protocol flags SAA+ as the run that may not fit a 12h Colab session. Existing
infrastructure already covers that: the runner is resumable and the store is crash-safe, with the
category as simultaneously the download, resume and shard unit — SAA+ needs to fit N resumable
sessions, not one. What is missing, and what Phase C adds, is a **measured cost probe** whose three
recorded figures (per image, per category, full grid) any later scope reduction must cite.

## Global Constraints

- CPU tasks: every test passes with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`. No test and no module imported at load time may require torch, the SAA+ repo, GroundingDINO, SAM or any YAML library. The repos/checkpoints are reached only through the injected backend.
- **Test command for every CPU step:** `.venv/bin/python -m pytest`. If `.venv` does not exist, create it once with:
  `uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install numpy scipy scikit-learn pandas pyarrow pillow pytest && VIRTUAL_ENV=.venv uv pip install -e . --no-deps`
- Baseline before starting: **231 tests pass** (after the AdaCLIP plan; 223 before it). The suite must stay green after every task.
- SAA+ is **zero-shot and training-free** (`zero_shot = True`, no fit, no method-trained weights). No overlap audit — but SAA+ still gets a row in the paper's §3.1 table stating *training-free, no auxiliary data*, so the table has no silent gaps.
- **Provenance is the official repo at a pinned commit** (protocol §3 priority 1 — anomalib does not ship it). **Two** checkpoints are pinned with sha256, not one: GroundingDINO and SAM are independent artefacts with independent versions.
- Maps and scores are RAW (protocol v0.2.6); `SaaRef` upsamples the map to native resolution (protocol §4). Never per-image normalise, never smooth.
- **The §2 VisA reproduction gate is a hard acceptance criterion.** Gate metric, pre-registered: published VisA image-AUROC ±1.0; if SAA+'s paper does not publish it, the first of **I-AP, then I-F1max** that it does publish, with the metric actually used recorded explicitly. If it publishes no image-level VisA metric at all, SAA+ is reported as *ungated* in every table (protocol §2), not silently exempted.

## Files

| File | Responsibility |
|---|---|
| `src/vlmab/methods/saa.py` | `SaaRef` adapter over an injectable backend (CPU) |
| `configs/methods/saa.yaml` | pre-registered config incl. both checkpoints (CPU) |
| `configs/methods/saa_prompts.yaml` | the pre-registered per-category prompt table (CPU skeleton, Colab-filled) |
| `src/vlmab/methods/postprocess.py` | `distinct_levels` map-granularity diagnostic (CPU) |
| `src/vlmab/methods/registry.py` | register `saa` (CPU) |
| `tests/test_saa.py` | adapter tests against a fake backend (CPU) |
| `tests/test_postprocess.py` | diagnostic tests (CPU) |
| `tests/test_registry.py` | registry test + final re-pointing of the unknown-method example (CPU) |
| `docs/protocol.md` | §3 prompt amendment + map-provenance note + Changelog v0.2.9 (CPU) |
| `src/vlmab/methods/saa_backend.py` | the official-repo backend, lazy import (Colab) |
| `results/reproduction/saa_visa.md` | the recorded VisA reproduction table + gate metric used (Colab) |
| `results/saa/cost_probe.md` | the three measured cost figures (Colab) |

---

## Task 1: SaaRef adapter over an injectable backend

**Files:**
- Modify: `src/vlmab/methods/saa.py` (replaces the stub `Saa` class entirely)
- Modify: `configs/methods/saa.yaml` (replaces the two-line stub entirely)
- Create: `tests/test_saa.py`

**Interfaces:**
- Consumes: `AnomalyMethod`, `Prediction`, `MethodNotRunnable` (`vlmab.methods.base`); `upsample_to` (`vlmab.methods.postprocess`); `assert_valid_prediction` (`tests/method_contract.py`).
- Produces:
  - A backend protocol: an object with `score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]`.
  - `SaaRef(backend=None)` with `name = "saa"`, `zero_shot = True`.

- [ ] **Step 1: Pre-register the config**

Replace `configs/methods/saa.yaml` entirely with:

```yaml
name: saa
zero_shot: true
aux_trained: false               # training-free cascade; no method-trained weights, no auxiliary data
# SAA+ (Cao et al., "Segment Any Anomaly without Training"). anomalib does not ship it, so provenance
# is the official repo (protocol §3 priority 1). Pin the commit on the first Colab run.
source: https://github.com/caoyunkang/Segment-Any-Anomaly
commit: unpinned_until_first_colab_run     # record the exact SHA here
# TWO independent frozen artefacts, each pinned separately: swapping either moves every pixel metric
# without touching a line of code.
grounding_dino_checkpoint: unpinned_until_first_colab_run
grounding_dino_sha256: unpinned_until_first_colab_run
sam_checkpoint: unpinned_until_first_colab_run
sam_sha256: unpinned_until_first_colab_run
# Per-category prompts, taken VERBATIM from the repo at the pinned commit — never tuned by us, never
# adjusted after seeing a result. Protocol §3 amendment v0.2.9. Table: configs/methods/saa_prompts.yaml
prompts: per_category_verbatim_from_official_repo
prompts_file: configs/methods/saa_prompts.yaml
# The anomaly map is the repo's OWN final map, taken verbatim. Not recomposed, not smoothed
# (protocol §3: any composition of ours would be a re-implementation, priority 3).
anomaly_map_source: official_repo_final_map
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_saa.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.saa import SaaRef


class _FakeBackend:
    """Stands in for the real SAA+ cascade. The category is substantive here — it selects the
    per-object domain prompts — so the fake records it. The returned map mimics the real output
    shape: a near-binary field built from a couple of mask regions, not a smooth field.

    The score and the map's maximum are deliberately DIFFERENT values: if they matched, the
    image-score assertion below would still pass under an adapter bug that ignored the backend's
    score and returned the map's maximum instead. Passing the backend's score through untouched is
    the one substantive thing this adapter decides, so the fake has to be able to catch that."""

    def __init__(self):
        self.calls = []

    def score(self, image, category):
        self.calls.append(category)
        m = np.zeros((16, 16), dtype=np.float32)
        m[2:5, 2:5] = 0.81       # one high-confidence mask region
        m[10:12, 9:13] = 0.44    # one lower-confidence region
        return 0.93, m           # NOT the map max — see the class docstring


def test_is_zero_shot_and_training_free():
    m = SaaRef()
    assert m.zero_shot is True
    assert m.name == "saa"


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        SaaRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    """The category is substantive: it selects SAA+'s per-object domain prompts."""
    backend = _FakeBackend()
    m = SaaRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "can")
    assert backend.calls == ["vial", "can"]


def test_predict_returns_a_native_resolution_raw_map():
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(0.93)   # the BACKEND's score, not the map's max (0.81)
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native


def test_predict_does_not_smooth_or_renormalise_the_map():
    """The repo's own map is taken verbatim: upsampling only. A post-process no other method in the
    study receives would break comparability (protocol §3)."""
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((16, 16, 3), dtype=np.uint8), "vial")
    # Same size in and out, so upsample_to is identity and the values must survive untouched.
    assert pred.anomaly_map.max() == pytest.approx(0.81)
    assert pred.anomaly_map.min() == pytest.approx(0.0)
    assert sorted(np.unique(pred.anomaly_map).tolist()) == pytest.approx([0.0, 0.44, 0.81], abs=1e-6)


def test_predict_records_the_category_in_extras():
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "walnuts")
    assert pred.extras == {"category": "walnuts"}


def test_predict_without_a_backend_is_not_runnable():
    m = SaaRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_saa.py -q`
Expected: FAIL — `ImportError: cannot import name 'SaaRef' from 'vlmab.methods.saa'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/saa.py` entirely with:

```python
"""SAA+ training-free adapter (Cao et al., "Segment Any Anomaly without Training").

SAA+ is a cascade of two frozen foundation models: GroundingDINO proposes regions from defect
language prompts, SAM refines them into masks. Nothing is trained by the method, so — unlike
AnomalyCLIP and AdaCLIP — there is no auxiliary-training overlap audit. Two consequences shape this
adapter:

1. The category is substantive. SAA+'s contribution is hybrid prompt regularization: per-object domain
   knowledge (defect language expressions, object-specific area/size constraints). The category
   selects those prompts, so the seam takes it. Those prompts are taken VERBATIM from the official
   repo at the pinned commit (configs/methods/saa_prompts.yaml) — protocol §3 amendment v0.2.9, which
   exempts SAA+ from the no-per-category-prompts rule precisely because they are the method, not a
   tuning knob of ours.

2. The map is not dense. SAA+ emits region masks with confidence scores, so its anomaly map has very
   few distinct levels. We take the repo's OWN final map verbatim and only upsample it: compositing
   masks ourselves would be a re-implementation (protocol §3 priority 3, flagged), and smoothing it
   would be a post-process no other method in the study receives. Map granularity is reported instead,
   via postprocess.distinct_levels, and carried as a stated limitation in the paper's §5.2.

The backend is any object with:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class SaaRef(AnomalyMethod):
    name = "saa"
    zero_shot = True

    def __init__(self, backend=None):
        self._backend = backend

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real cascade is
        Colab-only (it needs the official repo plus GroundingDINO and SAM weights and a GPU),
        reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the repo + two checkpoints + GPU
            raise MethodNotRunnable(
                "SaaRef needs an SAA+ backend; build it in a GPU session from the official repo at "
                "the pinned commit, with both the GroundingDINO and SAM checkpoints, and inject it "
                "(see the SAA+ Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("SaaRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_saa.py -q`
Expected: PASS (7 passed)

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (238 passed — 231 baseline + 7 new). Record the count.

```bash
git add src/vlmab/methods/saa.py configs/methods/saa.yaml tests/test_saa.py
git commit -m "feat: SAA+ training-free adapter over an injectable backend (category selects prompts)"
```

---

## Task 2: The map-granularity diagnostic

SAA+'s map is built from a handful of mask regions, so it has very few distinct values. That depresses
AU-PRO in a way that reflects the output *format* as much as real localisation quality — a confound
that must be reported, not discovered by a reviewer. This makes the diagnostic a tested function rather
than an ad-hoc Colab cell, so the number in the paper is reproducible.

**Files:**
- Modify: `src/vlmab/methods/postprocess.py` (append; do not touch `upsample_to` or `normalise_to_unit`)
- Modify: `tests/test_postprocess.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `distinct_levels(amap: np.ndarray) -> int` in `vlmab.methods.postprocess`, used by Colab
  phase B.3 and by the paper's §5.2 limitation.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_postprocess.py`:

```python
def test_distinct_levels_counts_unique_values():
    from vlmab.methods.postprocess import distinct_levels

    amap = np.array([[0.0, 0.0, 0.5], [0.5, 0.9, 0.9]], dtype=np.float32)
    assert distinct_levels(amap) == 3


def test_distinct_levels_of_a_constant_map_is_one():
    from vlmab.methods.postprocess import distinct_levels

    assert distinct_levels(np.zeros((8, 8), dtype=np.float32)) == 1


def test_distinct_levels_separates_a_mask_map_from_a_continuous_one():
    """The whole point: a few-region SAA+-style map has orders of magnitude fewer levels than a
    continuous patch-based map of the same size, and that difference is what gets reported."""
    from vlmab.methods.postprocess import distinct_levels

    mask_style = np.zeros((32, 32), dtype=np.float32)
    mask_style[2:8, 2:8] = 0.7
    mask_style[20:24, 20:28] = 0.3

    rng = np.random.default_rng(0)
    continuous = rng.random((32, 32)).astype(np.float32)

    assert distinct_levels(mask_style) == 3
    assert distinct_levels(continuous) > 100


def test_distinct_levels_counts_exact_float32_values_without_bucketing():
    """Two values a float32 tick apart count as two, not one: no tolerance bucketing, so the number
    is unambiguous and reproducible across machines."""
    from vlmab.methods.postprocess import distinct_levels

    amap = np.array([[np.float32(0.1), np.float32(0.1) + np.float32(1e-7)]], dtype=np.float32)
    assert distinct_levels(amap) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -k distinct_levels -q`
Expected: FAIL — `ImportError: cannot import name 'distinct_levels' from 'vlmab.methods.postprocess'`

- [ ] **Step 3: Implement**

Append to `src/vlmab/methods/postprocess.py`:

```python
def distinct_levels(amap: np.ndarray) -> int:
    """How many distinct values an anomaly map contains — a diagnostic, never a metric.

    Mask-based methods (SAA+: GroundingDINO region proposals refined by SAM) emit a map composed of a
    handful of constant-confidence regions, so it has a handful of distinct levels. Patch-based methods
    emit a near-continuous field. Every threshold-sweeping pixel metric (P-AUROC, AU-PRO, SegF1) is
    sensitive to that difference, so a low count is a confound to report alongside the metric rather
    than a result to explain away (paper §5.2).

    Counts exact distinct float32 values with no tolerance bucketing, so the figure is unambiguous and
    reproducible.
    """
    return int(np.unique(np.asarray(amap, dtype=np.float32)).size)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_postprocess.py -q`
Expected: PASS — the four new tests plus the existing ones.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (242 passed). Record the count.

```bash
git add src/vlmab/methods/postprocess.py tests/test_postprocess.py
git commit -m "feat: distinct_levels map-granularity diagnostic for mask-based methods"
```

---

## Task 3: Register SAA+ and close the registry chain

Registering `saa` invalidates the registry's unknown-method test, which the AdaCLIP plan re-pointed to
`"saa"`. **This is the end of the chain**: with `saa` registered, every method this study plans is
registered, so there is no deferred wrapper left to borrow. The test moves to a permanent sentinel name
instead, and the "swap this later" comment goes away.

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: `SaaRef` (Task 1).
- Produces: `build_method("saa")` returns a `SaaRef`; `available()` includes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_builds_saa():
    from vlmab.methods.saa import SaaRef

    assert isinstance(build_method("saa"), SaaRef)
    assert "saa" in available()


def test_every_planned_method_is_registered():
    """The registry docstring promised an entry per deferred wrapper as each plan landed. With SAA+
    registered they all have one, so this pins the full set and makes an accidental removal fail."""
    assert set(available()) == {
        "adaclip",
        "anomalyclip",
        "intensity_baseline",
        "mllm_qwen",
        "patchcore_ref",
        "saa",
        "winclip",
    }
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_registry.py -k "saa or planned" -q`
Expected: FAIL — `KeyError: "unknown method 'saa'; available: [...]"`

- [ ] **Step 3: Register it and move the unknown-method test to a permanent sentinel**

In `src/vlmab/methods/registry.py`, add the import in alphabetical position among the method imports
(after the `patchcore_ref` import):

```python
from vlmab.methods.saa import SaaRef
```

and add the `_REGISTRY` entry, keeping the dict alphabetically ordered (between `patchcore_ref` and
`winclip`):

```python
    "saa": SaaRef,                  # CPU-usable only with an injected backend; prepare() gates the rest
```

Update the module docstring's last sentence, which still describes wrappers as pending. Replace:

```python
its own entry when its plan lands, so an unknown name fails with the list of what actually runs
rather than a promise.
```

with:

```python
its own entry when its plan lands. As of the SAA+ plan every planned method is registered, so an
unknown name fails with the list of what actually runs rather than a promise.
```

Then, in `tests/test_registry.py`, `test_unknown_method_lists_the_known_ones` calls
`build_method("saa")` — now registered, and no unregistered method name is left to borrow. Replace that
call and the comment above it with a permanent sentinel:

```python
        # a name that is deliberately not a method and never will be: every planned method is now
        # registered, so borrowing a real one would make this test fail as each plan lands.
        build_method("not_a_method_sentinel")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q`
Expected: PASS — `test_builds_saa`, `test_every_planned_method_is_registered`, and the re-pointed
`test_unknown_method_lists_the_known_ones`.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (244 passed). Record the count.

```bash
git add src/vlmab/methods/registry.py tests/test_registry.py
git commit -m "feat: register SAA+; close the registry chain with a permanent sentinel name"
```

---

## Task 4: The prompt amendment and the map-provenance note (protocol v0.2.9)

The pre-registration deliverable for SAA+. This is the first time the protocol concedes something
substantive for a single method, so the amendment states the condition that makes it defensible —
prompts verbatim from the repo, committed before any scoring — and the prompt-table file is created
here so the Colab phase has a structure to fill rather than a blank page.

**Files:**
- Create: `configs/methods/saa_prompts.yaml`
- Modify: `docs/protocol.md` (§3 + Changelog)

**Interfaces:**
- Consumes: the three decisions in this plan's header.
- Produces: no code symbols. `configs/methods/saa_prompts.yaml` is read by a human and filled in
  Colab phase A.3; nothing imports it (no YAML dependency in CI).

- [ ] **Step 1: Create the pre-registered prompt table skeleton**

Create `configs/methods/saa_prompts.yaml`:

```yaml
# SAA+ per-category prompts — PRE-REGISTERED (protocol §3 amendment v0.2.9)
#
# SAA+'s contribution is hybrid prompt regularization: per-object domain knowledge, consisting of
# defect language expressions plus object-specific property constraints (anomaly region area, size,
# position). Those prompts ARE the method, which is why protocol §3's no-per-category-prompts rule is
# amended for SAA+ rather than applied to it.
#
# THE CONDITION THAT MAKES THE AMENDMENT DEFENSIBLE — read before editing this file:
#   1. Every value below is copied VERBATIM from the official repo at the pinned commit recorded in
#      configs/methods/saa.yaml. Cite the exact file and line range you copied it from.
#   2. Nothing here is written, reworded or tuned by us.
#   3. Nothing here is changed after seeing any result, on any dataset. If a prompt turns out to be
#      wrong, the fix is to re-read the repo and record a dated correction — not to improve the prompt.
#   4. This file is committed BEFORE the first scoring run (Colab phase A.3, before phase C).
#
# If the repo has no prompt for one of our categories (MVTec AD 2 categories are not MVTec AD classic
# categories), do NOT invent one. Record `prompt: NOT_PUBLISHED_IN_REPO` and the fallback the repo's
# own code uses for an unknown object, and note it in results/reproduction/saa_visa.md — an invented
# prompt would be exactly the per-category engineering §3 forbids.

source_commit: unpinned_until_first_colab_run
source_files: unrecorded            # e.g. "SAA/prompts.py L12-L88" — the exact provenance
recorded_on: unrecorded             # date the values below were copied

categories:
  can:          {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  fabric:       {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  fruit_jelly:  {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  rice:         {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  sheet_metal:  {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  vial:         {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  wallplugs:    {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}
  walnuts:      {prompt: unrecorded, property_constraints: unrecorded, copied_from: unrecorded}

# VisA objects, for the §2 reproduction gate. Fill from the repo the same way.
visa_categories: unrecorded
```

- [ ] **Step 2: Amend the protocol §3**

Append to the `## 3. Methods & implementations` section of `docs/protocol.md`, after the AdaCLIP bullet:

```markdown
- **SAA+ (training-free; per-category prompts, by amendment).** SAA+ is a cascade of two frozen
  foundation models (GroundingDINO region proposals refined by SAM). Nothing is trained by the method,
  so no auxiliary-training overlap audit applies — SAA+ appears in the §3.1 table as *training-free, no
  auxiliary data*, so that table has no silent gaps. **This bullet amends §3's "no per-category prompt
  engineering" rule for SAA+ alone.** SAA+'s contribution is hybrid prompt regularization: per-object
  defect language expressions and object-specific property constraints. Running it with a single generic
  prompt would measure a method its own paper does not report. The exemption is conditional: the
  per-object prompts are taken **verbatim from the official repo at the pinned commit**, recorded with
  their exact file provenance in configs/methods/saa_prompts.yaml, committed before the first scoring
  run, and never adjusted after seeing a result. Where the repo publishes no prompt for one of our
  categories, that is recorded as such — no prompt is invented.
- **Mask-based anomaly maps (SAA+).** SAA+ emits region masks with confidence scores rather than a
  dense per-pixel field, so its anomaly map has few distinct levels, and every threshold-sweeping pixel
  metric is sensitive to that. The map is taken **verbatim from the repo's own inference output** and
  only upsampled: recomposing it from masks would be a re-implementation (priority 3), and smoothing it
  would be a post-process no other method in this study receives. Map granularity is instead measured
  (`postprocess.distinct_levels`) and reported alongside SAA+'s pixel metrics as a stated confound.
- **Two pinned checkpoints for SAA+.** GroundingDINO and SAM are independent artefacts with independent
  versions; both are pinned by sha256 in configs/methods/saa.yaml. Changing either moves every pixel
  metric without any code change.
```

- [ ] **Step 3: Add the Changelog entry**

Append to the Changelog at the bottom of `docs/protocol.md`:

```markdown
- 2026-07-29 — v0.2.9. Amends §3 for SAA+ only, and states the conditions. (a) Prompts: SAA+'s
  per-object domain prompts are its contribution, not tuning, so the "no per-category prompt
  engineering" rule is lifted for SAA+ on the condition that every prompt is verbatim from the official
  repo at the pinned commit, recorded with file provenance in configs/methods/saa_prompts.yaml,
  committed before the first scoring run, and never changed after seeing a result; unpublished prompts
  are recorded as unpublished, never invented. (b) Map provenance: SAA+'s mask-based map is taken
  verbatim from the repo and only upsampled — never recomposed, never smoothed — with its granularity
  measured and reported as a confound. (c) Two checkpoints (GroundingDINO, SAM) are pinned by sha256,
  not one. Decided before any SAA+ number exists. No evaluation rule for other methods changed.
```

- [ ] **Step 4: Verify and commit**

Run: `grep -n "v0.2.9" docs/protocol.md`
Expected: one hit, the new Changelog entry.

Run: `grep -c "verbatim" docs/protocol.md configs/methods/saa_prompts.yaml`
Expected: at least 1 in each.

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (244 passed — no code changed).

```bash
git add configs/methods/saa_prompts.yaml docs/protocol.md
git commit -m "docs: SAA+ prompt exemption + map-provenance rules; protocol v0.2.9"
```

---

# Colab phases (manual, GPU) — the official-repo backend

SAA+ has no packaged API; the backend is derived by reading the official repo's inference entry point at
the pinned commit. These phases give the recipe and the shape, not verified packaged code — the VERIFY
steps, the smoke test and the gate are the real proof. SAA+ needs **two** model checkpoints, and it is
the heaviest and slowest method in the study, so Phase A checks VRAM headroom and Phase C measures cost
before any grid is committed.

## Phase A — Environment, pinned repo, two checkpoints, the prompt table

- [ ] **A.1 — GPU, our repo, the SAA+ repo pinned**

```python
!nvidia-smi -L && nvidia-smi --query-gpu=memory.total --format=csv
!git clone https://github.com/andrudebaran7/vlm-anomaly-bench.git && cd vlm-anomaly-bench && pip install -e . --quiet
!git clone https://github.com/caoyunkang/Segment-Any-Anomaly.git
%cd Segment-Any-Anomaly && git rev-parse HEAD   # record this SHA into configs/methods/saa.yaml
!pip install -r requirements.txt --quiet
%cd ..
```

Record the pinned SHA: ____ . Record total VRAM: ____ (a T4 is 16 GB; GroundingDINO and SAM are both
resident, so note headroom now — it is the most likely hard failure in this plan).

- [ ] **A.2 — Fetch both checkpoints and record their sha256**

Follow the repo's README for the exact download commands and the exact SAM variant it expects (ViT-H vs
ViT-L vs ViT-B changes both VRAM and every pixel metric — do not substitute).

```python
import hashlib, glob, os
paths = sorted(
    glob.glob("Segment-Any-Anomaly/**/*.pth", recursive=True)
    + glob.glob("Segment-Any-Anomaly/**/*.pt", recursive=True)
    + glob.glob("weights/**/*.pth", recursive=True)
)
for f in paths:
    print(f, round(os.path.getsize(f) / 1e6, 1), "MB",
          hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
```

Record both files, both sizes and both sha256 into `configs/methods/saa.yaml`
(`grounding_dino_*`, `sam_*`).

- [ ] **A.3 — Fill and commit the prompt table BEFORE any scoring**

This is the condition the §3 amendment rests on, so it happens now, not after a run. Find where the
repo defines its per-object prompts and property constraints (grep is the fastest route):

```python
!grep -rn "prompt" Segment-Any-Anomaly --include=*.py | head -40
```

Copy the values **verbatim** into `configs/methods/saa_prompts.yaml`, filling `source_commit`,
`source_files` (exact file and line range), `recorded_on`, and each category's `prompt`,
`property_constraints` and `copied_from`. For any MVTec AD 2 category the repo has no prompt for, set
`prompt: NOT_PUBLISHED_IN_REPO` and record the repo's own unknown-object fallback. **Invent nothing.**

```bash
git add configs/methods/saa_prompts.yaml configs/methods/saa.yaml
git commit -m "docs: record SAA+ prompts verbatim from the pinned repo (pre-registration, §3 v0.2.9)"
```

- [ ] **A.4 — VERIFY the inference path**

Read the repo's inference entry point (its demo/eval script — the README names it). Identify, from the
actual code at this commit:
- how the cascade is constructed and both checkpoints loaded (the classes, the load calls);
- how the per-object prompt and property constraints are passed in for a given object;
- the image preprocessing and input resolution;
- **how the repo produces its final anomaly map and image-level score** — this is the map we take
  verbatim, so record precisely which variable it is and what scale it is on.

Write down those exact calls. Phase B is a thin wrapper around them and they cannot be taken from memory.

## Phase B — The backend

- [ ] **B.1 — Write `src/vlmab/methods/saa_backend.py`**

The skeleton below is the *shape*; fill the repo-specific calls (marked FILL) from A.4.

```python
"""SAA+ official-repo backend for SaaRef (GPU/Colab only).

Wraps the official repo's training-free cascade (GroundingDINO region proposals refined by SAM) behind
the seam score(image, category) -> (raw_score, raw_map). Both checkpoints are loaded once at
construction. The per-object prompt and property constraints come from the repo's own prompt
definitions, keyed by category, recorded verbatim in configs/methods/saa_prompts.yaml (protocol §3
amendment v0.2.9) — this backend selects among them, it never composes them.

The returned map is the repo's OWN final anomaly map, unmodified (protocol v0.2.6 and the §3
map-provenance rule): not recomposed from masks, not smoothed, not per-image normalised. SaaRef only
upsamples it to native resolution.

The repo and torch are imported lazily so this module imports in CI (never constructed there).
"""
import sys
import numpy as np


class SaaBackend:
    def __init__(
        self,
        repo_dir: str,
        grounding_dino_checkpoint: str,
        sam_checkpoint: str,
        device: str = "cuda",
    ):
        import torch

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # FILL from A.4: construct the cascade exactly as the repo's inference script does, loading
        # both checkpoints. Keep the repo's own config object if it uses one.
        self._model = _build_cascade(grounding_dino_checkpoint, sam_checkpoint, device)  # FILL
        # FILL from A.4: the repo's own per-object prompt lookup. Select from it; never build prompts.
        self._prompts = _load_repo_prompts(repo_dir)                                     # FILL
        # FILL from A.4: the repo's own fallback prompt for an object it has no per-object entry for
        # (e.g. its "unknown object" or default prompt) — used, never invented by us. See score() below.
        self._fallback_prompt = _load_repo_fallback_prompt(repo_dir)                     # FILL
        self._device = device

    def score(self, image: np.ndarray, category: str) -> tuple[float, np.ndarray]:
        import torch

        # An unprompted category is a pre-registration gap: record it (configs/methods/saa_prompts.yaml,
        # configs/methods/saa.yaml's prompt_coverage) AND still score it, on the repo's own fallback
        # prompt for an unknown object — never a silent KeyError, and never an invented prompt. Numbers
        # produced this way are marked on every table as "SAA+ without its per-object prompts" (protocol
        # §3 v0.2.9), never reported as SAA+ proper.
        prompt = self._prompts.get(category, self._fallback_prompt)
        with torch.no_grad():
            # FILL from A.4: run the cascade and take the repo's OWN final score and anomaly map.
            image_score, anomaly_map = _run_cascade(self._model, image, prompt)
        score = float(np.asarray(image_score).reshape(-1)[0])
        amap = np.asarray(anomaly_map)
        amap = amap.reshape(amap.shape[-2], amap.shape[-1]).astype(np.float32)
        return score, amap
```

Replace the four `FILL` helpers (`_build_cascade`, `_load_repo_prompts`, `_load_repo_fallback_prompt`,
`_run_cascade`) with the exact calls from the repo. Keep them small and named. **Do not guess these — follow the repo's inference
script.**

- [ ] **B.2 — Smoke test the backend**

```python
import numpy as np, torch
from vlmab.methods.saa_backend import SaaBackend

b = SaaBackend("Segment-Any-Anomaly", "<grounding_dino.pth>", "<sam.pth>")   # A.2 names
rng = np.random.default_rng(0)
img = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
s, m = b.score(img, "vial")
assert isinstance(s, float) and np.isfinite(s)
assert m.ndim == 2 and np.isfinite(m).all()
print("score:", round(s, 4), " map:", m.shape)
print("peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))
```

Assert only type/shape/finiteness on a random image. Record the output **and the peak VRAM** — it is
the number that decides whether the full grid is feasible on a T4 at all.

- [ ] **B.3 — Record the map granularity (the §5.2 confound)**

```python
from vlmab.methods.postprocess import distinct_levels

print("distinct levels in the SAA+ map:", distinct_levels(m), "of", m.size, "pixels")
```

Record this. A count in the single or low double digits is the expected, reportable outcome — it is the
evidence behind the paper's §5.2 limitation, not a bug. Re-run it on a few real Vial images in Phase C
and record the range.

- [ ] **B.4 — Commit the backend**

```bash
git add src/vlmab/methods/saa_backend.py
git commit -m "feat: SAA+ official-repo cascade backend bridging the score seam (GPU-only)"
```

Confirm `python -c "import vlmab.methods.saa_backend"` works **without** the repo on `sys.path` (lazy
imports), so CI stays green.

## Phase C — Vial end to end, and the cost probe

- [ ] **C.1 — Verify Vial** (`docs/datasets-access.md`):

```bash
python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
```

- [ ] **C.2 — The cost probe: measure before committing sessions**

Run this **before** the full category. It replaces the protocol's untested "may not fit 12h"
assumption with three recorded numbers.

```python
import time, numpy as np, torch, json
from vlmab.datasets.mvtec_ad2 import MVTecAD2

ds = MVTecAD2("data/mvtec_ad2")
all_vial = list(ds.samples("test_public", category="vial"))
probe_samples = all_vial[:10]

torch.cuda.reset_peak_memory_stats()
times = []
for s in probe_samples:
    img = ds.load_image(s)          # Sample carries image_path; the dataset loads and RGB-converts
    t0 = time.perf_counter()
    b.score(img, "vial")            # time the cascade only, not disk I/O
    times.append(time.perf_counter() - t0)

per_image = float(np.median(times))
n_vial = len(all_vial)
GRID_IMAGES = 0   # FILL: sum of test_public counts over all 8 categories, from prepare_data.py output

probe = {
    "seconds_per_image_median": round(per_image, 3),
    "seconds_per_image_p95": round(float(np.percentile(times, 95)), 3),
    "vial_category_hours": round(per_image * n_vial / 3600, 2),
    "full_grid_hours": round(per_image * GRID_IMAGES / 3600, 2),
    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    "n_probe_images": len(times),
}
print(json.dumps(probe, indent=2))
```

Write all of it to `results/saa/cost_probe.md` and commit. **This file is what any later decision to
reduce SAA+'s scope must cite** — protocol §6 requires a failure reported as a failure with its measured
cost, not an omission. Note that the runner is resumable and the store crash-safe, with the category as
the resume unit, so a `full_grid_hours` above 12 is a multi-session plan, not an abort; an abort is
justified only if the total exceeds the compute budget you are willing to spend, and that decision gets
recorded here with the number next to it.

```bash
git add results/saa/cost_probe.md && git commit -m "results: SAA+ measured cost probe (per image, per category, full grid)"
```

- [ ] **C.3 — Run SAA+ over Vial's public test split**

```python
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.saa import SaaRef
from vlmab.methods.saa_backend import SaaBackend

method = SaaRef(backend=b)                     # the backend built in B.1
store = ResultStore("results/saa/vial/shards")
meta = run_meta({"method": "saa", "split": "test_public", "prompts": "repo_verbatim"}, seed=0)
run_evaluation(MVTecAD2("data/mvtec_ad2"), method, store, meta,
               categories=["vial"], split="test_public", maps_dir="results/saa/vial/maps")
```

Training-free, so the runner never calls `fit`. Aggregate per lighting condition and record:

```python
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore("results/saa/vial/shards").load_all()
print(aggregate(df, by="meta_lighting")[
    ["meta_lighting", "i_auroc", "au_pro_030", "au_pro_005", "n"]].to_string(index=False))
```

Expect above-chance I-AUROC on `regular`. If AU-PRO looks anomalously low next to I-AUROC, check B.3's
granularity figure before treating it as a localisation result — that is exactly the confound the §3
map-provenance rule exists to surface.

## Phase D — VisA reproduction gate (protocol §2)

- [ ] **D.1 — Establish which metric the gate runs on**

Read SAA+'s paper and record, before scoring anything:

- Does it publish per-object VisA **image-AUROC**? → the gate runs on I-AUROC.
- If not, does it publish VisA **I-AP**? → the gate runs on I-AP.
- If not, does it publish VisA **I-F1max**? → the gate runs on I-F1max.
- If it publishes no image-level VisA metric at all → SAA+ is reported as **ungated** in every table
  (protocol §2). Do not substitute a pixel metric and do not drop the gate silently.

Record the decision, the exact table/page it came from, and the per-object published values.

- [ ] **D.2 — Get VisA** (CC BY 4.0, no account):

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar . && tar -xf VisA_20220922.tar
```

- [ ] **D.3 — Score each VisA object and run the gate**

```python
import numpy as np, pandas as pd
from PIL import Image
from vlmab.metrics.image_level import i_auroc, i_ap, i_f1max

# Set from D.1 — one of these three, recorded explicitly.
GATE_METRIC_NAME = "i_auroc"
GATE_METRIC = {"i_auroc": i_auroc, "i_ap": i_ap, "i_f1max": i_f1max}[GATE_METRIC_NAME]
PUBLISHED_SAA_VISA = {...}   # per-object published values from D.1, on GATE_METRIC_NAME

def load(paths):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

rows = []
for obj, test_paths, test_labels in visa_objects():   # your loader over VisA's split CSV
    scores = np.array([b.score(img, obj)[0] for img in load(test_paths)], dtype=np.float64)
    rows.append({"object": obj,
                 GATE_METRIC_NAME: GATE_METRIC(np.array(test_labels), scores),
                 "n": len(test_labels)})
res = pd.DataFrame(rows)
res["published"] = res["object"].map(PUBLISHED_SAA_VISA)
res["delta"] = res[GATE_METRIC_NAME] - res["published"]
print("gate metric:", GATE_METRIC_NAME)
print(res.to_string(index=False)); print("max |delta|:", res["delta"].abs().max())
```

Note that `b.score(img, obj)` needs a prompt for each VisA object — fill `visa_categories` in
`configs/methods/saa_prompts.yaml` from the repo first (A.3), verbatim.

Gate: **every object within ±1.0, `res["delta"].abs().max() <= 1.0`.** The tolerance does not change
with the metric.

- [ ] **D.4 — Record the reproduction and pin provenance**

Write `results/reproduction/saa_visa.md`: the table, **which metric the gate ran on and why** (D.1), the
repo commit, both checkpoint sha256s, the map-granularity range from B.3/C.3, and pass/fail vs ±1.0.
Confirm `configs/methods/saa.yaml` and `configs/methods/saa_prompts.yaml` have no `unrecorded` or
`unpinned_until_first_colab_run` values left:

```bash
grep -n "unrecorded\|unpinned_until_first_colab_run" configs/methods/saa.yaml configs/methods/saa_prompts.yaml
```

Expected: no output.

```bash
git add results/reproduction/saa_visa.md configs/methods/saa.yaml configs/methods/saa_prompts.yaml notebooks/saa_colab.ipynb
git commit -m "results: SAA+ VisA reproduction (§2 gate); pin repo commit + both checkpoint shas"
```

**If any object misses ±1.0:** do not report MVTec AD 2 numbers. Likely causes, in order — the wrong SAM
variant (ViT-H vs ViT-L vs ViT-B; A.2), a prompt not matching the repo's for that object (re-check A.3
verbatim, including the property constraints, which are easy to drop), taking a different variable than
the repo's own final map (re-check A.4), the repo commit differing from the paper's, or comparing
against a published number that is a pixel metric rather than the image-level one D.1 selected. Debug
against the repo's own inference output on VisA; if it genuinely cannot be reproduced, flag SAA+ as such
in every table (protocol §2).

---

## What this plan does NOT do, and why

- **The full MVTec AD 2 grid** (M3) — begins only once Phase D passes, and its session plan comes from
  C.2's measured figures, not from an estimate.
- **The evaluation server** (M4) — behind the pending registration (`docs/datasets-access.md`).
- **A generic-prompt SAA+ variant.** Considered and rejected in the spec: it would measure a method
  SAA+'s own paper does not report, and doubling the study's slowest method's compute to report both
  was not judged worth it. If §5.1's prompt ablation later needs it, that is a separate decision with
  C.2's cost figures in hand.
- **Any auxiliary-training overlap audit.** SAA+ is training-free; it appears in the §3.1 table as
  such. AnomalyCLIP and AdaCLIP have their own audit docs.
