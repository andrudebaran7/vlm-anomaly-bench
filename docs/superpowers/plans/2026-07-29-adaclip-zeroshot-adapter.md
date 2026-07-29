# AdaCLIP Zero-Shot Adapter Implementation Plan

> **Split execution.** Tasks 1–3 are CPU-verifiable and run by subagents (TDD / docs). The Colab
> phases (A–D) need the official AdaCLIP repo, CLIP weights and a GPU — none in the dev environment —
> so they run **interactively in a Colab GPU session**. Like AnomalyCLIP and unlike the anomalib
> playbooks, AdaCLIP has no packaged API: its backend is derived by reading the official repo's
> `test.py` at a pinned commit, so the Colab phases give the integration recipe with VERIFY steps, not
> verified packaged code. Fabricating an unpackaged repo's internals is exactly what the protocol
> forbids. Steps use checkbox (`- [ ]`) syntax.
>
> For the CPU tasks only: REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the AdaCLIP zero-shot adapter over an injectable backend (CPU-testable), pre-register the auxiliary-training overlap audit that makes its numbers trustworthy (protocol §3.1) including the single-checkpoint contingency, and — on Colab — wrap the official repo and reproduce AdaCLIP's published zero-shot VisA image-AUROC within ±1.0 before any MVTec AD 2 number.

**Architecture:** AdaCLIP adapts CLIP with hybrid learnable prompts — a static learned component fixed in the checkpoint plus a per-image dynamic component. It is zero-shot at inference (no fit) and *auxiliary-trained*, so which checkpoint is used for which test set is a pre-registration decision audited in a committed doc. The backend seam **takes a category** — `score(image, category) -> (raw_score, raw_map)` — as the superset choice: whether AdaCLIP's inference path consumes a class name cannot be verified without the repo, and an inert argument costs nothing while a missing one would force an adapter rewrite mid-GPU-session. anomalib does not ship AdaCLIP, so provenance is the official repo at a pinned commit (protocol §3 priority 1). The real inference runs on Colab behind the injected backend; the adapter is fully CPU-tested with a fake.

**Tech Stack:** Python 3.10+, numpy, pytest (CPU tasks); Colab T4 GPU, the official `caoyunkang/AdaCLIP` repo at a pinned commit, torch, open-clip (Colab phases).

**Spec:** `docs/superpowers/specs/2026-07-29-adaclip-saa-design.md`

## The auxiliary-training decision that governs this plan

AdaCLIP is auxiliary-trained: its static prompts are learned on an anomaly-detection dataset, then it
scores target images zero-shot. Its numbers are valid only if that auxiliary data does not overlap the
test set (protocol §1, §3.1). The checkpoint choice is **pre-registered here, identical to
AnomalyCLIP's** — both aux-trained methods following the same rule is what makes the paper's §3.1
table legible:

| Test set | Checkpoint | Why it is clean |
|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | Conservative: avoids domain proximity with the MVTec family. |
| VisA (§2 gate) | MVTec-AD-trained | Clean: different datasets. |

**Contingency, pre-registered (this is not verified — the repo may publish only one checkpoint):**

> If only the MVTec-AD-trained checkpoint is published, use it for MVTec AD 2 **and** record the
> domain proximity explicitly in `docs/adaclip-overlap-audit.md` and as a footnote in every table where
> AdaCLIP appears. Do not use it as though it were clean, and do not drop the method silently.

## Global Constraints

- CPU tasks: every test passes with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`. No test and no module imported at load time may require torch, the AdaCLIP repo, transformers, open_clip or any YAML library. The repo/CLIP are reached only through the injected backend.
- **Test command for every CPU step:** `.venv/bin/python -m pytest`. If `.venv` does not exist, create it once with:
  `uv venv --python 3.11 .venv && VIRTUAL_ENV=.venv uv pip install numpy scipy scikit-learn pandas pyarrow pillow pytest && VIRTUAL_ENV=.venv uv pip install -e . --no-deps`
- Baseline before starting: **223 tests pass** (2026-07-29). The suite must stay green after every task.
- AdaCLIP is **zero-shot** at inference (`zero_shot = True`, no fit). It is **auxiliary-trained**; the checkpoint-per-test-set mapping above is fixed and audited (protocol §3.1).
- **Provenance is the official repo at a pinned commit** (protocol §3 priority 1 — anomalib does not ship it). Record the commit and the checkpoint identities in `configs/methods/adaclip.yaml`.
- Maps and scores are RAW (protocol v0.2.6); `AdaClipRef` upsamples the map to native resolution (protocol §4). Never per-image normalise.
- **The §2 VisA reproduction gate is a hard acceptance criterion**: no MVTec AD 2 number until AdaCLIP reproduces its published zero-shot VisA image-AUROC within ±1.0 (with the MVTec-AD-trained checkpoint) through this adapter and this repo's metrics.

## Files

| File | Responsibility |
|---|---|
| `src/vlmab/methods/adaclip.py` | `AdaClipRef` adapter over an injectable backend (CPU) |
| `configs/methods/adaclip.yaml` | pre-registered config incl. the checkpoints (CPU) |
| `src/vlmab/methods/registry.py` | register `adaclip` (CPU) |
| `tests/test_adaclip.py` | adapter tests against a fake backend (CPU) |
| `tests/test_registry.py` | registry test + re-pointed unknown-method example (CPU) |
| `docs/adaclip-overlap-audit.md` | the §3.1 auxiliary-training overlap audit (CPU) |
| `docs/protocol.md` | §3 amendment + Changelog v0.2.8 (CPU) |
| `src/vlmab/methods/adaclip_backend.py` | the official-repo backend, lazy import (Colab) |
| `results/reproduction/adaclip_visa.md` | the recorded VisA reproduction table (Colab) |

---

## Task 1: AdaClipRef adapter over an injectable backend

**Files:**
- Modify: `src/vlmab/methods/adaclip.py` (replaces the stub `Adaclip` class entirely)
- Modify: `configs/methods/adaclip.yaml` (replaces the two-line stub entirely)
- Create: `tests/test_adaclip.py`

**Interfaces:**
- Consumes: `AnomalyMethod`, `Prediction`, `MethodNotRunnable` (`vlmab.methods.base`); `upsample_to` (`vlmab.methods.postprocess`); `assert_valid_prediction` (`tests/method_contract.py`).
- Produces:
  - A backend protocol: an object with `score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]`.
  - `AdaClipRef(backend=None)` with `name = "adaclip"`, `zero_shot = True`.

- [ ] **Step 1: Pre-register the config**

Replace `configs/methods/adaclip.yaml` entirely with:

```yaml
name: adaclip
zero_shot: true
aux_trained: true                # learns hybrid (static + dynamic) prompts on auxiliary AD data
# AdaCLIP (Cao et al., ECCV 2024). anomalib does not ship it, so provenance is the official repo
# (protocol §3 priority 1). Pin the commit on the first Colab run.
source: https://github.com/caoyunkang/AdaCLIP
commit: unpinned_until_first_colab_run     # record the exact SHA here
backbone: verify_at_integration            # read from the repo's config at the pinned commit
# Checkpoint per test set, so the auxiliary data never overlaps the test set
# (docs/adaclip-overlap-audit.md):
checkpoint_for_mvtec_ad2: visa_trained     # conservative: AD2 is MVTec-family; train on VisA
checkpoint_for_visa_reproduction: mvtec_ad_trained   # clean: train on MVTec AD, test on VisA
# If the repo publishes only the MVTec-AD-trained checkpoint, use it for MVTec AD 2 and set
# domain_proximity_caveat: true — never report it as clean (see the audit).
domain_proximity_caveat: false
prompts: learned_hybrid_static_plus_dynamic   # protocol §3: from the official checkpoint, not tuned
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_adaclip.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.base import MethodNotRunnable


class _FakeBackend:
    """Stands in for the real AdaCLIP backend. The seam takes a category (superset choice: AdaCLIP
    may or may not consume it — verified on Colab), and returns a raw score plus a coarse map like
    the model's patch anomaly map before upsampling."""

    def __init__(self):
        self.calls = []

    def score(self, image, category):
        self.calls.append(category)
        m = np.zeros((16, 16), dtype=np.float32)
        m[4, 7] = 2.5
        return 2.5, m


def test_is_zero_shot():
    assert AdaClipRef().zero_shot is True


def test_name_is_the_registry_name():
    assert AdaClipRef().name == "adaclip"


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        AdaClipRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    """The seam takes a category, so a category-consuming backend is possible without a rewrite."""
    backend = _FakeBackend()
    m = AdaClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    assert backend.calls == ["vial"]


def test_predict_returns_a_native_resolution_raw_map():
    m = AdaClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(2.5)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(2.5)


def test_predict_records_the_category_in_extras():
    m = AdaClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "can")
    assert pred.extras == {"category": "can"}


def test_predict_without_a_backend_is_not_runnable():
    m = AdaClipRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_adaclip.py -q`
Expected: FAIL — `ImportError: cannot import name 'AdaClipRef' from 'vlmab.methods.adaclip'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/adaclip.py` entirely with:

```python
"""AdaCLIP zero-shot adapter (Cao et al., ECCV 2024) — hybrid learnable prompts.

AdaCLIP adapts CLIP with hybrid prompts: a static component learned on auxiliary anomaly data and
fixed in the checkpoint, plus a dynamic component generated per test image. It scores any image
zero-shot. It is auxiliary-trained, so which checkpoint (trained on which dataset) is used for which
test set is a pre-registration decision audited in docs/adaclip-overlap-audit.md — the auxiliary data
must not overlap the test set.

The backend seam takes a category:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

This is the superset choice. Whether AdaCLIP's inference path actually consumes a class name is
verified on Colab against the official repo; if it does not, the backend ignores the argument and
says so in its own docstring. An inert argument costs nothing (AnomalyMethod.predict() receives
`category` unconditionally), while a missing one would force an adapter rewrite mid-GPU-session.

anomalib does not ship AdaCLIP, so provenance is the official repo at a pinned commit (protocol §3
priority 1). The real inference runs on Colab behind the injected backend; everything here is tested
with a fake backend.

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class AdaClipRef(AnomalyMethod):
    name = "adaclip"
    zero_shot = True

    def __init__(self, backend=None):
        self._backend = backend

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real backend is
        Colab-only (it needs the official repo, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the official repo + CLIP + GPU
            raise MethodNotRunnable(
                "AdaClipRef needs an AdaCLIP backend; build it in a GPU session from the official "
                "repo at the pinned commit and inject it (see the AdaCLIP Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("AdaClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_adaclip.py -q`
Expected: PASS (7 passed)

- [ ] **Step 6: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (230 passed — 223 baseline + 7 new). Record the count.

```bash
git add src/vlmab/methods/adaclip.py configs/methods/adaclip.yaml tests/test_adaclip.py
git commit -m "feat: AdaCLIP zero-shot adapter over an injectable backend (category-taking seam)"
```

---

## Task 2: Register AdaCLIP

Registering `adaclip` invalidates the registry's unknown-method test, which currently uses
`"adaclip"` as its unregistered example (the AnomalyCLIP plan left it there deliberately, and a
comment in the test flags it). This task swaps it for `"saa"` — still unregistered. **The SAA+ plan
will have to move it again when it registers `saa`**; that plan says so too.

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: `AdaClipRef` (Task 1).
- Produces: `build_method("adaclip")` returns an `AdaClipRef`; `available()` includes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_builds_adaclip():
    from vlmab.methods.adaclip import AdaClipRef

    assert isinstance(build_method("adaclip"), AdaClipRef)
    assert "adaclip" in available()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_registry.py -k adaclip -q`
Expected: FAIL — `KeyError: "unknown method 'adaclip'; available: [...]"`

- [ ] **Step 3: Register it and re-point the unknown-method test**

In `src/vlmab/methods/registry.py`, add the import in alphabetical position among the method imports
(immediately after the `anomalyclip` import):

```python
from vlmab.methods.adaclip import AdaClipRef
```

and add the `_REGISTRY` entry, keeping the dict alphabetically ordered (so `"adaclip"` goes first):

```python
    "adaclip": AdaClipRef,          # CPU-usable only with an injected backend; prepare() gates the rest
```

Then, in `tests/test_registry.py`, `test_unknown_method_lists_the_known_ones` currently calls
`build_method("adaclip")` as its unregistered example — now registered. Change that one call to the
one deferred wrapper still unregistered:

```python
        build_method("saa")
```

Keep the existing comment above it, which already tells the next person to re-swap the name; update
its parenthetical so it no longer offers `"adaclip"` as an option:

```python
        # a deferred wrapper, not registered yet — swap for another unregistered name
        # when saa itself gets registered, or this test starts failing for the wrong reason.
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_registry.py -q`
Expected: PASS — both `test_builds_adaclip` and the re-pointed `test_unknown_method_lists_the_known_ones`.

- [ ] **Step 5: Run the full suite and commit**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (231 passed). Record the count.

```bash
git add src/vlmab/methods/registry.py tests/test_registry.py
git commit -m "feat: register AdaCLIP; re-point the unknown-method example to saa"
```

---

## Task 3: The auxiliary-training overlap audit (protocol §3.1)

This is the pre-registration deliverable that makes AdaCLIP's numbers trustworthy — the same
"credibility hole" the protocol names for AnomalyCLIP. Documentation, committed to the repo.

**Files:**
- Create: `docs/adaclip-overlap-audit.md`
- Modify: `docs/protocol.md` (§3 + Changelog)

**Interfaces:**
- Consumes: the checkpoint-per-test-set decision and contingency (this plan's header).
- Produces: no code symbols.

- [ ] **Step 1: Write the overlap audit**

Create `docs/adaclip-overlap-audit.md`:

```markdown
# AdaCLIP auxiliary-training overlap audit (protocol §3.1)

AdaCLIP is auxiliary-trained: its static prompt component is learned on an anomaly-detection dataset
and fixed in the checkpoint, and a dynamic component is generated per test image. Its numbers are only
valid if the auxiliary training data does not overlap the test set — a common credibility hole in this
literature, which this table closes. It is the companion to docs/anomalyclip-overlap-audit.md, and
follows the same rule deliberately, so the paper's §3.1 table reads consistently across both
auxiliary-trained methods.

## Which checkpoint we use for which test set, and why it is clean

| Our test set | Checkpoint used | Auxiliary data | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | VisA | None — VisA is a different dataset and provider; MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. |
| VisA (reproduction §2) | MVTec-AD-trained | MVTec AD (classic) | None — MVTec AD classic and VisA are different datasets. |

## The subtlety we deliberately avoid

MVTec AD 2 is a *different dataset* from MVTec AD classic (new images, mostly new object categories),
so even the MVTec-AD-trained checkpoint would not overlap it at the image level. But MVTec AD 2 shares
MVTec AD classic's provider and industrial-inspection domain, which is a softer, domain-proximity form
of leakage. We use the **VisA-trained** checkpoint for MVTec AD 2 precisely to avoid it. This is the
conservative choice; it removes the domain-proximity question rather than arguing it away.

## Contingency: what if only one checkpoint is published?

The table above assumes AdaCLIP publishes both a VisA-trained and an MVTec-AD-trained checkpoint. That
is **unverified at the time of writing** and is checked in Colab phase A.2.

Pre-registered rule, written before any AdaCLIP number exists:

> If only the MVTec-AD-trained checkpoint is published, we use it for MVTec AD 2 anyway, **and** record
> the domain proximity explicitly: `domain_proximity_caveat: true` in `configs/methods/adaclip.yaml`,
> a filled-in row below, and a footnote on every table in the paper where AdaCLIP appears. We do not
> report it as clean, and we do not drop the method silently.
>
> If, conversely, only a VisA-trained checkpoint is published, the §2 VisA reproduction gate cannot be
> run cleanly (VisA-on-VisA). In that case the gate runs against AdaCLIP's published **MVTec AD
> (classic)** numbers with the VisA-trained checkpoint instead, and results/reproduction/adaclip_visa.md
> is renamed to record which dataset the gate actually ran on.

## Recorded on first Colab run

- Pinned repo commit: ____
- Checkpoints published by the repo (list every one found): ____
- VisA-trained checkpoint file + sha256: ____
- MVTec-AD-trained checkpoint file + sha256: ____
- Contingency triggered? (none / MVTec-only / VisA-only): ____
- If triggered, the caveat text used in the paper's tables: ____
```

- [ ] **Step 2: Amend the protocol §3**

Append to the `## 3. Methods & implementations` section of `docs/protocol.md`, immediately after the
existing AnomalyCLIP bullet:

```markdown
- **AdaCLIP (auxiliary-trained).** AdaCLIP adapts CLIP with hybrid prompts — a static component learned
  on auxiliary AD data and fixed in the checkpoint, plus a per-image dynamic component — and scores
  zero-shot; the auxiliary data must not overlap the test set (§3.1 credibility requirement).
  Provenance is the official repo at a pinned commit (anomalib does not ship it). The checkpoint is
  chosen per test set on the same rule as AnomalyCLIP, so the auxiliary data is clean: the VisA-trained
  checkpoint for MVTec AD 2 (conservative, avoiding MVTec-family domain proximity), the
  MVTec-AD-trained checkpoint for the VisA reproduction. If the repo publishes only one checkpoint, the
  pre-registered contingency in docs/adaclip-overlap-audit.md applies: the available checkpoint is used
  and the resulting domain proximity is recorded as a caveat on every table, never reported as clean.
  Full audit: docs/adaclip-overlap-audit.md.
```

- [ ] **Step 3: Add the Changelog entry**

Append to the Changelog at the bottom of `docs/protocol.md`:

```markdown
- 2026-07-29 — v0.2.8. Records AdaCLIP's auxiliary-training regime and its checkpoint-per-test-set
  overlap audit (§3, docs/adaclip-overlap-audit.md), on the same rule already fixed for AnomalyCLIP:
  VisA-trained checkpoint for MVTec AD 2, MVTec-AD-trained for the VisA reproduction. Also pre-registers
  the single-checkpoint contingency — if the repo publishes only one, it is used and the domain
  proximity is carried as a caveat on every table rather than reported as clean. Decided before any
  AdaCLIP number exists. No evaluation rule for other methods changed.
```

- [ ] **Step 4: Verify and commit**

Run: `grep -c "overlap" docs/adaclip-overlap-audit.md docs/protocol.md`
Expected: at least 1 in each.

Run: `grep -n "v0.2.8" docs/protocol.md`
Expected: one hit, the new Changelog entry.

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (231 passed — no code changed).

```bash
git add docs/adaclip-overlap-audit.md docs/protocol.md
git commit -m "docs: AdaCLIP auxiliary-training overlap audit; protocol v0.2.8"
```

---

# Colab phases (manual, GPU) — the official-repo backend

AdaCLIP has no packaged API; the backend is derived by reading the official repo's `test.py` at the
pinned commit. These phases give the recipe and the shape, not verified packaged code — the VERIFY
steps and the smoke test are the real proof.

## Phase A — Environment, pinned repo, checkpoints

- [ ] **A.1 — GPU, our repo, the AdaCLIP repo pinned**

```python
!nvidia-smi -L
!git clone https://github.com/andrudebaran7/vlm-anomaly-bench.git && cd vlm-anomaly-bench && pip install -e . --quiet
!git clone https://github.com/caoyunkang/AdaCLIP.git
%cd AdaCLIP && git rev-parse HEAD   # record this SHA into configs/methods/adaclip.yaml
!pip install -r requirements.txt --quiet 2>/dev/null || pip install ftfy regex open_clip_torch --quiet
%cd ..
```

Record the pinned SHA: ____ .

- [ ] **A.2 — Enumerate the published checkpoints and record their sha256**

This is where the single-checkpoint contingency is resolved. List **every** checkpoint the repo ships
or links, and identify which auxiliary dataset each was trained on (read the repo's README and its
test/eval scripts — they name the checkpoint per dataset).

```python
import hashlib, glob, os
paths = sorted(glob.glob("AdaCLIP/**/*.pth", recursive=True) + glob.glob("AdaCLIP/**/*.pt", recursive=True))
for f in paths:
    print(f, os.path.getsize(f), hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
print("total checkpoints found:", len(paths))
```

If the repo ships no weights in-tree, follow its README download link first, then re-run the cell.

**Decide and record now, before any scoring:**
- Both checkpoints published → proceed as planned, contingency not triggered.
- Only MVTec-AD-trained → set `domain_proximity_caveat: true` in `configs/methods/adaclip.yaml` and
  fill the caveat row in `docs/adaclip-overlap-audit.md`.
- Only VisA-trained → the §2 gate runs against AdaCLIP's published **MVTec AD (classic)** numbers
  instead (see the audit's contingency), and Phase D's table is relabelled accordingly.

Write the outcome into `docs/adaclip-overlap-audit.md`'s "Recorded on first Colab run".

- [ ] **A.3 — VERIFY the inference path**

Read `AdaCLIP/test.py` (and whatever eval shell script the README points at). Identify, from the actual
code at this commit:
- how the model is built and a checkpoint loaded (the class, the load call);
- the image preprocessing (CLIP transform and input resolution);
- how a forward produces the image-level anomaly score and the pixel anomaly map;
- **whether the inference path takes a class/category name at all** — this is what settles the seam
  question. Record the answer either way; it goes in the backend's docstring.

Write down those exact calls. Phase B's backend is a thin wrapper around them and they cannot be taken
from memory.

## Phase B — The backend

- [ ] **B.1 — Write `src/vlmab/methods/adaclip_backend.py`**

The skeleton below is the *shape*; fill the three repo-specific calls (marked FILL) from what you read
in A.3. The repo import path assumes `AdaCLIP/` is on `sys.path`.

```python
"""AdaCLIP official-repo backend for AdaClipRef (GPU/Colab only).

Wraps the official repo's zero-shot inference (test.py) behind the seam
score(image, category) -> (raw_score, raw_map). The model and one checkpoint are loaded once at
construction; each image is preprocessed with the repo's CLIP transform and forwarded to the RAW image
score and anomaly map (protocol v0.2.6 — no per-image normalisation; AdaClipRef upsamples the map to
native resolution).

Whether `category` reaches the model is recorded in `consumes_category` below, set from what A.3 found
in the repo: AdaCLIP's prompts may be fully object-agnostic, in which case the argument is accepted for
seam compatibility and deliberately unused.

The repo and torch are imported lazily so this module imports in CI (never constructed there).
"""
import sys
import numpy as np


class AdaClipBackend:
    # Set from Colab A.3 against the pinned commit. Documented, not guessed.
    consumes_category = False   # FILL: True if the repo's inference path takes a class name

    def __init__(self, repo_dir: str, checkpoint_path: str, device: str = "cuda"):
        import torch

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # FILL from A.3: build the AdaCLIP model and load `checkpoint_path`, reproducing test.py's
        # construction exactly (backbone, prompt-learner state dict, any config object it passes).
        self._model = _build_and_load(checkpoint_path, device)   # FILL
        self._transform = _build_preprocess()                    # FILL (the repo's CLIP transform)
        self._device = device
        self._model.eval()

    def score(self, image: np.ndarray, category: str) -> tuple[float, np.ndarray]:
        import torch
        from PIL import Image

        tensor = self._transform(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            # FILL from test.py. Pass `category` only if A.3 found the path consumes it; if not,
            # leave it out and keep consumes_category = False.
            image_score, anomaly_map = _forward(self._model, tensor)
        score = float(np.asarray(image_score).reshape(-1)[0])
        amap = np.asarray(anomaly_map)
        amap = amap.reshape(amap.shape[-2], amap.shape[-1]).astype(np.float32)
        return score, amap
```

Replace the three `FILL` helpers (`_build_and_load`, `_build_preprocess`, `_forward`) with the exact
calls from the repo's `test.py`. Keep them small and named, so a reader sees precisely what repo code is
being reused. **Do not guess these — if `test.py` differs from the shape above, follow `test.py`.**

- [ ] **B.2 — Smoke test the backend (with the VisA-trained checkpoint)**

```python
import numpy as np
from vlmab.methods.adaclip_backend import AdaClipBackend

b = AdaClipBackend("AdaCLIP", "AdaCLIP/checkpoints/<visa_trained>.pth")   # A.2 name
rng = np.random.default_rng(0)
img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
s, m = b.score(img, "vial")
assert isinstance(s, float) and np.isfinite(s)
assert m.ndim == 2 and np.isfinite(m).all()
print("score:", round(s, 4), " map:", m.shape, " consumes_category:", b.consumes_category)
```

Assert only type/shape/finiteness on a random image (a meaningless image has no ground truth). Real
separation is proven in Phase C/D. Record the output.

- [ ] **B.3 — Commit the backend**

```bash
git add src/vlmab/methods/adaclip_backend.py
git commit -m "feat: AdaCLIP official-repo backend bridging the score seam (GPU-only)"
```

Confirm `python -c "import vlmab.methods.adaclip_backend"` works **without** the repo on `sys.path`
(lazy imports), so CI stays green.

## Phase C — End to end on one Vial category (VisA-trained checkpoint, per §3.1)

- [ ] **C.1 — Verify Vial** (`docs/datasets-access.md`):

```bash
python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
```

- [ ] **C.2 — Run AdaCLIP over Vial's public test split**

```python
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.adaclip_backend import AdaClipBackend

# MVTec AD 2 uses the VisA-trained checkpoint (overlap audit, §3.1).
backend = AdaClipBackend("AdaCLIP", "AdaCLIP/checkpoints/<visa_trained>.pth")
method = AdaClipRef(backend=backend)
store = ResultStore("results/adaclip/vial/shards")
meta = run_meta({"method": "adaclip", "split": "test_public", "checkpoint": "visa_trained"}, seed=0)
run_evaluation(MVTecAD2("data/mvtec_ad2"), method, store, meta,
               categories=["vial"], split="test_public", maps_dir="results/adaclip/vial/maps")
```

Zero-shot, so the runner never calls `fit`. Aggregate per lighting condition and record:

```python
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore("results/adaclip/vial/shards").load_all()
print(aggregate(df, by="meta_lighting")[
    ["meta_lighting", "i_auroc", "au_pro_030", "au_pro_005", "n"]].to_string(index=False))
```

Expect above-chance I-AUROC on `regular`; the lighting-shift drop is the study's story.

## Phase D — VisA reproduction gate (MVTec-AD-trained checkpoint, protocol §2)

- [ ] **D.1 — Get VisA** (CC BY 4.0, no account):

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar . && tar -xf VisA_20220922.tar
```

- [ ] **D.2 — Score each VisA object and compute I-AUROC (MVTec-AD-trained checkpoint)**

```python
import numpy as np, pandas as pd
from PIL import Image
from vlmab.methods.adaclip_backend import AdaClipBackend
from vlmab.metrics.image_level import i_auroc

# VisA reproduction uses the MVTec-AD-trained checkpoint (clean; §3.1).
b = AdaClipBackend("AdaCLIP", "AdaCLIP/checkpoints/<mvtec_ad_trained>.pth")
def load(paths):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

rows = []
for obj, test_paths, test_labels in visa_objects():   # your loader over VisA's split CSV
    imgs = load(test_paths)
    scores = np.array([b.score(img, obj)[0] for img in imgs], dtype=np.float64)
    rows.append({"object": obj, "i_auroc": i_auroc(np.array(test_labels), scores), "n": len(test_labels)})
res = pd.DataFrame(rows)
res["published"] = res["object"].map(PUBLISHED_ADACLIP_VISA_IAUROC)  # AdaCLIP paper's VisA table
res["delta"] = res["i_auroc"] - res["published"]
print(res.to_string(index=False)); print("max |delta|:", res["delta"].abs().max())
```

`PUBLISHED_ADACLIP_VISA_IAUROC` is the per-object zero-shot VisA image-AUROC from the AdaCLIP paper
(record the exact source next to the table — protocol §3). Gate: **every object within ±1.0,
`res["delta"].abs().max() <= 1.0`.**

If A.2 triggered the VisA-only contingency, this phase instead scores AdaCLIP's published **MVTec AD
(classic)** table with the VisA-trained checkpoint; keep everything else identical and relabel the
output file.

- [ ] **D.3 — Record the reproduction and pin provenance**

Write `results/reproduction/adaclip_visa.md` (table, repo commit, checkpoint shas, which checkpoint and
which dataset the gate ran on, pass/fail vs ±1.0). Fill the `commit`, `backbone` and checkpoint fields
in `configs/methods/adaclip.yaml` and the "Recorded on first Colab run" fields in
`docs/adaclip-overlap-audit.md`. Commit:

```bash
git add results/reproduction/adaclip_visa.md configs/methods/adaclip.yaml docs/adaclip-overlap-audit.md notebooks/adaclip_colab.ipynb
git commit -m "results: AdaCLIP VisA reproduction (§2 gate); pin repo commit + checkpoint shas"
```

**If any object misses ±1.0:** do not report MVTec AD 2 numbers. Likely causes, in order — the wrong
checkpoint (VisA-trained vs MVTec-AD-trained mixed up), a preprocessing mismatch (AdaCLIP's exact CLIP
resolution/transform), the category argument being passed when the repo does not expect it (or omitted
when it does — re-check A.3), the repo commit differing from the paper's, or the wrong published
baseline (confirm it is zero-shot image-AUROC and not a pixel metric). Debug against the repo's own
`test.py` output on VisA; if it genuinely cannot be reproduced, flag AdaCLIP as such in every table
(protocol §2).

---

## What this plan does NOT do, and why

- **The full MVTec AD 2 grid** (M3) — begins only once Phase D passes.
- **The evaluation server** (M4) — behind the pending registration (`docs/datasets-access.md`).
- **SAA+** — its own plan (`docs/superpowers/plans/2026-07-29-saa-trainingfree-adapter.md`), executed
  after this one. It is training-free, so it needs no overlap audit, but it carries a §3 prompt
  amendment and a compute probe this plan does not.
