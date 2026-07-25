# AnomalyCLIP Zero-Shot Adapter Implementation Plan

> **Split execution.** Tasks 1–3 are CPU-verifiable and run by subagents (TDD / docs). The Colab
> phases (A–D) need the official AnomalyCLIP repo, CLIP weights and a GPU — none in the dev
> environment — so they run **interactively in a Colab GPU session**. Unlike the anomalib playbooks,
> AnomalyCLIP has no packaged API: its backend is derived by reading the official repo's `test.py`
> at a pinned commit, so the Colab phases give the integration recipe with VERIFY steps, not
> verified packaged code. Fabricating an unpackaged repo's internals is exactly what the protocol
> forbids. Steps use checkbox (`- [ ]`) syntax.
>
> For the CPU tasks only: REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the AnomalyCLIP zero-shot adapter over an injectable backend (CPU-testable), pre-register the auxiliary-training overlap audit that makes its numbers trustworthy (protocol §3.1), and — on Colab — wrap the official repo and reproduce AnomalyCLIP's published zero-shot VisA numbers within ±1.0 before any MVTec AD 2 number.

**Architecture:** AnomalyCLIP is zero-shot at inference (no fit) and uses object-AGNOSTIC learned prompts, so — unlike WinCLIP — the backend seam takes no category: `score(image) -> (raw_score, raw_map)`. It is *auxiliary-trained*: its prompts are learned on an auxiliary AD dataset that must not overlap the test set, so the choice of checkpoint per test set is a pre-registration decision, audited in a committed doc. anomalib does not ship AnomalyCLIP (verified: `list_models()` has WinClip but not AnomalyCLIP), so provenance is the official repo at a pinned commit (protocol §3 priority 1). The real inference runs on Colab behind the injected backend; the adapter is fully CPU-tested with a fake.

**Tech Stack:** Python 3.10+, numpy, pytest (CPU tasks); Colab T4 GPU, the official `zqhang/AnomalyCLIP` repo at a pinned commit, torch, open-clip (Colab phases).

## The auxiliary-training facts that govern this plan (verified from the official repo)

The official repo states, verbatim: *"We test all datasets by training once on MVTec AD. For MVTec
AD, AnomalyCLIP is trained on VisA."* So AnomalyCLIP ships two relevant checkpoints:

- **MVTec-AD-trained** — the main checkpoint; tests cleanly on VisA and every non-MVTec dataset.
- **VisA-trained** — used to test on MVTec AD, to avoid MVTec-on-MVTec overlap.

**Decisions (pre-registered here):**
- **MVTec AD 2 (primary test): the VisA-trained checkpoint.** MVTec AD 2 is a different dataset from
  MVTec AD classic (new images, mostly new objects) but the *same provider and domain*, so we use
  the VisA-trained checkpoint to avoid even domain-proximity leakage — consistent with AnomalyCLIP's
  own "for MVTec, train on VisA" rule. This is conservative and keeps the §3.1 table clean.
- **VisA reproduction (§2 gate): the MVTec-AD-trained checkpoint.** Training on MVTec AD and testing
  on VisA is clean and is the setup AnomalyCLIP publishes VisA numbers for.
- **Seam takes no category** (object-agnostic prompts): `score(image) -> (float, map)`.

## Global Constraints

- CPU tasks: every test passes with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`. No test and no module imported at load time may require torch, the AnomalyCLIP repo, transformers, open_clip or any YAML library. The repo/CLIP are reached only through the injected backend.
- AnomalyCLIP is **zero-shot** at inference (`zero_shot = True`, no fit). It is **auxiliary-trained**; the checkpoint-per-test-set mapping above is fixed and audited (protocol §3.1).
- **Provenance is the official repo at a pinned commit** (protocol §3 priority 1 — anomalib does not ship it). Record the commit and the two checkpoint identities in `configs/methods/anomalyclip.yaml`.
- Maps and scores are RAW (protocol v0.2.6); `AnomalyClipRef` upsamples the map to native resolution (protocol §4). Never per-image normalise.
- **The §2 VisA reproduction gate is a hard acceptance criterion**: no MVTec AD 2 number until AnomalyCLIP reproduces its published zero-shot VisA image-AUROC within ±1.0 (with the MVTec-AD-trained checkpoint) through this adapter and this repo's metrics.

## Files

| File | Responsibility |
|---|---|
| `src/vlmab/methods/anomalyclip.py` | `AnomalyClipRef` adapter over an injectable backend (CPU) |
| `configs/methods/anomalyclip.yaml` | pre-registered config incl. the two checkpoints (CPU) |
| `src/vlmab/methods/registry.py` | register `anomalyclip` (CPU) |
| `docs/anomalyclip-overlap-audit.md` | the §3.1 auxiliary-training overlap audit (CPU) |
| `docs/protocol.md` | §3 amendment recording the aux-training regime (CPU) |
| `src/vlmab/methods/anomalyclip_backend.py` | the official-repo backend, lazy import (Colab) |
| `results/reproduction/anomalyclip_visa.md` | the recorded VisA reproduction table (Colab) |

---

## Task 1: AnomalyClipRef adapter over an injectable backend

**Files:**
- Modify: `src/vlmab/methods/anomalyclip.py` (replaces the stub)
- Modify: `configs/methods/anomalyclip.yaml`
- Create: `tests/test_anomalyclip.py`

**Interfaces:**
- Consumes: `AnomalyMethod`, `Prediction`, `MethodNotRunnable` (base); `upsample_to` (postprocess);
  `assert_valid_prediction` (contract harness).
- Produces:
  - A backend protocol: an object with `score(image: np.ndarray) -> tuple[float, np.ndarray]`.
  - `AnomalyClipRef(backend=None)` with `name = "anomalyclip"`, `zero_shot = True`.

- [ ] **Step 1: Pre-register the config**

Replace `configs/methods/anomalyclip.yaml` entirely with:

```yaml
name: anomalyclip
zero_shot: true
aux_trained: true                # learns object-agnostic prompts on auxiliary AD data
# AnomalyCLIP (Zhou et al., ICLR 2024). anomalib does not ship it, so provenance is the official
# repo (protocol §3 priority 1). Pin the commit on the first Colab run.
source: https://github.com/zqhang/AnomalyCLIP
commit: unpinned_until_first_colab_run     # record the exact SHA here
backbone: ViT-L-14-336                     # AnomalyCLIP's CLIP backbone (verify at integration)
# Two official checkpoints, each used only where its auxiliary training data does NOT overlap the
# test set (docs/anomalyclip-overlap-audit.md):
checkpoint_for_mvtec_ad2: visa_trained     # conservative: AD2 is MVTec-family; train on VisA
checkpoint_for_visa_reproduction: mvtec_ad_trained   # clean: train on MVTec AD, test on VisA
prompts: learned_object_agnostic           # protocol §3: from the official checkpoint, not tuned
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_anomalyclip.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.base import MethodNotRunnable


class _FakeBackend:
    """Stands in for the real AnomalyCLIP backend. Object-agnostic, so score() takes no category;
    returns a raw score plus a coarse map (like the model's patch anomaly map before upsampling)."""

    def __init__(self):
        self.calls = 0

    def score(self, image):
        self.calls += 1
        m = np.zeros((16, 16), dtype=np.float32)
        m[5, 6] = 3.1
        return 3.1, m


def test_is_zero_shot():
    assert AnomalyClipRef().zero_shot is True


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        AnomalyClipRef().prepare(device="cpu")


def test_predict_calls_the_backend_with_the_image_only():
    """Object-agnostic: the backend seam takes no category (unlike WinCLIP)."""
    backend = _FakeBackend()
    m = AnomalyClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    assert backend.calls == 1


def test_predict_returns_a_native_resolution_raw_map():
    m = AnomalyClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(3.1)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(3.1)


def test_predict_without_a_backend_is_not_runnable():
    m = AnomalyClipRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
```

- [ ] **Step 3: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_anomalyclip.py -q`
Expected: FAIL — `ImportError: cannot import name 'AnomalyClipRef' from 'vlmab.methods.anomalyclip'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/anomalyclip.py` entirely with:

```python
"""AnomalyCLIP zero-shot adapter (Zhou et al., ICLR 2024) — object-agnostic learned prompts.

AnomalyCLIP learns object-AGNOSTIC text prompts on auxiliary anomaly data, then scores any image
zero-shot without per-object prompts. So, unlike WinCLIP, the backend seam takes no category:
score(image) -> (raw_score, raw_map). It is auxiliary-trained, so which checkpoint (trained on
which dataset) is used for which test set is a pre-registration decision audited in
docs/anomalyclip-overlap-audit.md — the auxiliary data must not overlap the test set.

anomalib does not ship AnomalyCLIP, so provenance is the official repo at a pinned commit
(protocol §3 priority 1). The real inference runs on Colab behind the injected backend; everything
here is tested with a fake backend.

The backend is any object with:
    score(image: np.ndarray) -> tuple[float, np.ndarray]   # (raw image score, raw anomaly map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class AnomalyClipRef(AnomalyMethod):
    name = "anomalyclip"
    zero_shot = True

    def __init__(self, backend=None):
        self._backend = backend

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real backend is
        Colab-only (it needs the official repo, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the official repo + CLIP + GPU
            raise MethodNotRunnable(
                "AnomalyClipRef needs an AnomalyCLIP backend; build it in a GPU session from the "
                "official repo at the pinned commit and inject it (see the AnomalyCLIP Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        # `category` is part of the AnomalyMethod contract but unused: AnomalyCLIP's prompts are
        # object-agnostic, so the backend scores the image without an object name.
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("AnomalyClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(image_score=float(raw_score), anomaly_map=amap)
```

- [ ] **Step 5: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_anomalyclip.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/anomalyclip.py configs/methods/anomalyclip.yaml tests/test_anomalyclip.py
git commit -m "feat: AnomalyCLIP zero-shot adapter over an injectable backend (object-agnostic seam)"
```

---

## Task 2: Register AnomalyCLIP

Registering `anomalyclip` invalidates the registry's unknown-method test, which currently uses
`"anomalyclip"` as its unregistered example (a comment there already flags this). This task swaps it.

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: `AnomalyClipRef` (Task 1).
- Produces: `build_method("anomalyclip")` returns an `AnomalyClipRef`; `available()` includes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_builds_anomalyclip():
    from vlmab.methods.anomalyclip import AnomalyClipRef

    assert isinstance(build_method("anomalyclip"), AnomalyClipRef)
    assert "anomalyclip" in available()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -k anomalyclip -q`
Expected: FAIL — `KeyError: "unknown method 'anomalyclip'..."`

- [ ] **Step 3: Register it and re-point the unknown-method test**

In `src/vlmab/methods/registry.py`, add the import next to the other method imports:

```python
from vlmab.methods.anomalyclip import AnomalyClipRef
```

and the `_REGISTRY` entry:

```python
    "anomalyclip": AnomalyClipRef,  # CPU-usable only with an injected backend; prepare() gates the rest
```

Then, in `tests/test_registry.py`, `test_unknown_method_lists_the_known_ones` currently calls
`build_method("anomalyclip")` as its unregistered example — now registered. Change that one call to
an example that is still unregistered:

```python
        build_method("adaclip")
```

(`adaclip` is confirmed absent from the registry. Keep the existing comment about re-swapping when
it too gets registered.)

- [ ] **Step 4: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -q`
Expected: PASS — both `test_builds_anomalyclip` and the re-pointed `test_unknown_method_lists_the_known_ones`.

- [ ] **Step 5: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/registry.py tests/test_registry.py
git commit -m "feat: register AnomalyCLIP; re-point the unknown-method example to adaclip"
```

---

## Task 3: The auxiliary-training overlap audit (protocol §3.1)

This is the pre-registration deliverable that makes AnomalyCLIP's numbers trustworthy, and it is the
"credibility hole" the protocol names. Documentation, committed to the repo.

**Files:**
- Create: `docs/anomalyclip-overlap-audit.md`
- Modify: `docs/protocol.md` (§3 + Changelog)

**Interfaces:**
- Consumes: the checkpoint-per-test-set decision (this plan's header).
- Produces: no code symbols.

- [ ] **Step 1: Write the overlap audit**

Create `docs/anomalyclip-overlap-audit.md`:

```markdown
# AnomalyCLIP auxiliary-training overlap audit (protocol §3.1)

AnomalyCLIP is auxiliary-trained: it learns object-agnostic prompts on an anomaly-detection dataset,
then scores target images zero-shot. Its numbers are only valid if the auxiliary training data does
not overlap the test set — a common credibility hole in this literature, which this table closes.

## The official checkpoints (from https://github.com/zqhang/AnomalyCLIP)

The repo states: "We test all datasets by training once on MVTec AD. For MVTec AD, AnomalyCLIP is
trained on VisA." So there are two relevant checkpoints:

| Checkpoint | Auxiliary training data |
|---|---|
| MVTec-AD-trained (main) | MVTec AD (classic) |
| VisA-trained | VisA |

## Which checkpoint we use for which test set, and why it is clean

| Our test set | Checkpoint used | Auxiliary data | Overlap with the test set? |
|---|---|---|---|
| MVTec AD 2 (primary) | VisA-trained | VisA | None — VisA is a different dataset and provider; and MVTec AD 2, though MVTec-family, is never touched by a VisA-trained model. |
| VisA (reproduction §2) | MVTec-AD-trained | MVTec AD (classic) | None — MVTec AD classic and VisA are different datasets. |

## The subtlety we deliberately avoid

MVTec AD 2 is a *different dataset* from MVTec AD classic (new images, mostly new object categories),
so even the MVTec-AD-trained checkpoint would not overlap it at the image level. But MVTec AD 2 shares
MVTec AD classic's provider and industrial-inspection domain, which is a softer, domain-proximity
form of leakage. We use the **VisA-trained** checkpoint for MVTec AD 2 precisely to avoid it —
consistent with AnomalyCLIP's own rule of training on VisA for any MVTec-family test. This is the
conservative choice; it removes the domain-proximity question rather than arguing it away.

## Recorded on first Colab run

- Pinned repo commit: ____
- VisA-trained checkpoint file + sha256: ____
- MVTec-AD-trained checkpoint file + sha256: ____
```

- [ ] **Step 2: Amend the protocol §3**

Append to `docs/protocol.md` §3:

```markdown
- **AnomalyCLIP (auxiliary-trained).** AnomalyCLIP learns object-agnostic prompts on an auxiliary AD
  dataset and scores zero-shot; the auxiliary data must not overlap the test set (§3.1 credibility
  requirement). Provenance is the official repo at a pinned commit (anomalib does not ship it). The
  checkpoint is chosen per test set so the auxiliary data is clean: the VisA-trained checkpoint for
  MVTec AD 2 (conservative, avoiding MVTec-family domain proximity), the MVTec-AD-trained checkpoint
  for the VisA reproduction. Full audit: docs/anomalyclip-overlap-audit.md.
```

- [ ] **Step 3: Add the Changelog entry**

Append to the Changelog in `docs/protocol.md`:

```markdown
- 2026-07-25 — v0.2.7. Records AnomalyCLIP's auxiliary-training regime and the checkpoint-per-test-
  set overlap audit (§3, docs/anomalyclip-overlap-audit.md): VisA-trained checkpoint for MVTec AD 2
  (conservative against MVTec-family domain proximity), MVTec-AD-trained for the VisA reproduction.
  Decided before any AnomalyCLIP number exists. No evaluation rule for other methods changed.
```

- [ ] **Step 4: Verify and commit**

Run: `grep -c "overlap" docs/anomalyclip-overlap-audit.md docs/protocol.md`
Expected: at least 1 in each.

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS (no code changed).

```bash
git add docs/anomalyclip-overlap-audit.md docs/protocol.md
git commit -m "docs: AnomalyCLIP auxiliary-training overlap audit; protocol v0.2.7"
```

---

# Colab phases (manual, GPU) — the official-repo backend

AnomalyCLIP has no packaged API; the backend is derived by reading the official repo's `test.py` at
the pinned commit. These phases give the recipe and the shape, not verified packaged code — the
VERIFY steps and the smoke test are the real proof.

## Phase A — Environment, pinned repo, checkpoints

- [ ] **A.1 — GPU, our repo, the AnomalyCLIP repo pinned**

```python
!nvidia-smi -L
!git clone https://github.com/andrudebaran7/vlm-anomaly-bench.git && cd vlm-anomaly-bench && pip install -e . --quiet
!git clone https://github.com/zqhang/AnomalyCLIP.git
%cd AnomalyCLIP && git rev-parse HEAD   # record this SHA into configs/methods/anomalyclip.yaml
!pip install -r requirements.txt open_clip_torch --quiet 2>/dev/null || pip install ftfy regex open_clip_torch --quiet
%cd ..
```

Record the pinned SHA: ____ .

- [ ] **A.2 — Get both checkpoints and record their sha256**

The repo ships checkpoints under `AnomalyCLIP/checkpoints/`. Confirm which file is the VisA-trained
one and which is the MVTec-AD-trained one (read the repo's `test.sh` — it names the checkpoint per
dataset). If a checkpoint is not in the repo, follow the repo README's download link.

```python
import hashlib, glob
for f in sorted(glob.glob("AnomalyCLIP/checkpoints/*")):
    print(f, hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
```

Record each file + sha256 into `docs/anomalyclip-overlap-audit.md`'s "Recorded on first Colab run".

- [ ] **A.3 — VERIFY the inference path**

Read `AnomalyCLIP/test.py` and `test.sh`. Identify, from the actual code at this commit:
- how the model is built and a checkpoint loaded (the class, the load call);
- the image preprocessing (CLIP transform / resolution — AnomalyCLIP typically uses 518px);
- how a forward produces the image-level anomaly score and the pixel anomaly map.

Write down those exact calls — Phase B's backend is a thin wrapper around them, and they cannot be
taken from memory.

## Phase B — The backend

- [ ] **B.1 — Write `src/vlmab/methods/anomalyclip_backend.py`**

The skeleton below is the *shape*; fill the three repo-specific calls (marked FILL) from what you
read in A.3. The repo import path assumes `AnomalyCLIP/` is on `sys.path`.

```python
"""AnomalyCLIP official-repo backend for AnomalyClipRef (GPU/Colab only).

Wraps the official repo's zero-shot inference (test.py) behind the seam score(image) -> (raw_score,
raw_map). The model and one checkpoint are loaded once at construction; each image is preprocessed
with the repo's CLIP transform and forwarded to the RAW image score and anomaly map (protocol
v0.2.6 — no per-image normalisation; AnomalyClipRef upsamples the map to native resolution).

The repo and torch are imported lazily so this module imports in CI (never constructed there).
"""
import sys
import numpy as np


class AnomalyClipBackend:
    def __init__(self, repo_dir: str, checkpoint_path: str, device: str = "cuda"):
        import torch

        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # FILL from A.3: build the AnomalyCLIP model and load `checkpoint_path`. In the repo this is
        # roughly: build CLIP (open_clip) at the AnomalyCLIP config, wrap it, load the prompt-learner
        # state dict from checkpoint_path. Reproduce test.py's construction exactly.
        self._model = _build_and_load(checkpoint_path, device)   # FILL
        self._transform = _build_preprocess()                    # FILL (CLIP transform, ~518px)
        self._device = device
        self._model.eval()

    def score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch
        from PIL import Image

        tensor = self._transform(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            image_score, anomaly_map = _forward(self._model, tensor)   # FILL from test.py
        score = float(np.asarray(image_score).reshape(-1)[0])
        amap = np.asarray(anomaly_map).reshape(anomaly_map.shape[-2], anomaly_map.shape[-1]).astype(np.float32)
        return score, amap
```

Replace the three `FILL` helpers (`_build_and_load`, `_build_preprocess`, `_forward`) with the exact
calls from the repo's `test.py`. Keep them small and named, so a reader sees precisely what repo code
is being reused. **Do not guess these — if `test.py` differs from the shape above, follow `test.py`.**

- [ ] **B.2 — Smoke test the backend (with the VisA-trained checkpoint)**

```python
import numpy as np
from vlmab.methods.anomalyclip_backend import AnomalyClipBackend

b = AnomalyClipBackend("AnomalyCLIP", "AnomalyCLIP/checkpoints/<visa_trained>.pth")   # A.2 name
rng = np.random.default_rng(0)
img = rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)
s, m = b.score(img)
assert isinstance(s, float) and np.isfinite(s)
assert m.ndim == 2 and np.isfinite(m).all()
print("score:", round(s, 4), " map:", m.shape)
```

Assert only type/shape/finiteness on a random image (a meaningless image has no ground truth). Real
separation is proven in Phase C/D. Record the output.

- [ ] **B.3 — Commit the backend**

```bash
git add src/vlmab/methods/anomalyclip_backend.py
git commit -m "feat: AnomalyCLIP official-repo backend bridging the score seam (GPU-only)"
```

Confirm `import vlmab.methods.anomalyclip_backend` works without the repo installed (lazy imports),
so CI stays green.

## Phase C — End to end on one Vial category (VisA-trained checkpoint, per §3.1)

- [ ] **C.1 — Verify Vial** (`docs/datasets-access.md`):

```bash
python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
```

- [ ] **C.2 — Run AnomalyCLIP over Vial's public test split**

```python
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.anomalyclip_backend import AnomalyClipBackend

# MVTec AD 2 uses the VisA-trained checkpoint (overlap audit, §3.1).
backend = AnomalyClipBackend("AnomalyCLIP", "AnomalyCLIP/checkpoints/<visa_trained>.pth")
method = AnomalyClipRef(backend=backend)
store = ResultStore("results/anomalyclip/vial/shards")
meta = run_meta({"method": "anomalyclip", "split": "test_public", "checkpoint": "visa_trained"}, seed=0)
run_evaluation(MVTecAD2("data/mvtec_ad2"), method, store, meta,
               categories=["vial"], split="test_public", maps_dir="results/anomalyclip/vial/maps")
```

Zero-shot, so the runner never calls `fit`. Aggregate per lighting condition and record:

```python
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore("results/anomalyclip/vial/shards").load_all()
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
from vlmab.methods.anomalyclip_backend import AnomalyClipBackend
from vlmab.metrics.image_level import i_auroc

# VisA reproduction uses the MVTec-AD-trained checkpoint (clean; §3.1).
b = AnomalyClipBackend("AnomalyCLIP", "AnomalyCLIP/checkpoints/<mvtec_ad_trained>.pth")
def load(paths):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

rows = []
for obj, test_paths, test_labels in visa_objects():   # your loader over VisA's split CSV
    scores = np.array([b.score(img)[0] for img in load(test_paths)], dtype=np.float64)
    rows.append({"object": obj, "i_auroc": i_auroc(np.array(test_labels), scores), "n": len(test_labels)})
res = pd.DataFrame(rows)
res["published"] = res["object"].map(PUBLISHED_ANOMALYCLIP_VISA_IAUROC)  # AnomalyCLIP paper's VisA table
res["delta"] = res["i_auroc"] - res["published"]
print(res.to_string(index=False)); print("max |delta|:", res["delta"].abs().max())
```

`PUBLISHED_ANOMALYCLIP_VISA_IAUROC` is the per-object zero-shot VisA image-AUROC from the AnomalyCLIP
paper (record the source). Gate: **every object within ±1.0, `res["delta"].abs().max() <= 1.0`.**

- [ ] **D.3 — Record the reproduction and pin provenance**

Write `results/reproduction/anomalyclip_visa.md` (table, repo commit, checkpoint shas, pass/fail vs
±1.0). Fill the `commit` and checkpoint fields in `configs/methods/anomalyclip.yaml` and the "Recorded
on first Colab run" fields in `docs/anomalyclip-overlap-audit.md`. Commit:

```bash
git add results/reproduction/anomalyclip_visa.md configs/methods/anomalyclip.yaml docs/anomalyclip-overlap-audit.md notebooks/anomalyclip_colab.ipynb
git commit -m "results: AnomalyCLIP VisA reproduction (§2 gate); pin repo commit + checkpoint shas"
```

**If any object misses ±1.0:** do not report MVTec AD 2 numbers. Likely causes, in order — the wrong
checkpoint (VisA-trained vs MVTec-AD-trained mixed up), a preprocessing mismatch (AnomalyCLIP's exact
CLIP resolution/transform), the repo commit differing from the paper's, or the wrong published
baseline (confirm it is zero-shot image-AUROC). Debug against the repo's own `test.py` results on
VisA; if it genuinely cannot be reproduced, flag AnomalyCLIP as such in every table (protocol §2).

---

## What this plan does NOT do, and why

- **The full MVTec AD 2 grid** (M3) — begins only once Phase D passes.
- **The evaluation server** (M4) — behind the pending registration (`docs/datasets-access.md`).
- **AdaCLIP and SAA+** — each its own plan. AdaCLIP is also auxiliary-trained and needs the same kind
  of overlap audit; SAA+ is training-free (GroundingDINO + SAM), no aux-training concern.
