# WinCLIP Zero-Shot Adapter Implementation Plan

> **Split execution.** Tasks 1–2 are CPU-verifiable and run by subagents (TDD, `pytest`). The
> Colab phases (A–D) need anomalib, CLIP weights and a GPU — none in the dev environment — so they
> run **interactively in a Colab GPU session**, like the PatchCore backend playbook. The anomalib
> code in the Colab phases is written from the library's published API (context7), and each phase
> has a VERIFY step against the installed version, because the `>=1.1` range varies and fabricating
> an upstream API is what the protocol forbids. Steps use checkbox (`- [ ]`) syntax.
>
> For the CPU tasks only: REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**Goal:** Build the WinCLIP zero-shot adapter over an injectable backend (CPU-testable), then implement and validate the real anomalib WinCLIP backend on Colab against the ±1pt VisA reproduction gate.

**Architecture:** WinCLIP is zero-shot — no fit, it slots into the existing `AnomalyMethod` contract with just `prepare` + `predict`. The adapter `WinClipRef` wraps a backend with a single `score(image, category) -> (raw_score, raw_map)`; the category is passed through because it is the object noun in WinCLIP's handcrafted prompt ensemble (verbatim from the paper, protocol §3), which changes the text embeddings. anomalib and CLIP are reached only through the injected backend, so the adapter is fully CPU-tested with a fake and the real CLIP forward is GPU-only. Raw score and map are passed through and the map upsampled to native resolution (protocol v0.2.6).

**Tech Stack:** Python 3.10+, numpy, pytest (CPU tasks); Colab T4 GPU, anomalib `>=1.1`, torch, open-clip (Colab phases).

## Global Constraints

- CPU tasks: every test passes with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`. No test and no module imported at load time may require torch, anomalib, transformers, open_clip or any YAML library. anomalib/CLIP are reached only through the injected backend.
- WinCLIP is **zero-shot** (`zero_shot = True`): no fit path. WinCLIP+ (few-shot) is out of scope — a separate plan.
- **Prompts verbatim from the WinCLIP paper** (protocol §3): no prompt tuning, no per-category prompt engineering. anomalib's WinClip encodes the paper's ensemble; the `class_name` is the only per-category knob and is set to the MVTec AD 2 category noun, consistent across the fixed ensemble.
- **Provenance is anomalib** (protocol §3 priority 2 — the official code's ensemble reproduced by anomalib). The §2 VisA reproduction gate is the check that this reproduction is faithful. Record the resolved anomalib version in `configs/methods/winclip.yaml`.
- Anomaly maps and scores are RAW (protocol v0.2.6); `WinClipRef` upsamples the map to the input image's native resolution (protocol §4). Never per-image normalise.
- **The §2 VisA reproduction gate is a hard acceptance criterion**: no MVTec AD 2 number until WinCLIP reproduces its published zero-shot VisA image-AUROC within ±1.0 through this adapter and this repo's metrics. A miss flags the method in every table.

## Files

| File | Responsibility |
|---|---|
| `src/vlmab/methods/winclip.py` | `WinClipRef` adapter over an injectable backend (CPU) |
| `configs/methods/winclip.yaml` | pre-registered WinCLIP config (CPU) |
| `src/vlmab/methods/registry.py` | register `winclip` (CPU) |
| `src/vlmab/methods/winclip_backend.py` | the real anomalib WinCLIP backend, lazy import (Colab) |
| `notebooks/winclip_colab.ipynb` | the Colab driver — the phases below (Colab) |
| `results/reproduction/winclip_visa.md` | the recorded VisA reproduction table (Colab) |

---

## Task 1: WinClipRef adapter over an injectable backend

**Files:**
- Modify: `src/vlmab/methods/winclip.py` (replaces the stub)
- Modify: `configs/methods/winclip.yaml`
- Create: `tests/test_winclip.py`

**Interfaces:**
- Consumes: `AnomalyMethod`, `Prediction`, `MethodNotRunnable` (base); `upsample_to` (postprocess);
  `assert_valid_prediction` (contract harness).
- Produces:
  - A backend protocol: an object with `score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]`.
  - `WinClipRef(backend=None)` with `name = "winclip"`, `zero_shot = True`.

- [ ] **Step 1: Pre-register the config**

Replace `configs/methods/winclip.yaml` entirely with:

```yaml
name: winclip
zero_shot: true
# WinCLIP zero-shot (Jeong et al., CVPR 2023). Provenance: anomalib implementation (protocol §3
# priority 2 — the official code's prompt ensemble is reproduced by anomalib; the §2 VisA
# reproduction gate is the check). Record the resolved anomalib version on first Colab run.
source: anomalib
anomalib_version: ">=1.1"        # exact pin resolved at integration; record the resolved version
backbone: ViT-B-16-plus-240      # WinCLIP's CLIP backbone
k_shot: 0                        # zero-shot; WinCLIP+ (few-shot) is a separate plan
scales: [2, 3]                   # multi-scale window sizes
class_name: per_category         # the object noun in the prompt ensemble = the MVTec AD 2 category
prompts: paper_verbatim          # protocol §3: no prompt tuning; anomalib encodes the paper ensemble
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_winclip.py`:

```python
import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.winclip import WinClipRef


class _FakeBackend:
    """Stands in for the real anomalib WinCLIP backend. Records the category, returns a raw
    score plus a coarse map (like CLIP's windowed anomaly map before upsampling)."""

    def __init__(self):
        self.seen = []

    def score(self, image, category):
        self.seen.append(category)
        m = np.zeros((15, 15), dtype=np.float32)
        m[3, 4] = 2.7
        return 2.7, m


def test_is_zero_shot():
    assert WinClipRef().zero_shot is True


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        WinClipRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    backend = _FakeBackend()
    m = WinClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "fruit_jelly")
    assert backend.seen == ["fruit_jelly"]


def test_predict_returns_a_native_resolution_raw_map():
    m = WinClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((75, 60, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(2.7)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (75, 60)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(2.7)   # the hot window survives upsampling


def test_predict_without_a_backend_is_not_runnable():
    m = WinClipRef()  # no backend, prepare() not called
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
```

- [ ] **Step 3: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_winclip.py -q`
Expected: FAIL — `ImportError: cannot import name 'WinClipRef' from 'vlmab.methods.winclip'`

- [ ] **Step 4: Implement**

Replace `src/vlmab/methods/winclip.py` entirely with:

```python
"""WinCLIP zero-shot adapter (Jeong et al., CVPR 2023) — CLIP with handcrafted prompt ensembles.

WinCLIP's substance is the anomalib call: a pre-trained CLIP scores each image against text
embeddings of normal/anomalous state prompts, with multi-scale windows for localisation. That
needs anomalib, CLIP weights and a GPU, so it lives behind an injected backend written on Colab
(the WinCLIP Colab phases in docs/superpowers/plans/2026-07-24-winclip-zeroshot-adapter.md). Everything here — passing the per-category class name into
the prompts, the native-resolution map, the raw pass-through — is tested with a fake backend.

WinCLIP is zero-shot: no fit, no training. The category is passed to the backend because it is the
object noun in the prompt ensemble ("a photo of a {damaged} {vial}"), taken verbatim from the
paper (protocol §3), which changes the text embeddings.

The backend is any object with:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class WinClipRef(AnomalyMethod):
    name = "winclip"
    zero_shot = True

    def __init__(self, backend=None):
        self._backend = backend

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real anomalib WinCLIP
        backend is Colab-only (it needs anomalib, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs anomalib + CLIP + GPU
            raise MethodNotRunnable(
                "WinClipRef needs an anomalib WinCLIP backend; build it in a GPU session with "
                "anomalib installed and inject it (see the WinCLIP Colab plan under docs/superpowers/plans/)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("WinClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_winclip.py -q`
Expected: PASS (5 passed)

- [ ] **Step 6: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/winclip.py configs/methods/winclip.yaml tests/test_winclip.py
git commit -m "feat: WinCLIP zero-shot adapter over an injectable backend"
```

---

## Task 2: Register WinCLIP

**Files:**
- Modify: `src/vlmab/methods/registry.py`
- Modify: `tests/test_registry.py`

**Interfaces:**
- Consumes: `WinClipRef` (Task 1).
- Produces: `build_method("winclip")` returns a `WinClipRef`; `available()` includes it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_registry.py`:

```python
def test_builds_winclip():
    from vlmab.methods.winclip import WinClipRef

    assert isinstance(build_method("winclip"), WinClipRef)
    assert "winclip" in available()
```

- [ ] **Step 2: Run to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -k winclip -q`
Expected: FAIL — `KeyError: "unknown method 'winclip'..."`

- [ ] **Step 3: Register it**

In `src/vlmab/methods/registry.py`, add the import (next to the other method imports) and the entry.
The imports gain:

```python
from vlmab.methods.winclip import WinClipRef
```

and the `_REGISTRY` dict gains the entry:

```python
    "winclip": WinClipRef,          # CPU-usable only with an injected backend; prepare() gates the rest
```

- [ ] **Step 4: Run to verify it passes**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_registry.py -q`
Expected: PASS

- [ ] **Step 5: Run the full suite and commit**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS. Record the count.

```bash
git add src/vlmab/methods/registry.py tests/test_registry.py
git commit -m "feat: register the WinCLIP zero-shot adapter"
```

---

# Colab phases (manual, GPU) — the real anomalib WinCLIP backend

These run in a Colab GPU session. Every anomalib call is from the published API; the VERIFY steps
check it against the installed version. The acceptance gate is Phase D.

## Phase A — Environment and API verification

- [ ] **A.1 — GPU, repo, anomalib**

```python
!nvidia-smi -L
!git clone https://github.com/andrudebaran7/vlm-anomaly-bench.git && cd vlm-anomaly-bench && pip install -e . --quiet
!pip install "anomalib>=1.1" open_clip_torch --quiet
import anomalib, torch; print("anomalib", anomalib.__version__, "| cuda", torch.cuda.is_available())
```

Record the resolved `anomalib.__version__`: ____ (goes into `configs/methods/winclip.yaml`, Phase D).

- [ ] **A.2 — VERIFY the WinClip API against the installed version**

```python
import inspect
from anomalib.models import WinClip
print(inspect.signature(WinClip.__init__))
#   expect class_name, k_shot (default 0), scales (default (2,3)) among the params.
print(WinClip.configure_pre_processor().transform)   # the image transform score() will apply
```

Confirm `k_shot=0` is the zero-shot default and the constructor accepts `class_name` and `scales`.
Any mismatch → adapt Phase B to the installed signature before proceeding.

## Phase B — The WinCLIP backend

- [ ] **B.1 — Write `src/vlmab/methods/winclip_backend.py`**

```python
"""anomalib WinCLIP backend for WinClipRef (GPU/Colab only).

Bridges the seam score(image, category) -> (raw_score, raw_map) to anomalib's zero-shot WinClip.
WinCLIP is zero-shot (no fit); the category is the object noun in the prompt ensemble, so a
WinClip is built (and its text embeddings computed) once per distinct category and cached. The
forward returns anomalib's RAW pred_score and anomaly_map (protocol v0.2.6 — no per-image
normalisation; WinClipRef upsamples the map to native resolution).

anomalib, torch and CLIP are imported lazily so this module imports in CI (never constructed there).
"""
import numpy as np


class WinClipBackend:
    def __init__(
        self,
        backbone: str = "ViT-B-16-plus-240",
        k_shot: int = 0,
        scales: tuple[int, ...] = (2, 3),
        device: str = "cuda",
    ):
        self._backbone = backbone
        self._k_shot = k_shot
        self._scales = tuple(scales)
        self._device = device
        self._cache: dict[str, object] = {}   # category -> a prepared WinClip

    def _model_for(self, category: str):
        import torch
        from anomalib.models import WinClip

        if category not in self._cache:
            model = WinClip(class_name=category, k_shot=self._k_shot, scales=self._scales)
            # Zero-shot WinCLIP computes its text (prompt) embeddings from class_name; ensure the
            # model is set up and in eval mode on the GPU before forwarding.
            # VERIFY: the installed version may compute embeddings in setup()/on first forward, or
            # expose a collect_text_embeddings-style call. Adapt this line to what it needs.
            if hasattr(model, "setup"):
                model.setup(stage="predict")
            model.eval().to(self._device)
            self._transform = type(model).configure_pre_processor().transform
            self._cache[category] = model
        return self._cache[category]

    def score(self, image: np.ndarray, category: str) -> tuple[float, np.ndarray]:
        import torch
        from PIL import Image

        model = self._model_for(category)
        tensor = self._transform(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(self._device)
        with torch.no_grad():
            out = model(tensor)   # -> InferenceBatch(pred_score, anomaly_map, ...)

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy()
        amap = amap.reshape(amap.shape[-2], amap.shape[-1]).astype(np.float32)
        return score, amap
```

VERIFY on the installed version: the setup/embedding-computation call, and that `model(tensor)`
returns `.pred_score` and `.anomaly_map`. If the container differs, adapt the extraction only.

- [ ] **B.2 — Smoke test the backend**

```python
import numpy as np
from vlmab.methods.winclip_backend import WinClipBackend

rng = np.random.default_rng(0)
normal = rng.integers(90, 110, (240, 240, 3), dtype=np.uint8)
anom = normal.copy(); anom[40:90, 40:90] = 255

b = WinClipBackend()
s_n, m_n = b.score(normal, "object")
s_a, m_a = b.score(anom, "object")
assert m_a.ndim == 2 and np.isfinite(m_a).all()
assert isinstance(s_a, float) and np.isfinite(s_a)
print("normal:", round(s_n, 4), " anomalous:", round(s_a, 4), " map:", m_a.shape)
```

WinCLIP is zero-shot, so a random-noise "defect" may not clearly separate on a meaningless image —
do not hard-assert `s_a > s_n` here (unlike PatchCore, which fits on the same normals). The real
separation check is Phase C/D on actual data. Assert only shape, finiteness, and type. Record the
scores.

- [ ] **B.3 — Commit the backend**

```bash
git add src/vlmab/methods/winclip_backend.py
git commit -m "feat: anomalib WinCLIP zero-shot backend bridging the score seam (GPU-only)"
```

Confirm `import vlmab.methods.winclip_backend` works without anomalib (lazy imports), so CI stays green.

## Phase C — End to end through the runner on one Vial category

- [ ] **C.1 — Verify Vial is present** (per `docs/datasets-access.md`):

```bash
python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
```

- [ ] **C.2 — Run WinCLIP over Vial's public test split**

```python
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.winclip import WinClipRef
from vlmab.methods.winclip_backend import WinClipBackend

method = WinClipRef(backend=WinClipBackend())
store = ResultStore("results/winclip/vial/shards")
meta = run_meta({"method": "winclip", "split": "test_public"}, seed=0)
run_evaluation(MVTecAD2("data/mvtec_ad2"), method, store, meta,
               categories=["vial"], split="test_public", maps_dir="results/winclip/vial/maps")
```

WinCLIP is zero-shot, so the runner never calls `fit` (it checks `not method.zero_shot`); it scores
each public-test image directly. The per-category `class_name="vial"` is set inside the backend.

- [ ] **C.3 — Aggregate per lighting condition**

```python
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore("results/winclip/vial/shards").load_all()
print(aggregate(df, by="meta_lighting")[
    ["meta_lighting", "i_auroc", "au_pro_030", "au_pro_005", "n"]].to_string(index=False))
```

Expected: a zero-shot method — I-AUROC above chance on `regular`, but not near PatchCore's full-shot
ceiling; the point of the study is exactly this gap. Record the table.

## Phase D — The VisA zero-shot reproduction gate (protocol §2)

The acceptance criterion. No MVTec AD 2 number until WinCLIP reproduces its published **zero-shot**
VisA image-AUROC within ±1.0 through this adapter and this repo's metrics.

- [ ] **D.1 — Get VisA** (CC BY 4.0, no account):

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar . && tar -xf VisA_20220922.tar
```

- [ ] **D.2 — Score each VisA object zero-shot and compute I-AUROC with the repo's metric**

```python
import numpy as np, pandas as pd
from PIL import Image
from vlmab.methods.winclip_backend import WinClipBackend
from vlmab.metrics.image_level import i_auroc

b = WinClipBackend()   # zero-shot: no fit, one backend for all objects; class_name set per object
def load(paths):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

rows = []
for obj, test_paths, test_labels in visa_objects():   # your loader over VisA's split CSV
    scores = np.array([b.score(img, obj)[0] for img in load(test_paths)], dtype=np.float64)
    rows.append({"object": obj, "i_auroc": i_auroc(np.array(test_labels), scores),
                 "n": len(test_labels)})
res = pd.DataFrame(rows)
res["published"] = res["object"].map(PUBLISHED_WINCLIP_VISA_IAUROC)   # WinCLIP paper's zero-shot table
res["delta"] = res["i_auroc"] - res["published"]
print(res.to_string(index=False)); print("max |delta|:", res["delta"].abs().max())
```

`PUBLISHED_WINCLIP_VISA_IAUROC` is the per-object zero-shot VisA image-AUROC from the WinCLIP paper
(record the exact table/source). Gate: **every object within ±1.0, i.e. `res["delta"].abs().max()
<= 1.0`.** Note the object noun each VisA object uses as `class_name` (e.g. "candle", "capsules") —
this is the WinCLIP-standard setup and part of what makes the reproduction faithful.

- [ ] **D.3 — Record the reproduction and pin the version**

Write `results/reproduction/winclip_visa.md` (table, anomalib version, pass/fail vs ±1.0). In
`configs/methods/winclip.yaml`, set `anomalib_version` to the resolved version from A.1. Commit:

```bash
git add results/reproduction/winclip_visa.md configs/methods/winclip.yaml notebooks/winclip_colab.ipynb
git commit -m "results: WinCLIP zero-shot VisA reproduction (protocol §2 gate); pin anomalib version"
```

**If any object misses ±1.0:** do not report MVTec AD 2 numbers. Likely causes, in order — the
`class_name` object noun differs from the paper's, a preprocessing mismatch (WinCLIP's exact CLIP
resolution), the wrong published baseline (confirm it is zero-shot image-AUROC), or anomalib's
prompt ensemble differing from the paper's (this is the priority-2-source risk protocol §3 names —
if so, flag WinCLIP as an anomalib reproduction in every table and record the discrepancy).

---

## What this plan does NOT do, and why

- **WinCLIP+ (few-shot)** — a separate plan. It needs `k_shot > 0`, `zero_shot = False`, and the
  runner's `fit` path to store reference embeddings; the zero-shot adapter here does not.
- **The full MVTec AD 2 grid** (M3) — begins only once Phase D passes; a mechanical loop over
  categories using this validated backend.
- **The evaluation server** (M4) — behind the pending registration (`docs/datasets-access.md`).
- **The remaining wrappers** (AnomalyCLIP, AdaCLIP, SAA+) — each its own plan; AnomalyCLIP and
  AdaCLIP are aux-trained (an overlap audit against MVTec AD 2 is required, protocol §3.1).
