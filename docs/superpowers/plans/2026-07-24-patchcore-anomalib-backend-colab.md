# PatchCore anomalib Backend — Colab Integration Playbook

> **NOT a CPU/TDD subagent plan.** This backend needs anomalib installed and a GPU, neither of
> which exists in the dev environment. It is executed **interactively in a Colab GPU session**,
> not by CPU subagents. Every anomalib call below is written from the library's published API
> (fetched via context7), but the `>=1.1` range varies, so **each phase has a VERIFY step that
> checks the call against the actually-installed version before relying on it** — the protocol
> requires verification over memory. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `PatchCoreBackend` that bridges the repo's `fit`/`score` seam to anomalib's PatchCore, and prove it reproduces PatchCore's published VisA image-AUROC within ±1.0 (protocol §2) before any MVTec AD 2 number is produced.

**Architecture:** The seam is already frozen and merged (Plan 4): `PatchCoreRef(backend)` expects a backend with `fit(train_images: Iterable[np.ndarray])` and `score(image: np.ndarray) -> tuple[float, np.ndarray]`. anomalib is directory-and-datamodule oriented, so the backend bridges: `fit` materialises the streamed normal images to a temp dir, builds a `Folder` datamodule and runs `engine.train` to fill the coreset memory bank; `score` pre-processes one image with PatchCore's own transform, runs the trained torch model forward, and returns anomalib's raw `pred_score` and `anomaly_map`. The backend lives in the repo with anomalib imported lazily, so the module still imports in CI (it is never constructed there). The Colab notebook injects it into `PatchCoreRef` and drives the existing runner.

**Tech Stack:** Colab (T4 GPU), anomalib `>=1.1` (exact version pinned in Phase 0), torch, the existing `vlmab` package.

## Global Constraints

- **Runs only on a GPU (Colab T4).** Nothing here is CPU-testable; the repo's CI never imports anomalib. `src/vlmab/methods/patchcore_backend.py` must import anomalib lazily (inside methods), so `import vlmab.methods.patchcore_backend` succeeds in CI but constructing/using the backend requires anomalib+GPU.
- **Anomaly maps and scores are RAW** (protocol v0.2.6): pass anomalib's `pred_score` and `anomaly_map` through unchanged; `PatchCoreRef` upsamples the map to native resolution. Never per-image normalise.
- **Hyperparameters are pre-registered** in `configs/methods/patchcore_ref.yaml`: `wide_resnet50_2`, layers `layer2`+`layer3`, `coreset_sampling_ratio: 0.1`, `num_neighbors: 9`. Use exactly these; record the resolved anomalib version there.
- **The §2 reproduction gate is a hard acceptance criterion.** No MVTec AD 2 number is reported until PatchCore reproduces its published VisA image-AUROC within ±1.0 through this adapter and the project's own metrics. A miss flags the anchor in every table (protocol §2).
- **The evaluation resolution is native** (protocol §4): the anomaly map is upsampled to the input image's native resolution by `PatchCoreRef` before metrics; the mask is never downsampled.

## Files

| File | Responsibility |
|---|---|
| `src/vlmab/methods/patchcore_backend.py` | the `PatchCoreBackend` bridging the seam to anomalib (lazy import) |
| `configs/methods/patchcore_ref.yaml` | record the resolved anomalib version (Phase 4) |
| `notebooks/patchcore_colab.ipynb` | the Colab driver (env, smoke test, VisA gate) — kept in-repo for provenance |
| `results/reproduction/patchcore_visa.md` | the recorded VisA reproduction table (Phase 3) |

---

## Phase 0 — Colab environment and API verification

The point of this phase is to pin the version and **catch API drift before writing backend code
against a signature that changed.**

- [ ] **Step 0.1 — GPU + repo**

```python
!nvidia-smi -L                      # confirm a GPU is attached
!git clone https://github.com/andrudebaran7/vlm-anomaly-bench.git
%cd vlm-anomaly-bench
!pip install -e . --quiet
```

- [ ] **Step 0.2 — Install anomalib and record the resolved version**

```python
!pip install "anomalib>=1.1" --quiet
import anomalib, torch
print("anomalib", anomalib.__version__, "| torch", torch.__version__,
      "| cuda", torch.cuda.is_available())
```

Record the resolved `anomalib.__version__` here: ____ . It goes into
`configs/methods/patchcore_ref.yaml` in Phase 4.

- [ ] **Step 0.3 — VERIFY the API shape against the installed version**

Run each of these and confirm they match what the backend code in Phase 1 assumes. If any differs,
stop and adapt Phase 1 to the installed signatures — do NOT proceed on the docs' shape alone.

```python
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode

# (a) Patchcore constructor accepts the pre-registered hyperparameters:
import inspect
print(inspect.signature(Patchcore.__init__))
#   expect backbone, layers, coreset_sampling_ratio, num_neighbors among the params.

# (b) The default pre-processor transform (used to preprocess a single image in score()):
print(Patchcore.configure_pre_processor().transform)
#   expect Resize(256) -> CenterCrop(224) -> Normalize(imagenet).

# (c) A forward pass returns an object exposing pred_score and anomaly_map:
print([f for f in ("pred_score", "anomaly_map", "pred_label", "pred_mask")])
#   confirm InferenceBatch (docs) — verified concretely by the smoke test in Phase 1.

# (d) Folder accepts normal-only with splits disabled:
print(inspect.signature(Folder.__init__))
#   expect name, root, normal_dir, val_split_mode, test_split_mode, task.
```

Note in your run whether (a)–(d) matched. Any mismatch is the single most likely failure of this
whole plan, and this step is where it is cheap to find.

---

## Phase 1 — The backend

Bridges the frozen seam to anomalib. Written from the published API; the smoke test at the end is
what actually proves it works on the installed version.

- [ ] **Step 1.1 — Write `src/vlmab/methods/patchcore_backend.py`**

```python
"""anomalib PatchCore backend for PatchCoreRef (GPU/Colab only).

Bridges the repo's injectable seam — fit(train_images) / score(image) -> (float, HxW map) — to
anomalib, which is directory-and-datamodule oriented. `fit` materialises the streamed normal
images to a temp directory and runs anomalib's PatchCore training to fill the coreset memory bank;
`score` pre-processes one image with PatchCore's own transform, runs the trained torch model
forward, and returns anomalib's RAW pred_score and anomaly_map (protocol v0.2.6 — no per-image
normalisation; PatchCoreRef upsamples the map to native resolution).

anomalib and torch are imported lazily so this module imports in CI (it is never constructed
there). Hyperparameters come from configs/methods/patchcore_ref.yaml.
"""
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


class PatchCoreBackend:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: tuple[str, ...] = ("layer2", "layer3"),
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cuda",
    ):
        from anomalib.models import Patchcore  # lazy: GPU-only
        from anomalib.engine import Engine

        self._device = device
        self._model = Patchcore(
            backbone=backbone,
            layers=list(layers),
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
        )
        # PatchCore trains for a single epoch to fill the memory bank; disable checkpoints/logging.
        self._engine = Engine(
            task="classification",
            accelerator="gpu",
            devices=1,
            max_epochs=1,
            logger=False,
            enable_checkpointing=False,
        )
        self._transform = type(self._model).configure_pre_processor().transform
        self._fitted = False

    def fit(self, train_images: Iterable[np.ndarray]) -> None:
        from anomalib.data import Folder
        from anomalib.data.utils import TestSplitMode, ValSplitMode
        from PIL import Image

        # Materialise the streamed normal images to a temp dir anomalib's Folder can read.
        self._tmp = tempfile.TemporaryDirectory()
        good = Path(self._tmp.name) / "good"
        good.mkdir(parents=True)
        n = 0
        for i, img in enumerate(train_images):
            Image.fromarray(np.asarray(img, dtype=np.uint8)).save(good / f"{i:05d}.png")
            n += 1
        if n == 0:
            raise ValueError("PatchCoreBackend.fit received no training images")

        datamodule = Folder(
            name="fit",
            root=self._tmp.name,
            normal_dir="good",
            val_split_mode=ValSplitMode.NONE,
            test_split_mode=TestSplitMode.NONE,
            task="classification",
        )
        datamodule.setup()
        self._engine.train(model=self._model, datamodule=datamodule)
        self._model.eval()
        self._fitted = True

    def score(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        import torch
        from PIL import Image

        if not self._fitted:
            raise RuntimeError("PatchCoreBackend.score called before fit")

        tensor = self._transform(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        tensor = tensor.unsqueeze(0).to(self._device)  # 1x3xHxW
        with torch.no_grad():
            out = self._model(tensor)   # -> InferenceBatch(pred_score, anomaly_map, ...)

        score = float(out.pred_score.reshape(-1)[0].item())
        amap = out.anomaly_map.detach().cpu().numpy().reshape(
            out.anomaly_map.shape[-2], out.anomaly_map.shape[-1]
        ).astype(np.float32)
        return score, amap
```

VERIFY, on the installed version, the two calls the docs are least certain about:
- `self._model(tensor)` returning an object with `.pred_score` and `.anomaly_map`. If the installed
  version returns a different container, adapt the two extraction lines only.
- The `Engine(...)` kwargs (`accelerator`/`devices`/`max_epochs`/`logger`/`enable_checkpointing`).
  If a kwarg is rejected, drop it — they are convenience settings, not correctness.

- [ ] **Step 1.2 — Smoke test the backend in the notebook (the real verification)**

```python
import numpy as np
from vlmab.methods.patchcore_backend import PatchCoreBackend

rng = np.random.default_rng(0)
normal = [rng.integers(90, 110, (256, 256, 3), dtype=np.uint8) for _ in range(20)]  # flat-ish
anom = normal[0].copy(); anom[40:80, 40:80] = 255                                   # bright patch

b = PatchCoreBackend()
b.fit(iter(normal))
s_normal, m_normal = b.score(normal[1])
s_anom, m_anom = b.score(anom)

assert m_anom.ndim == 2 and np.isfinite(m_anom).all(), (m_anom.shape,)
assert isinstance(s_anom, float) and np.isfinite(s_anom)
assert s_anom > s_normal, (s_anom, s_normal)   # the defect must score higher
print("OK  normal:", round(s_normal, 4), " anomalous:", round(s_anom, 4), " map:", m_anom.shape)
```

Expected: the assertion holds — a bright injected patch scores above a normal image, the map is a
finite 2-D array. **If `s_anom <= s_normal`, the fit or score path is wrong; debug here before
touching real data.** Record the two scores in your run.

- [ ] **Step 1.3 — Commit the backend**

```bash
git add src/vlmab/methods/patchcore_backend.py
git commit -m "feat: anomalib PatchCore backend bridging the fit/score seam (GPU-only)"
```

Also confirm CI stays green (the module imports without anomalib because the imports are lazy):
locally, `import vlmab.methods.patchcore_backend` must not require anomalib. If the repo's CI run on
this commit fails on an anomalib import, the import is not lazy enough — move it inside the method.

---

## Phase 2 — End to end through the existing runner, on one Vial category

Prove the backend drives the whole path — the same `run_evaluation` → shard → `aggregate` the
`intensity_baseline` already exercises — before spending a full grid.

- [ ] **Step 2.1 — Download and verify one MVTec AD 2 category**

Fetch the Vial category to `data/mvtec_ad2/vial` (per `docs/datasets-access.md`), then:

```bash
python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
```

Expected: exit 0, `OK`, the verified counts (train regular=291, validation regular=41, test_public
140 across 7 conditions, test_private 276, test_private_mixed 276).

- [ ] **Step 2.2 — Run PatchCore over Vial's public test split via the runner**

```python
from pathlib import Path
from vlmab.datasets.mvtec_ad2 import MVTecAD2
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.eval.provenance import run_meta
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.patchcore_backend import PatchCoreBackend

method = PatchCoreRef(backend=PatchCoreBackend())
store = ResultStore("results/patchcore/vial/shards")
meta = run_meta({"method": "patchcore_ref", "split": "test_public"}, seed=0)

run_evaluation(
    MVTecAD2("data/mvtec_ad2"), method, store, meta,
    categories=["vial"], split="test_public",
    maps_dir="results/patchcore/vial/maps",
)
```

The runner calls `method.fit(train_images, "vial")` (from the `train` split, per protocol §3) before
scoring the 140 public-test images. If it raises the runner's float16 overflow guard, PatchCore's raw
scores exceed 65504 — record the value; it means the map needs a documented scale, which is a §4
amendment, not a silent change.

- [ ] **Step 2.3 — Aggregate per lighting condition and sanity-check**

```python
from vlmab.eval.store import ResultStore
from vlmab.eval.aggregate import aggregate
df = ResultStore("results/patchcore/vial/shards").load_all()
print(aggregate(df, by="meta_lighting")[
    ["meta_lighting", "i_auroc", "au_pro_030", "au_pro_005", "n"]].to_string(index=False))
```

Expected: PatchCore, a strong full-shot method, scores I-AUROC **well above** the
`intensity_baseline`'s 0.5 on `regular` lighting — a full-shot anchor that does not beat chance on
its easiest condition is broken. Record the table. A large drop from `regular` to the `shift_*` /
`over/underexposed` conditions is the expected lighting-robustness story, not an error.

---

## Phase 3 — The VisA ±1pt reproduction gate (protocol §2)

The acceptance criterion. Until PatchCore reproduces its published VisA image-AUROC within ±1.0
through THIS adapter and THIS repo's metrics, no MVTec AD 2 number is reported.

- [ ] **Step 3.1 — Get VisA (public, no account)**

VisA is CC BY 4.0 on AWS Open Data (`docs/datasets-access.md`):

```bash
aws s3 cp --no-sign-request s3://amazon-visual-anomaly/VisA_20220922.tar . && tar -xf VisA_20220922.tar
```

VisA's layout differs from MVTec AD 2 (per-object `Data/Images/{Normal,Anomaly}` + a split CSV), so
it does NOT use `MVTecAD2`. For the reproduction only, load each object's train-normal images (for
`fit`) and its test images with labels (for scoring) directly, and feed them through
`PatchCoreBackend` + the repo's `image_metrics`.

- [ ] **Step 3.2 — Score each VisA object and compute I-AUROC with the repo's own metric**

```python
import numpy as np, pandas as pd
from PIL import Image
from vlmab.methods.patchcore_backend import PatchCoreBackend
from vlmab.metrics.image_level import i_auroc

def load(paths):
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths]

rows = []
for obj, train_normal_paths, test_paths, test_labels in visa_objects():  # your loader over the CSV
    b = PatchCoreBackend()
    b.fit(iter(load(train_normal_paths)))
    scores = np.array([b.score(img)[0] for img in load(test_paths)], dtype=np.float64)
    rows.append({"object": obj, "i_auroc": i_auroc(np.array(test_labels), scores),
                 "n": len(test_labels)})
res = pd.DataFrame(rows)
res["published"] = res["object"].map(PUBLISHED_VISA_IAUROC)   # from the PatchCore/anomalib VisA table
res["delta"] = res["i_auroc"] - res["published"]
print(res.to_string(index=False)); print("max |delta|:", res["delta"].abs().max())
```

`PUBLISHED_VISA_IAUROC` is the per-object VisA image-AUROC from anomalib's own reported PatchCore
results (record the exact source next to the table). The gate: **every object within ±1.0 point,
i.e. `res["delta"].abs().max() <= 1.0`.**

- [ ] **Step 3.3 — Record the reproduction result**

Write `results/reproduction/patchcore_visa.md` with the table (object, ours, published, delta),
the anomalib version, and the pass/fail verdict against the ±1.0 gate. Commit it.

```bash
git add results/reproduction/patchcore_visa.md
git commit -m "results: PatchCore VisA reproduction (protocol §2 gate)"
```

**If any object misses ±1.0:** do not proceed to MVTec AD 2 reporting. The likely causes, in order —
a preprocessing mismatch (VisA images are large; confirm the Resize/CenterCrop matches published),
the wrong published baseline (confirm the source and that it is image-AUROC, not pixel), or a
coreset-ratio/backbone mismatch. Debug against the published setup; if it genuinely cannot be
reproduced, flag the anchor as such in every table (protocol §2) and record why.

---

## Phase 4 — Freeze provenance

- [ ] **Step 4.1 — Record the resolved anomalib version in the config**

In `configs/methods/patchcore_ref.yaml`, replace the `anomalib_version: ">=1.1"` line's placeholder
intent with the exact resolved version from Step 0.2, e.g. `anomalib_version: "1.2.0"` (use whatever
Step 0.2 printed), and add a comment noting it was the version the VisA gate passed on.

- [ ] **Step 4.2 — Commit the notebook and the config**

```bash
git add notebooks/patchcore_colab.ipynb configs/methods/patchcore_ref.yaml
git commit -m "chore: pin the anomalib version PatchCore reproduced VisA on; save the Colab driver"
```

- [ ] **Step 4.3 — Push and confirm CI**

Push the branch. CI runs on CPU and never imports anomalib, so the only thing it exercises from this
plan is that `patchcore_backend.py` imports lazily. Confirm both matrix jobs (3.11, 3.13) stay green.

---

## What this playbook does NOT do, and why

- **It does not run the full MVTec AD 2 grid.** That is M3, and it only begins once the VisA gate in
  Phase 3 passes. It is a separate, mechanical loop over categories using the exact backend this
  playbook validated.
- **It does not touch the evaluation server.** The private test split is M4, still behind the pending
  registration (`docs/datasets-access.md`).
- **It does not add the four zero-shot wrappers** (WinCLIP, AnomalyCLIP, AdaCLIP, SAA+). Each is its
  own Colab plan; none needs the full-shot `fit` path.

## Honesty note for whoever executes this

Every anomalib call here is from the library's published API, but the exact `>=1.1` signatures were
not run in the environment that wrote this plan. Phase 0 Step 0.3 and the Phase 1 smoke test are the
real verification — treat a mismatch there as expected maintenance, not a plan failure, and adapt the
call to the installed version. The one thing that must not be adapted away is the ±1.0 VisA gate: it
is the pre-registered protocol's guarantee that the anchor is trustworthy.
