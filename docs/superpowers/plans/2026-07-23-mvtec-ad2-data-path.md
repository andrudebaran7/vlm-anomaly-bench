# MVTec AD 2 Data Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the verified MVTec AD 2 layout into a working data path — loader, layout verification, and metric aggregation — so that adding a method adapter is the only thing left before real numbers exist.

**Architecture:** A loader that yields the existing `Sample` contract from the real five-split tree, parsing the lighting condition out of each filename so results can be broken down by it. A verification CLI that fails loudly on a malformed download. An aggregation module that turns the runner's parquet shards plus saved anomaly maps into image- and pixel-level metrics, grouped by lighting condition. Everything is tested against a synthetic fixture tree that mirrors the real layout, so the suite still needs no dataset on disk.

**Tech Stack:** Python 3.10+, numpy, scipy, scikit-learn, pandas + pyarrow, Pillow, pytest.

## Verified layout (measured from `vial.tar.gz`, 2026-07-23)

Do not re-derive these from the dataset paper or the download page — they were read off the real archive.

```
<root>/<category>/
  train/good/                    291 files   (all lighting: regular)
  validation/good/                41 files   (all lighting: regular)
  test_public/
    good/                         35 files   (7 conditions x 5)
    bad/                         105 files   (7 conditions x 15)
    ground_truth/bad/            105 files   (one mask per bad image)
  test_private/                  276 files   (flat; regular only; labels withheld)
  test_private_mixed/            276 files   (flat; mixed only; labels withheld)
  readme.txt, license.txt        (attribution + CC BY-NC-SA 4.0; no metric definitions)
```

- **Image filename:** `{index:03d}_{condition}.png`, e.g. `000_shift_1.png`.
- **Mask filename:** `{index:03d}_{condition}_mask.png`, inside `test_public/ground_truth/bad/`.
- **Lighting conditions in `test_public`:** `regular`, `overexposed`, `underexposed`, `shift_1`,
  `shift_2`, `shift_3`, `shift_4`. `train`/`validation`/`test_private` are `regular` only;
  `test_private_mixed` is `mixed` only.
- **Images are 1400x1900, 8-bit GRAYSCALE** (2.66 MP), not RGB. Verified on Vial only — other
  categories are unverified and the loader must not assume either way.
- `test_public/good/000_regular.png` and `test_public/bad/000_regular.png` **share a stem**. This
  is the collision the runner's `map_filename` already guards against; it is real, not theoretical.

## Global Constraints

- Compute target is Colab free tier: single T4, 16 GB VRAM, ~12.7 GB host RAM, fp16 only, sessions
  disconnect at ~12 hours.
- **Every test must pass with only `numpy scipy scikit-learn pandas pyarrow pillow pytest`
  installed.** No test may import torch, anomalib, or open_clip. Pillow is newly permitted and is
  added to CI by Task 2.
- **No test may require the real dataset.** Tests build a synthetic tree matching the layout above.
- Images are converted to 3-channel RGB at load time by `Image.convert("RGB")`, which replicates a
  grayscale channel and passes an already-RGB image through unchanged. Every method sees RGB.
- Private-split samples carry `label = LABEL_UNKNOWN` (`-1`). Accuracy metrics refuse to run on them.
- Source in `src/vlmab/`, tests in `tests/`.
- Category downloads are per-category (largest 10 GB, Vial 0.77 GB); one category is the download
  unit, the resume unit and the shard unit.

## File structure

| File | Responsibility |
|---|---|
| `src/vlmab/datasets/mvtec_ad2.py` | The loader: split enumeration, labels, mask pairing, lighting parsing, image decode |
| `src/vlmab/eval/aggregate.py` | Parquet shards + saved maps -> image/pixel metrics, optionally grouped |
| `scripts/prepare_data.py` | CLI that verifies an extracted category against the layout above |
| `tests/mvtec_tree.py` | Shared helper that builds a synthetic category tree |
| `tests/test_mvtec_ad2.py`, `tests/test_aggregate.py`, `tests/test_prepare_data.py` | Their tests |

Out of scope for this plan, with reasons, in the closing section.

---

### Task 1: Record the verified layout and amend the protocol

**Files:**
- Modify: `docs/datasets-access.md`
- Modify: `docs/protocol.md`

**Interfaces:**
- Consumes: nothing.
- Produces: the two decisions Tasks 2 and 5 implement — RGB conversion, and the private-split label
  sentinel. No code symbols.

- [ ] **Step 1: Append the verified layout to `docs/datasets-access.md`**

Under the MVTec AD 2 section, after the per-category download table, add:

```markdown
### Verified directory layout (read from `vial.tar.gz`, 2026-07-23)

    <root>/<category>/
      train/good/                  defect-free, lighting: regular only
      validation/good/             defect-free, lighting: regular only
      test_public/
        good/                      normal, all 7 lighting conditions
        bad/                       anomalous, all 7 lighting conditions
        ground_truth/bad/          one mask per bad image
      test_private/                flat, regular only, labels withheld
      test_private_mixed/          flat, mixed only, labels withheld

- Image filename: `{index:03d}_{condition}.png` (e.g. `000_shift_1.png`).
- Mask filename: `{index:03d}_{condition}_mask.png`.
- Lighting conditions: `regular`, `overexposed`, `underexposed`, `shift_1`..`shift_4`, and
  `mixed` in `test_private_mixed`.
- Vial counts: train 291, validation 41, test_public 35 good + 105 bad (+105 masks),
  test_private 276, test_private_mixed 276.
- **Vial images are 1400x1900, 8-bit grayscale (2.66 MP)** — not RGB. Verified for Vial only;
  record per category as each is downloaded, since the loader must handle both.
- `test_public/good/` and `test_public/bad/` reuse the same stems (`000_regular.png` exists in
  both), so anomaly-map filenames must be derived from more than the stem.
- The bundled `readme.txt` carries attribution and the CC BY-NC-SA 4.0 licence only. It does
  **not** document the official metric definitions — those remain behind the evaluation server.

Per-category image format (record on download):

| Category | Resolution | Channels |
|---|---|---|
| Vial | 1400x1900 | 8-bit grayscale |
| Can | ____ | ____ |
| Fabric | ____ | ____ |
| Fruit Jelly | ____ | ____ |
| Rice | ____ | ____ |
| Sheet Metal | ____ | ____ |
| Wallplugs | ____ | ____ |
| Walnuts | ____ | ____ |
```

- [ ] **Step 2: Amend `docs/protocol.md` §3 with the colour-channel decision**

Append to §3:

```markdown
- **Colour channels.** MVTec AD 2 images are grayscale in at least one category (Vial: 8-bit,
  1400x1900). Every method in this study expects 3-channel RGB input, so the loader converts
  with PIL's `convert("RGB")`, which replicates the single channel and passes an already-RGB
  image through unchanged. This is applied identically to every method and every category, so
  it cannot advantage one method over another. It is recorded here because it is a
  preprocessing decision that touches every reported number.
```

- [ ] **Step 3: Amend `docs/protocol.md` §2 with the private-split label rule**

Append to the MVTec AD 2 bullet in §2:

```markdown
  Samples from `test_private` and `test_private_mixed` carry no label, since their ground truth
  is withheld. The loader marks them `label = -1` and every accuracy metric refuses to run on
  them rather than silently treating unknown as normal.
```

- [ ] **Step 4: Add the Changelog entry**

Append to the Changelog in `docs/protocol.md`:

```markdown
- 2026-07-23 — v0.2.2. Layout verified against the real archive rather than the download page:
  five splits with the subdirectory structure recorded in docs/datasets-access.md, lighting
  condition encoded in every filename, masks named `{stem}_mask.png`. Adds two preprocessing
  rules that touch every number: grayscale images are converted to 3-channel RGB (§3), and
  withheld-label samples are marked -1 and excluded from accuracy metrics (§2). Official metric
  names still unverified. No evaluation rule changed.
```

- [ ] **Step 5: Verify the docs are consistent**

Run: `grep -c "test_private_mixed" docs/protocol.md docs/datasets-access.md`
Expected: at least 1 in each file.

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS (75 passed) — no code changed.

- [ ] **Step 6: Commit**

```bash
git add docs/protocol.md docs/datasets-access.md
git commit -m "docs: verified MVTec AD 2 layout; protocol v0.2.2 (RGB conversion, unlabelled splits)"
```

---

### Task 2: MVTec AD 2 loader

**Files:**
- Modify: `.github/workflows/ci.yml:11` (add `pillow`)
- Modify: `src/vlmab/datasets/mvtec_ad2.py` (currently a stub docstring)
- Create: `tests/mvtec_tree.py`
- Create: `tests/test_mvtec_ad2.py`

**Interfaces:**
- Consumes: `Sample` and `AnomalyDataset` from `src/vlmab/datasets/base.py`.
- Produces:
  - `LABEL_UNKNOWN: int = -1`
  - `lighting_condition(image_path: Path) -> str`
  - `MVTecAD2(root: Path)` implementing `.name = "mvtec_ad2"`, `.categories() -> list[str]`,
    `.samples(split: str, category: str | None = None) -> Iterator[Sample]`,
    `.load_image(sample: Sample) -> np.ndarray` (HxWx3 uint8),
    `.load_mask(sample: Sample) -> np.ndarray | None` (HxW uint8, 0/1)
  - `MVTecAD2.SPLITS`, `.LABELLED_SPLITS`, `.UNLABELLED_SPLITS`
  - `tests/mvtec_tree.py::build_category(root: Path, category: str, conditions: Sequence[str] = ..., n_good: int = 2, n_bad: int = 2, size: tuple[int, int] = (8, 6), mode: str = "L") -> Path`

- [ ] **Step 1: Add Pillow to CI and put `tests/` and `scripts/` on the import path**

In `.github/workflows/ci.yml`, change the dependency line to:

```yaml
      - run: pip install numpy scipy scikit-learn pandas pyarrow pillow pytest
```

In `pyproject.toml`, under `[tool.pytest.ini_options]`, add:

```toml
pythonpath = ["scripts", "tests"]
```

`tests/` makes `tests/mvtec_tree.py` importable from test modules without a package; `scripts/`
is needed by Task 3 and is set here so both tasks share one configuration change rather than
each editing the same file.

- [ ] **Step 2: Write the synthetic tree helper**

Create `tests/mvtec_tree.py`:

```python
"""Build a synthetic MVTec AD 2 category tree.

Mirrors the layout measured from the real `vial.tar.gz`: five splits, `good`/`bad`/`ground_truth`
under `test_public`, flat unlabelled private splits, and the `{index:03d}_{condition}` filename
convention with `_mask` appended for ground truth. Keeping this in one place means a layout change
is a one-file edit rather than a hunt through test modules.
"""
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

CONDITIONS = ("regular", "overexposed", "underexposed", "shift_1", "shift_2", "shift_3", "shift_4")


def _write(path: Path, size: tuple[int, int], mode: str, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    if mode == "L":
        arr = np.full((h, w), value, dtype=np.uint8)
    else:
        arr = np.full((h, w, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode=mode).save(path)


def build_category(
    root: Path,
    category: str,
    conditions: Sequence[str] = CONDITIONS,
    n_good: int = 2,
    n_bad: int = 2,
    size: tuple[int, int] = (8, 6),
    mode: str = "L",
) -> Path:
    """Create <root>/<category>/ with every split populated. Returns the category directory.

    Pixel values encode provenance so a test can tell which file it loaded: good images are
    filled with 10, bad with 200, private with 50, mixed with 60, train with 20, validation
    with 30. Masks are 255 in a 2x2 corner block and 0 elsewhere.
    """
    cat = root / category

    for i in range(n_good):
        _write(cat / "train" / "good" / f"{i:03d}_regular.png", size, mode, 20)
    for i in range(n_good):
        _write(cat / "validation" / "good" / f"{i:03d}_regular.png", size, mode, 30)

    for cond in conditions:
        for i in range(n_good):
            _write(cat / "test_public" / "good" / f"{i:03d}_{cond}.png", size, mode, 10)
        for i in range(n_bad):
            _write(cat / "test_public" / "bad" / f"{i:03d}_{cond}.png", size, mode, 200)
            w, h = size
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[:2, :2] = 255
            gt = cat / "test_public" / "ground_truth" / "bad" / f"{i:03d}_{cond}_mask.png"
            gt.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask, mode="L").save(gt)

    for i in range(n_bad):
        _write(cat / "test_private" / f"{i:03d}_regular.png", size, mode, 50)
        _write(cat / "test_private_mixed" / f"{i:03d}_mixed.png", size, mode, 60)

    return cat
```

- [ ] **Step 3: Write the failing tests**

Create `tests/test_mvtec_ad2.py`:

```python
from pathlib import Path

import numpy as np
import pytest

from mvtec_tree import CONDITIONS, build_category
from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN, MVTecAD2, lighting_condition


@pytest.fixture
def dataset(tmp_path):
    build_category(tmp_path, "vial")
    build_category(tmp_path, "can")
    return MVTecAD2(tmp_path)


def test_lighting_condition_parses_the_filename_suffix():
    assert lighting_condition(Path("/x/000_regular.png")) == "regular"
    assert lighting_condition(Path("/x/013_shift_1.png")) == "shift_1"
    assert lighting_condition(Path("/x/007_mixed.png")) == "mixed"


def test_lighting_condition_rejects_a_name_with_no_condition():
    with pytest.raises(ValueError):
        lighting_condition(Path("/x/000.png"))


def test_categories_are_the_directories_sorted(dataset):
    assert dataset.categories() == ["can", "vial"]


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        MVTecAD2(tmp_path / "nope")


def test_unknown_split_is_rejected(dataset):
    with pytest.raises(ValueError):
        list(dataset.samples("test", "vial"))


def test_train_and_validation_are_defect_free(dataset):
    for split in ("train", "validation"):
        samples = list(dataset.samples(split, "vial"))
        assert samples, split
        assert {s.label for s in samples} == {0}
        assert {s.mask_path for s in samples} == {None}
        assert {s.meta["lighting"] for s in samples} == {"regular"}


def test_test_public_labels_good_and_bad(dataset):
    samples = list(dataset.samples("test_public", "vial"))
    good = [s for s in samples if s.label == 0]
    bad = [s for s in samples if s.label == 1]
    assert len(good) == 2 * len(CONDITIONS)
    assert len(bad) == 2 * len(CONDITIONS)
    assert all(s.mask_path is None for s in good)
    assert all(s.mask_path is not None for s in bad)


def test_test_public_covers_every_lighting_condition(dataset):
    samples = list(dataset.samples("test_public", "vial"))
    assert {s.meta["lighting"] for s in samples} == set(CONDITIONS)


def test_bad_samples_pair_with_the_right_mask_file(dataset):
    for s in dataset.samples("test_public", "vial"):
        if s.label == 1:
            assert s.mask_path.name == f"{s.image_path.stem}_mask.png"
            assert s.mask_path.is_file()


def test_a_missing_mask_is_an_error_not_a_silent_none(tmp_path):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError):
        list(MVTecAD2(tmp_path).samples("test_public", "vial"))


def test_private_splits_are_unlabelled_and_flat(dataset):
    for split, cond in (("test_private", "regular"), ("test_private_mixed", "mixed")):
        samples = list(dataset.samples(split, "vial"))
        assert samples, split
        assert {s.label for s in samples} == {LABEL_UNKNOWN}
        assert {s.mask_path for s in samples} == {None}
        assert {s.meta["lighting"] for s in samples} == {cond}


def test_samples_without_a_category_span_every_category(dataset):
    cats = {s.category for s in dataset.samples("test_public")}
    assert cats == {"can", "vial"}


def test_load_image_returns_three_channel_uint8_from_grayscale(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 0)
    img = dataset.load_image(sample)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    # The grayscale channel is replicated, not dropped or rescaled.
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()
    assert img[0, 0, 0] == 10


def test_load_image_passes_rgb_through_unchanged(tmp_path):
    """Vial is grayscale, but that is verified for Vial only — RGB must work too."""
    build_category(tmp_path, "rgbcat", mode="RGB")
    ds = MVTecAD2(tmp_path)
    sample = next(s for s in ds.samples("test_public", "rgbcat") if s.label == 0)
    img = ds.load_image(sample)
    assert img.shape[2] == 3 and img[0, 0, 0] == 10


def test_load_mask_is_binary_and_matches_the_image_shape(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 1)
    mask = dataset.load_mask(sample)
    img = dataset.load_image(sample)
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[0, 0] == 1 and mask[-1, -1] == 0


def test_load_mask_is_none_when_there_is_no_ground_truth(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 0)
    assert dataset.load_mask(sample) is None
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_mvtec_ad2.py -q`
Expected: FAIL — `ImportError: cannot import name 'MVTecAD2' from 'vlmab.datasets.mvtec_ad2'`

If it fails with `ModuleNotFoundError: No module named 'PIL'`, run
`/tmp/vlmab-venv/bin/pip install pillow` first — Pillow is new to this plan.

- [ ] **Step 5: Implement the loader**

Replace `src/vlmab/datasets/mvtec_ad2.py` entirely with:

```python
"""MVTec AD 2 loader.

Layout verified by reading the real archive, not the dataset paper (docs/datasets-access.md):

    <root>/<category>/
      train/good/            defect-free, regular lighting
      validation/good/       defect-free, regular lighting
      test_public/
        good/                normal, all 7 lighting conditions
        bad/                 anomalous, all 7 lighting conditions
        ground_truth/bad/    one `{stem}_mask.png` per bad image
      test_private/          flat, labels withheld
      test_private_mixed/    flat, labels withheld

The lighting condition lives in the filename (`000_shift_1.png`), so it is parsed onto
`Sample.meta["lighting"]` — that is what lets results be broken down by lighting condition,
which is the whole reason this dataset exists.

`test_public/good/` and `test_public/bad/` reuse the same stems, so a consumer must never
identify a sample by `image_path.stem` alone.
"""
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from vlmab.datasets.base import AnomalyDataset, Sample

#: Label for samples whose ground truth the dataset withholds (the private test splits).
#: Deliberately not 0: treating "unknown" as "normal" would silently corrupt every accuracy
#: metric, so downstream code is expected to reject this value rather than average over it.
LABEL_UNKNOWN = -1


def lighting_condition(image_path: Path) -> str:
    """Lighting condition encoded in a filename like `000_shift_1.png` -> `shift_1`."""
    _, _, condition = image_path.stem.partition("_")
    if not condition:
        raise ValueError(
            f"{image_path.name} has no lighting-condition suffix; expected "
            "'{index}_{condition}.png' as in '000_regular.png'"
        )
    return condition


class MVTecAD2(AnomalyDataset):
    name = "mvtec_ad2"

    LABELLED_SPLITS = ("train", "validation", "test_public")
    UNLABELLED_SPLITS = ("test_private", "test_private_mixed")
    SPLITS = LABELLED_SPLITS + UNLABELLED_SPLITS

    def __init__(self, root: Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"dataset root {self.root} does not exist")

    def categories(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir())

    def samples(self, split: str, category: str | None = None) -> Iterator[Sample]:
        if split not in self.SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {self.SPLITS}")
        for cat in [category] if category is not None else self.categories():
            base = self.root / cat / split
            if split in ("train", "validation"):
                yield from self._from_dir(base / "good", cat, split, label=0)
            elif split == "test_public":
                yield from self._from_dir(base / "good", cat, split, label=0)
                yield from self._from_dir(
                    base / "bad", cat, split, label=1,
                    ground_truth=base / "ground_truth" / "bad",
                )
            else:
                yield from self._from_dir(base, cat, split, label=LABEL_UNKNOWN)

    def _from_dir(
        self,
        directory: Path,
        category: str,
        split: str,
        label: int,
        ground_truth: Path | None = None,
    ) -> Iterator[Sample]:
        for image_path in sorted(directory.glob("*.png")):
            mask_path = None
            if ground_truth is not None:
                mask_path = ground_truth / f"{image_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"no ground-truth mask for {image_path}: expected {mask_path}. "
                        "An anomalous image without a mask would silently score as having "
                        "nothing to find."
                    )
            yield Sample(
                image_path=image_path,
                label=label,
                category=category,
                mask_path=mask_path,
                meta={"lighting": lighting_condition(image_path), "split": split},
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        """HxWx3 uint8.

        `convert("RGB")` replicates the single channel of a grayscale image and passes an
        already-RGB image through untouched, so this handles both without the loader having to
        know which categories are which — Vial is grayscale, the rest are unverified
        (protocol §3).
        """
        from PIL import Image

        with Image.open(sample.image_path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def load_mask(self, sample: Sample) -> np.ndarray | None:
        """HxW uint8 in {0, 1}, or None when the sample has no ground truth."""
        if sample.mask_path is None:
            return None
        from PIL import Image

        with Image.open(sample.mask_path) as im:
            return (np.asarray(im.convert("L")) > 0).astype(np.uint8)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_mvtec_ad2.py -q`
Expected: PASS (16 passed)

- [ ] **Step 7: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS — 75 previous tests plus the 17 new ones. Nothing may fail or error.

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/ci.yml pyproject.toml src/vlmab/datasets/mvtec_ad2.py tests/mvtec_tree.py tests/test_mvtec_ad2.py
git commit -m "feat: MVTec AD 2 loader with lighting-condition metadata"
```

---

### Task 3: Layout verification CLI

A malformed or partial download must fail here, loudly, rather than halfway through a twelve-hour
grid run.

**Files:**
- Modify: `scripts/prepare_data.py`
- Create: `tests/test_prepare_data.py`

**Interfaces:**
- Consumes: `MVTecAD2`, `lighting_condition`, `LABEL_UNKNOWN` (Task 2).
- Produces:
  - `check_category(root: Path, category: str) -> list[str]` — returns a list of human-readable
    problems, empty when the category is well-formed
  - `summarize_category(root: Path, category: str) -> dict[str, dict[str, int]]` — per split, a
    mapping of lighting condition to file count
  - `main(argv: Sequence[str] | None = None) -> int` — CLI entry point, returns an exit code

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_data.py`:

```python
import numpy as np
import pytest
from PIL import Image

from mvtec_tree import CONDITIONS, build_category
from prepare_data import check_category, main, summarize_category


def test_a_well_formed_category_has_no_problems(tmp_path):
    build_category(tmp_path, "vial")
    assert check_category(tmp_path, "vial") == []


def test_a_missing_split_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    for p in sorted((cat / "validation" / "good").glob("*.png")):
        p.unlink()
    (cat / "validation" / "good").rmdir()
    (cat / "validation").rmdir()
    problems = check_category(tmp_path, "vial")
    assert any("validation" in p for p in problems)


def test_an_orphan_mask_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    orphan = cat / "test_public" / "ground_truth" / "bad" / "999_regular_mask.png"
    Image.fromarray(np.zeros((6, 8), dtype=np.uint8), mode="L").save(orphan)
    problems = check_category(tmp_path, "vial")
    assert any("999_regular" in p for p in problems)


def test_a_missing_mask_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    problems = check_category(tmp_path, "vial")
    assert any("mask" in p.lower() for p in problems)


def test_an_empty_split_directory_is_reported(tmp_path):
    cat = build_category(tmp_path, "vial")
    for p in (cat / "test_private").glob("*.png"):
        p.unlink()
    problems = check_category(tmp_path, "vial")
    assert any("test_private" in p and "empty" in p.lower() for p in problems)


def test_summary_counts_files_per_split_and_condition(tmp_path):
    build_category(tmp_path, "vial", n_good=2, n_bad=3)
    summary = summarize_category(tmp_path, "vial")
    assert summary["train"] == {"regular": 2}
    assert summary["validation"] == {"regular": 2}
    assert summary["test_public"]["regular"] == 5  # 2 good + 3 bad
    assert set(summary["test_public"]) == set(CONDITIONS)
    assert summary["test_private"] == {"regular": 3}
    assert summary["test_private_mixed"] == {"mixed": 3}


def test_main_returns_zero_for_a_good_tree(tmp_path, capsys):
    build_category(tmp_path, "vial")
    assert main(["--root", str(tmp_path)]) == 0
    assert "vial" in capsys.readouterr().out


def test_main_returns_nonzero_and_names_the_problem(tmp_path, capsys):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    assert main(["--root", str(tmp_path)]) == 1
    assert "mask" in capsys.readouterr().out.lower()


def test_main_can_check_one_category(tmp_path):
    build_category(tmp_path, "vial")
    build_category(tmp_path, "can")
    assert main(["--root", str(tmp_path), "--category", "vial"]) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_prepare_data.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_category' from 'prepare_data'`

If it fails with `ModuleNotFoundError: No module named 'prepare_data'`, the `pythonpath`
setting from Task 2 Step 1 is missing from `pyproject.toml` — add it before continuing.

- [ ] **Step 3: Implement**

Replace `scripts/prepare_data.py` entirely with:

```python
#!/usr/bin/env python
"""Verify an extracted MVTec AD 2 tree before anything expensive runs on it.

A twelve-hour grid that dies on a missing mask has wasted a session. This fails in seconds
instead, and names the file.

Usage:
    python scripts/prepare_data.py --root data/mvtec_ad2
    python scripts/prepare_data.py --root data/mvtec_ad2 --category vial
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from vlmab.datasets.mvtec_ad2 import MVTecAD2, lighting_condition

_EXPECTED_DIRS = {
    "train": ("good",),
    "validation": ("good",),
    "test_public": ("good", "bad"),
    "test_private": (),
    "test_private_mixed": (),
}


def _image_dirs(category_root: Path, split: str) -> list[Path]:
    subdirs = _EXPECTED_DIRS[split]
    return [category_root / split / s for s in subdirs] if subdirs else [category_root / split]


def check_category(root: Path, category: str) -> list[str]:
    """Problems found in one category. Empty list means the layout is sound."""
    category_root = Path(root) / category
    problems: list[str] = []

    for split in MVTecAD2.SPLITS:
        for directory in _image_dirs(category_root, split):
            if not directory.is_dir():
                problems.append(f"{category}/{split}: missing directory {directory}")
                continue
            images = sorted(directory.glob("*.png"))
            if not images:
                problems.append(f"{category}/{split}: {directory} is empty")
            for image in images:
                try:
                    lighting_condition(image)
                except ValueError as exc:
                    problems.append(f"{category}/{split}: {exc}")

    ground_truth = category_root / "test_public" / "ground_truth" / "bad"
    bad = category_root / "test_public" / "bad"
    if not ground_truth.is_dir():
        problems.append(f"{category}/test_public: missing directory {ground_truth}")
    elif bad.is_dir():
        bad_stems = {p.stem for p in bad.glob("*.png")}
        mask_stems = {p.stem[: -len("_mask")] for p in ground_truth.glob("*_mask.png")}
        for stem in sorted(bad_stems - mask_stems):
            problems.append(f"{category}/test_public: no mask for bad image {stem}.png")
        for stem in sorted(mask_stems - bad_stems):
            problems.append(f"{category}/test_public: orphan mask {stem}_mask.png has no image")

    return problems


def summarize_category(root: Path, category: str) -> dict[str, dict[str, int]]:
    """Per split, how many images of each lighting condition."""
    category_root = Path(root) / category
    summary: dict[str, dict[str, int]] = {}
    for split in MVTecAD2.SPLITS:
        counts: Counter[str] = Counter()
        for directory in _image_dirs(category_root, split):
            if not directory.is_dir():
                continue
            for image in directory.glob("*.png"):
                try:
                    counts[lighting_condition(image)] += 1
                except ValueError:
                    counts["<unparseable>"] += 1
        summary[split] = dict(sorted(counts.items()))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="dataset root directory")
    parser.add_argument("--category", help="check one category instead of all of them")
    args = parser.parse_args(argv)

    categories = [args.category] if args.category else MVTecAD2(args.root).categories()
    if not categories:
        print(f"no categories found under {args.root}")
        return 1

    failed = False
    for category in categories:
        problems = check_category(args.root, category)
        summary = summarize_category(args.root, category)
        total = sum(sum(c.values()) for c in summary.values())
        print(f"\n{category}: {total} images")
        for split, counts in summary.items():
            rendered = ", ".join(f"{k}={v}" for k, v in counts.items()) or "-"
            print(f"  {split:<20} {rendered}")
        for problem in problems:
            print(f"  PROBLEM: {problem}")
            failed = True

    print("\nFAILED" if failed else "\nOK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_prepare_data.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Verify against the real Vial download**

This is the point of the task, so do it for real. Extract the archive somewhere outside the
repository and run the CLI against it:

```bash
mkdir -p /tmp/mvtec_ad2 && tar -xzf ~/Downloads/vial.tar.gz -C /tmp/mvtec_ad2
/tmp/vlmab-venv/bin/python scripts/prepare_data.py --root /tmp/mvtec_ad2 --category vial
```

Expected: exit code 0, ending in `OK`, and counts matching the verified figures — train
`regular=291`, validation `regular=41`, test_public 140 images across 7 conditions (20 per
condition), test_private `regular=276`, test_private_mixed `mixed=276`.

Record the actual output in your report. **If the counts differ from these, stop and report it**
— it would mean the layout documentation is wrong, which matters more than this task.

- [ ] **Step 6: Commit**

```bash
git add scripts/prepare_data.py tests/test_prepare_data.py
git commit -m "feat: prepare_data verifies MVTec AD 2 layout and reports per-condition counts"
```

---

### Task 4: Carry the mask path into result rows

Aggregation needs to know where each sample's ground truth is. The runner currently drops it, so
a shard cannot be turned into pixel metrics without re-walking the dataset.

**Files:**
- Modify: `src/vlmab/eval/runner.py`
- Modify: `tests/test_runner.py`

**Interfaces:**
- Consumes: `run_evaluation` and the `Sample` contract as they already exist.
- Produces: result rows gain a `mask_path` column — `str` when the sample has ground truth,
  `None` when it does not.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_runner.py`:

```python
def test_runner_records_the_mask_path_when_the_sample_has_one(tmp_path, counting_method):
    """Pixel metrics are computed from the shard later, so the shard must say where the
    ground truth is; re-walking the dataset to find it would be a second source of truth."""
    class _WithMasks(AnomalyDataset):
        name = "masked"

        def categories(self):
            return ["alpha"]

        def samples(self, split, category=None):
            yield Sample(image_path=Path("/fake/alpha/bad/000_regular.png"), label=1,
                         category="alpha", mask_path=Path("/fake/gt/000_regular_mask.png"),
                         meta={})
            yield Sample(image_path=Path("/fake/alpha/good/000_regular.png"), label=0,
                         category="alpha", mask_path=None, meta={})

        def load_image(self, sample):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    store = ResultStore(tmp_path)
    run_evaluation(_WithMasks(), counting_method, store, {"seed": 0})
    df = pd.read_parquet(store.path_for("masked", "counting", "alpha"))
    assert list(df["mask_path"]) == ["/fake/gt/000_regular_mask.png", None]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -k mask_path -q`
Expected: FAIL — `KeyError: 'mask_path'`

- [ ] **Step 3: Implement**

In `src/vlmab/eval/runner.py`, in the row construction inside the sample loop, change:

```python
            row: dict[str, Any] = {
                "image_path": str(sample.image_path),
                "label": int(sample.label),
                "image_score": float(prediction.image_score),
                "split": split,
            }
```

to:

```python
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

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_runner.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/eval/runner.py tests/test_runner.py
git commit -m "feat: record each sample's mask path in result rows"
```

---

### Task 5: Metric aggregation

**Files:**
- Create: `src/vlmab/eval/aggregate.py`
- Create: `tests/test_aggregate.py`

**Interfaces:**
- Consumes: `i_auroc`, `i_ap`, `i_f1max` from `vlmab.metrics.image_level`; `p_auroc`, `seg_f1max`,
  `au_pro` from `vlmab.metrics.pixel_level`; `LABEL_UNKNOWN` from `vlmab.datasets.mvtec_ad2`;
  the `mask_path` and `map_path` columns from Task 4 and the runner.
- Produces:
  - `image_metrics(df: pd.DataFrame) -> dict[str, float]` with keys `i_auroc`, `i_ap`, `i_f1max`, `n`
  - `pixel_metrics(df: pd.DataFrame, max_pixels: int = 400_000_000) -> dict[str, float]` with keys
    `p_auroc`, `seg_f1max`, `au_pro_030`, `au_pro_005`, `n`
  - `aggregate(df: pd.DataFrame, by: str | None = None, max_pixels: int = 400_000_000) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_aggregate.py`:

```python
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.eval.aggregate import aggregate, image_metrics, pixel_metrics


def _shard(tmp_path, n=4, separable=True, with_pixels=True):
    """A shard whose scores separate perfectly, with matching masks and maps on disk."""
    rows = []
    for i in range(n):
        label = i % 2
        row = {
            "image_path": f"/fake/{i:03d}_regular.png",
            "label": label,
            "image_score": 0.9 if (label and separable) else 0.1,
            "split": "test_public",
            "meta_lighting": "regular" if i < n // 2 else "overexposed",
            "mask_path": None,
            "map_path": None,
        }
        if with_pixels:
            amap = np.zeros((6, 8), dtype=np.float16)
            mask = np.zeros((6, 8), dtype=np.uint8)
            if label:
                amap[:2, :2] = 1.0
                mask[:2, :2] = 255
                mask_path = tmp_path / f"{i:03d}_mask.png"
                Image.fromarray(mask, mode="L").save(mask_path)
                row["mask_path"] = str(mask_path)
            map_path = tmp_path / f"{i:03d}.npy"
            np.save(map_path, amap)
            row["map_path"] = str(map_path)
        rows.append(row)
    return pd.DataFrame(rows)


def test_image_metrics_on_perfect_separation(tmp_path):
    out = image_metrics(_shard(tmp_path))
    assert out["i_auroc"] == 1.0
    assert out["i_ap"] == 1.0
    assert out["i_f1max"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_image_metrics_refuses_unlabelled_rows(tmp_path):
    """Withheld labels must not be silently averaged in as normal."""
    df = _shard(tmp_path)
    df.loc[0, "label"] = LABEL_UNKNOWN
    with pytest.raises(ValueError, match="unlabelled"):
        image_metrics(df)


def test_image_metrics_refuses_a_single_class(tmp_path):
    df = _shard(tmp_path)
    df["label"] = 0
    with pytest.raises(ValueError):
        image_metrics(df)


def test_pixel_metrics_on_perfect_maps(tmp_path):
    out = pixel_metrics(_shard(tmp_path))
    assert out["p_auroc"] == pytest.approx(1.0)
    assert out["seg_f1max"] == pytest.approx(1.0)
    assert out["au_pro_030"] == pytest.approx(1.0)
    assert out["au_pro_005"] == pytest.approx(1.0)
    assert out["n"] == 4


def test_pixel_metrics_treats_a_missing_mask_as_all_normal(tmp_path):
    """A `good` image has no mask file; that means no anomalous pixels, not missing data."""
    df = _shard(tmp_path)
    assert df["mask_path"].isna().any()
    out = pixel_metrics(df)
    assert out["n"] == 4


def test_pixel_metrics_skips_rows_without_a_map(tmp_path):
    df = _shard(tmp_path)
    df.loc[0, "map_path"] = None
    out = pixel_metrics(df)
    assert out["n"] == 3


def test_pixel_metrics_returns_nothing_when_no_maps_were_saved(tmp_path):
    df = _shard(tmp_path, with_pixels=False)
    assert pixel_metrics(df) == {"n": 0}


def test_pixel_metrics_refuses_to_exceed_the_pixel_budget(tmp_path):
    """Colab has ~12.7 GB of RAM. Failing with a number beats being OOM-killed silently."""
    df = _shard(tmp_path)
    with pytest.raises(ValueError, match="max_pixels"):
        pixel_metrics(df, max_pixels=10)


def test_aggregate_returns_one_row_with_both_metric_families(tmp_path):
    out = aggregate(_shard(tmp_path))
    assert len(out) == 1
    assert {"i_auroc", "p_auroc", "au_pro_030", "n"} <= set(out.columns)


def test_aggregate_groups_by_lighting_condition(tmp_path):
    """The lighting breakdown is the reason this dataset exists."""
    out = aggregate(_shard(tmp_path), by="meta_lighting")
    assert list(out["meta_lighting"]) == ["overexposed", "regular"]
    assert len(out) == 2


def test_aggregate_rejects_a_missing_group_column(tmp_path):
    with pytest.raises(KeyError):
        aggregate(_shard(tmp_path), by="meta_nope")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_aggregate.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'vlmab.eval.aggregate'`

- [ ] **Step 3: Implement**

Create `src/vlmab/eval/aggregate.py`:

```python
"""Turn result shards into metrics.

Kept separate from the runner on purpose: the runner streams predictions to disk one category
at a time and never holds a grid's worth of anomaly maps in memory. Aggregation is where maps
are read back, so it is also where the memory ceiling has to be enforced — Colab's free tier has
roughly 12.7 GB of host RAM, and a category of 2.66 MP images adds up fast.
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN
from vlmab.metrics.image_level import i_ap, i_auroc, i_f1max
from vlmab.metrics.pixel_level import au_pro, p_auroc, seg_f1max


def image_metrics(df: pd.DataFrame) -> dict[str, float]:
    """I-AUROC, I-AP and I-F1max over a shard's image scores."""
    labels = df["label"].to_numpy()
    if (labels == LABEL_UNKNOWN).any():
        raise ValueError(
            f"{int((labels == LABEL_UNKNOWN).sum())} unlabelled rows (label "
            f"{LABEL_UNKNOWN}): these come from the private splits, whose ground truth is "
            "withheld. Scoring them would mean treating unknown as normal."
        )
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "image metrics need both normal and anomalous samples; this group has only "
            f"label {int(labels[0])}"
        )
    scores = df["image_score"].to_numpy(dtype=np.float64)
    return {
        "i_auroc": i_auroc(labels, scores),
        "i_ap": i_ap(labels, scores),
        "i_f1max": i_f1max(labels, scores),
        "n": int(len(df)),
    }


def _load_pair(row: Any) -> tuple[np.ndarray, np.ndarray]:
    """One (mask, anomaly map) pair. A row with no mask file has no anomalous pixels."""
    amap = np.load(row.map_path).astype(np.float32)
    if row.mask_path is None or (isinstance(row.mask_path, float) and np.isnan(row.mask_path)):
        return np.zeros(amap.shape, dtype=np.uint8), amap
    from PIL import Image

    with Image.open(row.mask_path) as im:
        mask = (np.asarray(im.convert("L")) > 0).astype(np.uint8)
    if mask.shape != amap.shape:
        raise ValueError(
            f"mask {row.mask_path} is {mask.shape} but its anomaly map is {amap.shape}"
        )
    return mask, amap


def pixel_metrics(df: pd.DataFrame, max_pixels: int = 400_000_000) -> dict[str, float]:
    """P-AUROC, SegF1max and AU-PRO at both FPR limits, over rows that saved a map.

    `max_pixels` is a guard, not a tuning knob: exceeding it raises rather than letting the
    session get OOM-killed with no diagnostic. Raise it deliberately if the machine can take it.
    """
    if "map_path" not in df.columns:
        return {"n": 0}
    with_maps = df[df["map_path"].notna()]
    if with_maps.empty:
        return {"n": 0}

    total = 0
    for path in with_maps["map_path"]:
        shape = np.load(path, mmap_mode="r").shape
        total += int(np.prod(shape))
    if total > max_pixels:
        raise ValueError(
            f"{total:,} pixels exceeds max_pixels={max_pixels:,}. Aggregate a smaller group "
            "(e.g. one lighting condition at a time) or raise the budget if this machine "
            "genuinely has the memory."
        )

    masks, amaps = [], []
    for row in with_maps.itertuples():
        mask, amap = _load_pair(row)
        masks.append(mask)
        amaps.append(amap)

    return {
        "p_auroc": p_auroc(masks, amaps),
        "seg_f1max": seg_f1max(masks, amaps),
        "au_pro_030": au_pro(masks, amaps, fpr_limit=0.3),
        "au_pro_005": au_pro(masks, amaps, fpr_limit=0.05),
        "n": int(len(with_maps)),
    }


def aggregate(
    df: pd.DataFrame,
    by: str | None = None,
    max_pixels: int = 400_000_000,
) -> pd.DataFrame:
    """One metrics row per group, or a single row when `by` is None.

    Grouping by `meta_lighting` is the headline use: MVTec AD 2 exists to measure robustness to
    lighting shift, and that question is only answerable per condition.
    """
    if by is not None and by not in df.columns:
        raise KeyError(f"no column {by!r} to group by; columns are {sorted(df.columns)}")

    groups = [(None, df)] if by is None else sorted(df.groupby(by), key=lambda kv: kv[0])

    records = []
    for key, group in groups:
        record: dict[str, Any] = {} if key is None else {by: key}
        record.update(image_metrics(group))
        pixels = pixel_metrics(group, max_pixels=max_pixels)
        record.update({k: v for k, v in pixels.items() if k != "n"})
        record["n_pixel_rows"] = pixels["n"]
        records.append(record)
    return pd.DataFrame(records)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/tmp/vlmab-venv/bin/python -m pytest tests/test_aggregate.py -q`
Expected: PASS (11 passed)

- [ ] **Step 5: Run the full suite**

Run: `/tmp/vlmab-venv/bin/python -m pytest -q`
Expected: PASS — everything from before plus the new tests. Nothing may fail or error.

- [ ] **Step 6: Commit**

```bash
git add src/vlmab/eval/aggregate.py tests/test_aggregate.py
git commit -m "feat: aggregate shards into image and pixel metrics, groupable by lighting"
```

---

## Out of scope for this plan, and why

- **Method adapters** (WinCLIP, AnomalyCLIP, AdaCLIP, SAA+, Qwen2.5-VL-3B, PatchCore). Each needs
  torch, GPU weights and its own pinned upstream commit; together they are a plan of their own.
  They are also the only remaining piece before real numbers exist, so they should be their own
  reviewed unit.
- **The Colab notebook.** It needs the per-category download URLs, which sit behind the MVTec
  account, and it is only meaningful once at least one adapter runs. Writing it now would mean
  inventing URLs.
- **The evaluation-server submission writer** and the official metric-name mapping. Both need the
  server's submission documentation, which is still behind the pending registration
  (`docs/datasets-access.md`).
- **Wiring the efficiency module into the runner.** `summarize_latencies`, `count_parameters` and
  `peak_vram_mb` exist and are tested but nothing calls them; per-sample timing belongs with the
  first real adapter, where there is something whose latency is worth measuring.
