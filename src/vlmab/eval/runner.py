"""Evaluation runner: (dataset x method) -> per-sample parquet shards in results/.

Built for a platform that disconnects: checkpointing is per CATEGORY, not per sample. Rows
for a category are only accumulated in memory as samples are predicted; the shard for that
category is written to disk once, only after its whole sample loop finishes. A restart skips
every category whose shard already exists on disk.

This means the granularity a user is protected at is a full category, not a sample: if the
process dies partway through a category, every row computed so far for that category is
lost, including samples already predicted — the resumed run recomputes the whole category
from scratch. Any anomaly maps already written to `maps_dir` for that partial category are
orphaned: they are never referenced by any result row (the shard that would reference them
was never written). The resumed run clears that category's map directory before
recomputing it, so orphans do not accumulate across restarts. Size categories against your
session budget (e.g. Colab's ~12-hour cap) with this in mind: a category that alone takes
longer than the remaining session time will repeatedly fail to checkpoint.

The method is only prepared (weights loaded) if there is actually something left to compute
— resuming a finished run must not pay for a model load.

Pixel metrics are not computed here. The runner persists scores and, optionally, anomaly
maps; aggregation into metrics is a separate step so a long grid never holds every map in
memory at once.
"""
import hashlib
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from vlmab.datasets.base import AnomalyDataset
from vlmab.eval.store import ResultStore
from vlmab.methods.base import AnomalyMethod

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def map_filename(image_path: Path) -> str:
    """Filename for a sample's anomaly map: readable tail + hash of the full path.

    `image_path.stem` alone is NOT unique. MVTec-style layouts restart numbering
    inside every subfolder, so `test_public/good/000.png` and `test_public/bad/000.png`
    both have stem `000`; naming maps by stem makes the second write overwrite the
    first, after which the `good` row points at the `bad` map and every pixel metric
    silently scores normal images against defect maps.

    Two candidate schemes were considered:

    * flattened trailing path components only — readable, but still not unique in
      general (two categories can share a `<split>/<defect>/<index>` tail, and nothing
      bounds how deep the distinguishing component sits);
    * a hash of the full path only — unique, but opaque: a stray `9f3c1a2b.npy` in a
      map directory cannot be traced back to a sample by eye.

    We use both. The readable prefix is the last three path components with the
    extension dropped and non-portable characters folded to `_` (so `.../good/000.png`
    becomes `test_public__good__000`), which is what you actually want when eyeballing
    a map directory or a `map_path` column. The 10-hex-character SHA-1 of the *full*
    path string is appended to make the name unique by construction rather than by
    assumption about layout depth. Collision probability across a benchmark-sized
    corpus (~1e5 images) is ~5e-9.

    The scheme is deterministic across runs, so a resumed run re-derives the same
    names for the samples it recomputes.
    """
    tail = list(image_path.parts[-3:])
    tail[-1] = image_path.stem
    readable = "__".join(_UNSAFE.sub("_", part).strip("_") or "_" for part in tail)
    digest = hashlib.sha1(str(image_path).encode("utf-8")).hexdigest()[:10]
    return f"{readable}__{digest}.npy"


def run_evaluation(
    dataset: AnomalyDataset,
    method: AnomalyMethod,
    store: ResultStore,
    meta: Mapping[str, Any],
    categories: Iterable[str] | None = None,
    split: str = "test",
    maps_dir: Path | None = None,
    device: str = "cuda",
) -> list[Path]:
    """Evaluate `method` on `dataset`, writing one shard per category. Returns new shards."""
    wanted = list(categories) if categories is not None else dataset.categories()
    todo = [c for c in wanted if not store.is_done(dataset.name, method.name, c)]
    if not todo:
        return []

    method.prepare(device=device)

    written: list[Path] = []
    for category in todo:
        # Accumulates in memory for this whole category; nothing here reaches disk (and
        # nothing is resumable) until the sample loop below completes and store.write() runs.
        rows: list[dict[str, Any]] = []
        category_maps = None
        if maps_dir is not None:
            category_maps = Path(maps_dir) / f"{dataset.name}__{method.name}__{category}"
            category_maps.mkdir(parents=True, exist_ok=True)
            # Every category in `todo` has no shard on disk, so nothing references any
            # .npy already sitting here: it is orphaned output from a run that died
            # mid-category, and this category is about to be recomputed from scratch.
            # Clearing it keeps resume working while letting the write below treat any
            # pre-existing file as what it now can only be — a genuine name collision.
            for stale in category_maps.glob("*.npy"):
                stale.unlink()

        for sample in dataset.samples(split, category):
            image = dataset.load_image(sample)
            prediction = method.predict(image, category)

            row: dict[str, Any] = {
                "image_path": str(sample.image_path),
                "label": int(sample.label),
                "image_score": float(prediction.image_score),
                "split": split,
            }
            row.update({f"meta_{k}": v for k, v in sample.meta.items() if k != "split"})

            if category_maps is not None:
                map_path = category_maps / map_filename(sample.image_path)
                if map_path.exists():
                    raise FileExistsError(
                        f"anomaly map {map_path} already exists for {sample.image_path}; "
                        "refusing to overwrite — two samples derived the same map "
                        "filename, so their result rows would point at the same map"
                    )
                np.save(map_path, prediction.anomaly_map.astype(np.float16))
                row["map_path"] = str(map_path)

            rows.append(row)

        written.append(store.write(dataset.name, method.name, category, rows, meta))

    return written
