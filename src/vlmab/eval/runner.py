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
was never written). The resumed run overwrites them as it recomputes the category. Size
categories against your
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
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from vlmab.datasets.base import AnomalyDataset
from vlmab.eval.provenance import config_hash
from vlmab.eval.store import ResultStore
from vlmab.methods.base import AnomalyMethod

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def run_id(meta: Mapping[str, Any]) -> str:
    """Short identifier for the run that owns a set of anomaly maps.

    Map directories are keyed on this in addition to dataset/method/category, because
    those three do NOT identify a run. Two runs of the same method with different
    configs write the same shard path, so they must use different store roots — and
    then they would share one map directory, where each would silently overwrite the
    other's maps while the first run's completed shard still pointed at them.

    `run_meta()` supplies `config_hash`, which is exactly the right key. When `meta`
    carries no `config_hash` (bare dicts in tests, or a caller assembling meta by
    hand) we hash the whole `meta` mapping instead: any two runs that differ in
    anything recorded about them get different directories, and two runs that are
    identical in every recorded respect are genuinely interchangeable. What we must
    not do is fall back to a constant, which would reintroduce the collision.
    """
    existing = meta.get("config_hash")
    return str(existing) if existing else config_hash(meta)


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
    # MVTec AD 2 has no split called "test"; the previous default raised out of the loader
    # for every caller that took it. test_public is the split with pixel ground truth, i.e.
    # the one every locally computed metric in this study is defined on.
    split: str = "test_public",
    fit_split: str = "train",
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
        if not method.zero_shot:
            # Full-shot anchor: build this category's memory bank from its defect-free train
            # images before scoring it (protocol §3). Passed as a lazy generator so the backend
            # streams them rather than holding a category of 2.66 MP images in memory at once.
            train_images = (
                dataset.load_image(s) for s in dataset.samples(fit_split, category)
            )
            method.fit(train_images, category)
        # Accumulates in memory for this whole category; nothing here reaches disk (and
        # nothing is resumable) until the sample loop below completes and store.write() runs.
        rows: list[dict[str, Any]] = []
        category_maps = None
        if maps_dir is not None:
            category_maps = (
                Path(maps_dir)
                / f"{dataset.name}__{method.name}__{run_id(meta)}__{category}"
            )
        # Created lazily on first write: a category that yields no samples must not
        # touch the filesystem before store.write() rejects it as empty.
        maps_dir_ready = False
        # Paths this run has written for this category. Only a repeat within one run is
        # a genuine filename collision (two samples deriving one name, which would make
        # their rows point at the same map). A file left by an earlier crashed run is
        # not: no shard references it, so overwriting it is how resume works. Deleting
        # such files up front would be unsafe — another run sharing `maps_dir` could
        # own them.
        written_maps: set[Path] = set()

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
            row.update({f"meta_{k}": v for k, v in sample.meta.items() if k != "split"})

            if category_maps is not None:
                map_path = category_maps / map_filename(sample.image_path)
                if map_path in written_maps:
                    raise FileExistsError(
                        f"anomaly map {map_path} was already written by this run for a "
                        f"different sample than {sample.image_path}; refusing to "
                        "overwrite — two samples derived the same map filename, so "
                        "their result rows would point at the same map"
                    )
                anomaly_map = np.asarray(prediction.anomaly_map, dtype=np.float32)
                # float16 storage is a silent-corruption trap: a finite float32 value
                # above float16's max (65504) casts to inf with no error (verified:
                # np.array([1e6], np.float32).astype(np.float16) is inf), and nothing
                # downstream re-checks finiteness before computing pixel metrics from
                # the saved array. Protocol v0.2.6 relaxed the map-scale contract from
                # [0,1] to "finite on the method's own scale", so a raw map exceeding
                # float16's range is now something a method can in principle produce.
                # Fail loud instead of clipping — clipping would hide a real problem
                # (a method whose scores don't fit float16 needs to be flagged, not
                # quietly truncated). NaN/inf coming IN from the method is already a
                # method bug; it is caught by the same guard rather than given a
                # separate path, since either way the map must not reach disk.
                float16_max = float(np.finfo(np.float16).max)
                max_abs = float(np.abs(anomaly_map).max()) if anomaly_map.size else 0.0
                if not np.isfinite(max_abs) or max_abs > float16_max:
                    raise ValueError(
                        f"anomaly map for {sample.image_path} has max abs value "
                        f"{max_abs} which does not survive float16 storage "
                        f"(finite range is +/-{float16_max}); refusing "
                        "to silently write an inf map"
                    )
                if not maps_dir_ready:
                    category_maps.mkdir(parents=True, exist_ok=True)
                    maps_dir_ready = True
                np.save(map_path, anomaly_map.astype(np.float16))
                written_maps.add(map_path)
                row["map_path"] = str(map_path)

            rows.append(row)

        written.append(store.write(dataset.name, method.name, category, rows, meta))

    return written
