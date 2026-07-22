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
was never written), and this runner does not clean them up. Size categories against your
session budget (e.g. Colab's ~12-hour cap) with this in mind: a category that alone takes
longer than the remaining session time will repeatedly fail to checkpoint.

The method is only prepared (weights loaded) if there is actually something left to compute
— resuming a finished run must not pay for a model load.

Pixel metrics are not computed here. The runner persists scores and, optionally, anomaly
maps; aggregation into metrics is a separate step so a long grid never holds every map in
memory at once.
"""
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from vlmab.datasets.base import AnomalyDataset
from vlmab.eval.store import ResultStore
from vlmab.methods.base import AnomalyMethod


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
                map_path = category_maps / f"{sample.image_path.stem}.npy"
                np.save(map_path, prediction.anomaly_map.astype(np.float16))
                row["map_path"] = str(map_path)

            rows.append(row)

        written.append(store.write(dataset.name, method.name, category, rows, meta))

    return written
