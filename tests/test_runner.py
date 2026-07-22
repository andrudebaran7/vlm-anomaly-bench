from pathlib import Path

import numpy as np
import pandas as pd

from vlmab.datasets.base import AnomalyDataset, Sample
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore


class _LitDataset(AnomalyDataset):
    """Local-only fixture: like FakeDataset, but meta carries a lighting condition
    alongside split, so the meta_ prefixing and split exclusion are both exercised."""

    name = "lit"

    def categories(self):
        return ["alpha"]

    def samples(self, split, category=None):
        for i in range(2):
            yield Sample(
                image_path=Path(f"/lit/alpha/{i}.png"),
                label=i % 2,
                category="alpha",
                meta={"split": split, "lighting": "low"},
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def load_mask(self, sample: Sample):
        return np.zeros((8, 8), dtype=np.uint8)


def test_runner_writes_one_shard_per_category(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    written = run_evaluation(fake_dataset, counting_method, store, {"seed": 0})
    assert len(written) == 2
    assert store.is_done("fake", "counting", "alpha")
    assert store.is_done("fake", "counting", "beta")


def test_runner_records_every_sample(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0})
    df = store.load_all()
    assert len(df) == 6
    assert set(df.columns) >= {"image_path", "label", "image_score", "category", "method", "seed"}


def test_runner_prepares_the_method_once(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0})
    assert counting_method.prepare_calls == 1


def test_runner_skips_completed_categories_on_resume(tmp_path, fake_dataset, counting_method):
    """The Colab disconnection case: a second run must not recompute finished work."""
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
    assert counting_method.seen == ["alpha"] * 3

    resumed = run_evaluation(fake_dataset, counting_method, store, {"seed": 0})
    assert counting_method.seen == ["alpha"] * 3 + ["beta"] * 3
    assert [p.name for p in resumed] == ["fake__counting__beta.parquet"]


def test_runner_does_not_prepare_when_everything_is_done(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0})
    counting_method.prepare_calls = 0
    assert run_evaluation(fake_dataset, counting_method, store, {"seed": 0}) == []
    assert counting_method.prepare_calls == 0


def test_runner_saves_maps_when_asked(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    run_evaluation(
        fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"], maps_dir=maps
    )
    saved = sorted((maps / "fake__counting__alpha").glob("*.npy"))
    assert len(saved) == 3
    assert np.load(saved[0]).dtype == np.float16

    df = pd.read_parquet(store.path_for("fake", "counting", "alpha"))
    assert df["map_path"].notna().all()


def test_runner_omits_map_path_when_not_saving(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha"))
    assert "map_path" not in df.columns


def test_runner_prefixes_non_split_meta_and_excludes_split(tmp_path, counting_method):
    """meta_ columns come from Sample.meta, excluding split (already its own column) —
    e.g. a lighting condition (MVTec AD 2) that later lets results be sliced by lighting."""
    store = ResultStore(tmp_path)
    run_evaluation(_LitDataset(), counting_method, store, {"seed": 0})
    df = pd.read_parquet(store.path_for("lit", "counting", "alpha"))
    assert (df["meta_lighting"] == "low").all()
    assert "meta_split" not in df.columns
