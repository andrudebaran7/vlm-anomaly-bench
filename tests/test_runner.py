from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlmab.datasets.base import AnomalyDataset, Sample
from vlmab.eval.runner import run_evaluation
from vlmab.eval.store import ResultStore
from vlmab.methods.base import AnomalyMethod, Prediction


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


class _MVTecLayoutDataset(AnomalyDataset):
    """Local-only fixture reproducing the real MVTec-style layout, where image
    numbering restarts inside every subfolder of a category: `test_public/good/000.png`
    and `test_public/bad/000.png` both have stem `000`.

    load_image() encodes a per-sample constant into the pixels so the map a row
    points at can be traced back to the sample it was produced from.
    """

    name = "mvt"

    # subfolder -> (label, per-sample constant)
    SUBDIRS = {"good": (0, 11.0), "bad": (1, 22.0)}

    def categories(self):
        return ["can"]

    def samples(self, split, category=None):
        for sub, (label, _) in self.SUBDIRS.items():
            yield Sample(
                image_path=Path(f"/data/mvt/can/test_public/{sub}/000.png"),
                label=label,
                category="can",
                meta={"split": split},
            )

    def load_image(self, sample: Sample) -> np.ndarray:
        sub = sample.image_path.parent.name
        return np.full((4, 4, 3), self.SUBDIRS[sub][1], dtype=np.uint8)

    def load_mask(self, sample: Sample):
        return np.zeros((4, 4), dtype=np.uint8)


class _DuplicatePathDataset(_MVTecLayoutDataset):
    """Two samples that really are the same path — the one case where a
    unique-by-construction filename scheme cannot separate them, so the runner
    must refuse to clobber rather than silently drop one sample's map."""

    name = "dup"

    def samples(self, split, category=None):
        for _ in range(2):
            yield Sample(
                image_path=Path("/data/mvt/can/test_public/good/000.png"),
                label=0,
                category="can",
                meta={"split": split},
            )


class _EchoMethod(AnomalyMethod):
    """Anomaly map filled with the constant its input image carries."""

    name = "echo"
    zero_shot = True

    def prepare(self, device="cuda"):
        pass

    def predict(self, image, category):
        value = float(image[0, 0, 0])
        return Prediction(
            image_score=value / 100.0,
            anomaly_map=np.full((4, 4), value, dtype=np.float32),
        )


def test_runner_pairs_every_row_with_its_own_map_despite_colliding_stems(tmp_path):
    """MVTec restarts numbering per subfolder, so `image_path.stem` is not unique.

    Naming maps by stem alone makes `good/000.png` and `bad/000.png` collide: the
    second write overwrites the first and the `good` row then points at the `bad`
    map. Every pixel metric downstream would score normal images against defect
    maps with no error raised.
    """
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    run_evaluation(_MVTecLayoutDataset(), _EchoMethod(), store, {"seed": 0}, maps_dir=maps)

    saved = sorted((maps / "mvt__echo__can").glob("*.npy"))
    assert len(saved) == 2, f"expected one map per sample, got {[p.name for p in saved]}"

    df = pd.read_parquet(store.path_for("mvt", "echo", "can"))
    assert df["map_path"].nunique() == 2
    for row in df.to_dict("records"):
        sub = Path(row["image_path"]).parent.name
        expected = _MVTecLayoutDataset.SUBDIRS[sub][1]
        loaded = np.load(row["map_path"])
        assert (loaded == expected).all(), (
            f"row for {row['image_path']} loaded a map belonging to another sample: "
            f"{np.unique(loaded)} != {expected}"
        )


def test_runner_refuses_to_overwrite_an_existing_map(tmp_path):
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    with pytest.raises(FileExistsError):
        run_evaluation(_DuplicatePathDataset(), _EchoMethod(), store, {"seed": 0}, maps_dir=maps)


def test_runner_clears_orphaned_maps_before_recomputing_a_category(tmp_path):
    """Resume-after-mid-category-crash must still work with maps enabled.

    A category with no shard on disk is recomputed from scratch, so any .npy left
    in its map directory is orphaned by construction: no result row references it.
    Those must be cleared, otherwise the overwrite guard would turn every resumed
    run into a crash.
    """
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    category_maps = maps / "mvt__echo__can"
    category_maps.mkdir(parents=True)
    orphan = category_maps / "left__over__from__a__dead__run.npy"
    np.save(orphan, np.zeros((4, 4), dtype=np.float16))

    run_evaluation(_MVTecLayoutDataset(), _EchoMethod(), store, {"seed": 0}, maps_dir=maps)

    assert not orphan.exists()
    assert len(sorted(category_maps.glob("*.npy"))) == 2


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
