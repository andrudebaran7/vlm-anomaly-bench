from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlmab.datasets.base import AnomalyDataset, Sample
from vlmab.eval.runner import map_filename, run_evaluation, run_id
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

    saved = sorted((maps / f"mvt__echo__{run_id({'seed': 0})}__can").glob("*.npy"))
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


def test_runner_leaves_another_runs_maps_alone_when_recomputing(tmp_path):
    """Resume-after-crash must work WITHOUT deleting anything.

    An earlier version cleared a category's map directory before recomputing, arguing
    that a category with no shard has no rows referencing its maps. That argument only
    holds within one store: the directory name did not identify the run, so a second
    run sharing `maps_dir` deleted a *finished* run's maps out from under its shard.
    The run id in the directory name now separates runs, and resume works by
    overwriting the names it is about to write rather than by clearing.
    """
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    category_maps = maps / f"mvt__echo__{run_id({'seed': 0})}__can"
    category_maps.mkdir(parents=True)
    orphan = category_maps / "left__over__from__a__dead__run.npy"
    np.save(orphan, np.zeros((4, 4), dtype=np.float16))

    run_evaluation(_MVTecLayoutDataset(), _EchoMethod(), store, {"seed": 0}, maps_dir=maps)

    assert store.is_done("mvt", "echo", "can")
    assert orphan.exists(), "the runner must not delete files it does not own"
    assert len(sorted(category_maps.glob("*.npy"))) == 3  # 2 recomputed + the orphan


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
    saved = sorted((maps / f"fake__counting__{run_id({'seed': 0})}__alpha").glob("*.npy"))
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


def test_a_second_run_sharing_maps_dir_cannot_destroy_a_finished_run(
    tmp_path, fake_dataset, counting_method
):
    """Two configs of one method write the same shard path, so they need separate store
    roots -- and then they share `maps_dir`. Clearing a category's maps up front
    deleted the other run's data while its shard still pointed at it.

    The second run must produce DISTINGUISHABLE maps, or "was not overwritten" passes
    trivially because both runs wrote identical arrays.
    """
    class _OtherConfigMethod(type(counting_method)):
        def predict(self, image, category):
            p = super().predict(image, category)
            return Prediction(image_score=p.image_score,
                              anomaly_map=np.full((8, 8), 0.99, dtype=np.float32))

    maps = tmp_path / "maps"
    store_a = ResultStore(tmp_path / "a")
    run_evaluation(fake_dataset, counting_method, store_a, {"config_hash": "cfgA"},
                   categories=["alpha"], maps_dir=maps)
    df_a = pd.read_parquet(store_a.path_for("fake", "counting", "alpha"))
    before = {path: np.load(path).copy() for path in df_a["map_path"]}
    assert before and not np.allclose(list(before.values())[0], 0.99)

    store_b = ResultStore(tmp_path / "b")
    run_evaluation(type(fake_dataset)(), _OtherConfigMethod(), store_b,
                   {"config_hash": "cfgB"}, categories=["alpha"], maps_dir=maps)

    for path, data in before.items():
        assert Path(path).exists(), "the second run deleted a finished run's maps"
        assert np.array_equal(np.load(path), data), "the second run overwrote them"


def test_resume_overwrites_orphaned_maps_from_a_crashed_attempt(
    tmp_path, fake_dataset, counting_method
):
    """Maps left by a run that died mid-category are referenced by no shard, so the
    resumed run must be free to overwrite them rather than refusing."""
    maps = tmp_path / "maps"
    store = ResultStore(tmp_path / "results")
    meta = {"config_hash": "cfg"}
    orphan_dir = maps / f"fake__counting__{run_id(meta)}__alpha"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / map_filename(Path("/fake/alpha/0.png"))
    np.save(orphan, np.zeros((2, 2), dtype=np.float16))

    run_evaluation(fake_dataset, counting_method, store, meta,
                   categories=["alpha"], maps_dir=maps)

    assert store.is_done("fake", "counting", "alpha")
    assert np.load(orphan).shape == (8, 8), "orphan was not recomputed over"


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
    # pandas>=3 defaults to a NaN-backed "str" dtype for object columns, so a missing
    # string entry reads back as NaN rather than None even though the column (and the
    # parquet file itself) stores a proper null. Real consumers read under this
    # default, so assert against it rather than opting into legacy behaviour.
    df = pd.read_parquet(store.path_for("masked", "counting", "alpha"))
    assert df["mask_path"][0] == "/fake/gt/000_regular_mask.png"
    assert pd.isna(df["mask_path"][1])


def test_empty_category_does_not_create_a_map_directory(tmp_path, counting_method):
    """store.write() rejects an empty shard; the map directory must not have been
    created on the way to that rejection."""
    maps = tmp_path / "maps"
    store = ResultStore(tmp_path / "results")

    class _Empty(AnomalyDataset):
        name = "empty"

        def categories(self):
            return ["alpha"]

        def samples(self, split, category=None):
            return iter(())

        def load_image(self, sample):
            raise AssertionError("no samples to load")

    with pytest.raises(ValueError):
        run_evaluation(_Empty(), counting_method, store, {"seed": 0}, maps_dir=maps)
    assert not maps.exists() or not list(maps.rglob("*.npy"))
