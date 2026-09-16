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
    run_evaluation(_MVTecLayoutDataset(), _EchoMethod(), store, {}, maps_dir=maps)

    saved = sorted((maps / f"mvt__echo__{run_id({'seed': None})}__can__unseeded").glob("*.npy"))
    assert len(saved) == 2, f"expected one map per sample, got {[p.name for p in saved]}"

    df = pd.read_parquet(store.path_for("mvt", "echo", "can", None))
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
        run_evaluation(_DuplicatePathDataset(), _EchoMethod(), store, {}, maps_dir=maps)


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
    category_maps = maps / f"mvt__echo__{run_id({'seed': None})}__can__unseeded"
    category_maps.mkdir(parents=True)
    orphan = category_maps / "left__over__from__a__dead__run.npy"
    np.save(orphan, np.zeros((4, 4), dtype=np.float16))

    run_evaluation(_MVTecLayoutDataset(), _EchoMethod(), store, {}, maps_dir=maps)

    assert store.is_done("mvt", "echo", "can", None)
    assert orphan.exists(), "the runner must not delete files it does not own"
    assert len(sorted(category_maps.glob("*.npy"))) == 3  # 2 recomputed + the orphan


def test_runner_writes_one_shard_per_category(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    written = run_evaluation(fake_dataset, counting_method, store, {})
    assert len(written) == 2
    assert store.is_done("fake", "counting", "alpha", None)
    assert store.is_done("fake", "counting", "beta", None)


def test_runner_records_every_sample(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {})
    df = store.load_all()
    assert len(df) == 6
    assert set(df.columns) >= {"image_path", "label", "image_score", "category", "method", "seed"}


def test_runner_prepares_the_method_once(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {})
    assert counting_method.prepare_calls == 1


def test_runner_skips_completed_categories_on_resume(tmp_path, fake_dataset, counting_method):
    """The Colab disconnection case: a second run must not recompute finished work."""
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    assert counting_method.seen == ["alpha"] * 3

    resumed = run_evaluation(fake_dataset, counting_method, store, {})
    assert counting_method.seen == ["alpha"] * 3 + ["beta"] * 3
    assert [p.name for p in resumed] == ["fake__counting__beta__unseeded.parquet"]


def test_runner_does_not_prepare_when_everything_is_done(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {})
    counting_method.prepare_calls = 0
    assert run_evaluation(fake_dataset, counting_method, store, {}) == []
    assert counting_method.prepare_calls == 0


def test_runner_saves_maps_when_asked(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    run_evaluation(
        fake_dataset, counting_method, store, {}, categories=["alpha"], maps_dir=maps
    )
    saved = sorted(
        (maps / f"fake__counting__{run_id({'seed': None})}__alpha__unseeded").glob("*.npy")
    )
    assert len(saved) == 3
    assert np.load(saved[0]).dtype == np.float16

    df = pd.read_parquet(store.path_for("fake", "counting", "alpha", None))
    assert df["map_path"].notna().all()


def test_runner_omits_map_path_when_not_saving(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha", None))
    assert "map_path" not in df.columns


def test_runner_prefixes_non_split_meta_and_excludes_split(tmp_path, counting_method):
    """meta_ columns come from Sample.meta, excluding split (already its own column) —
    e.g. a lighting condition (MVTec AD 2) that later lets results be sliced by lighting."""
    store = ResultStore(tmp_path)
    run_evaluation(_LitDataset(), counting_method, store, {})
    df = pd.read_parquet(store.path_for("lit", "counting", "alpha", None))
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
    df_a = pd.read_parquet(store_a.path_for("fake", "counting", "alpha", None))
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
    orphan_dir = maps / f"fake__counting__{run_id(meta)}__alpha__unseeded"
    orphan_dir.mkdir(parents=True)
    orphan = orphan_dir / map_filename(Path("/fake/alpha/0.png"))
    np.save(orphan, np.zeros((2, 2), dtype=np.float16))

    run_evaluation(fake_dataset, counting_method, store, meta,
                   categories=["alpha"], maps_dir=maps)

    assert store.is_done("fake", "counting", "alpha", None)
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
    run_evaluation(_WithMasks(), counting_method, store, {})
    # pandas>=3 defaults to a NaN-backed "str" dtype for object columns, so a missing
    # string entry reads back as NaN rather than None even though the column (and the
    # parquet file itself) stores a proper null. Real consumers read under this
    # default, so assert against it rather than opting into legacy behaviour.
    df = pd.read_parquet(store.path_for("masked", "counting", "alpha", None))
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
        run_evaluation(_Empty(), counting_method, store, {}, maps_dir=maps)
    assert not maps.exists() or not list(maps.rglob("*.npy"))


def test_default_split_is_a_real_mvtec_ad2_split(tmp_path, counting_method):
    """`split="test"` is not one of MVTec AD 2's five splits.

    Any caller relying on the default got a ValueError out of the loader before a single
    sample was read, so the default was not merely unhelpful — it could not work at all.
    The default has to be `test_public`: it is the only split with pixel ground truth and
    the only one every local metric is computed on.
    """
    from mvtec_tree import build_category

    from vlmab.datasets.mvtec_ad2 import MVTecAD2

    build_category(tmp_path / "data", "vial", conditions=("regular",), n_good=1, n_bad=1)
    dataset = MVTecAD2(tmp_path / "data")
    store = ResultStore(tmp_path / "results")

    written = run_evaluation(dataset, counting_method, store, {})

    assert len(written) == 1
    df = pd.read_parquet(written[0])
    assert set(df["split"]) == {"test_public"}
    assert sorted(df["label"]) == [0, 1]  # one good, one bad, i.e. the public test split


def test_runner_records_a_positive_latency_per_row(tmp_path, fake_dataset, counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha", None))
    assert "latency_ms" in df.columns
    assert (df["latency_ms"] >= 0).all()
    assert df["latency_ms"].notna().all()


def test_runner_carries_prediction_extras_into_prefixed_columns(tmp_path, fake_dataset):
    """API methods report tokens+cost via extras (protocol §4); the shard must keep them."""
    class _ExtrasMethod(AnomalyMethod):
        name = "extras"
        def prepare(self, device="cuda"):
            pass
        def predict(self, image, category):
            return Prediction(image_score=0.5, anomaly_map=np.zeros((8, 8), dtype=np.float32),
                              extras={"tokens": 123, "parse_ok": True})

    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _ExtrasMethod(), store, {}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "extras", "alpha", None))
    assert list(df["extras_tokens"]) == [123, 123, 123]
    assert list(df["extras_parse_ok"]) == [True, True, True]


def test_runner_omits_extras_columns_when_a_method_returns_none(tmp_path, fake_dataset,
                                                                counting_method):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    df = pd.read_parquet(store.path_for("fake", "counting", "alpha", None))
    assert not [c for c in df.columns if c.startswith("extras_")]


def test_runner_fits_a_full_shot_method_once_per_category_before_predicting(tmp_path, fake_dataset):
    """A full-shot method must see a category's train images (via fit) before it scores that
    category's test images."""
    events = []

    class _FullShot(AnomalyMethod):
        name = "fullshot"
        zero_shot = False

        def prepare(self, device="cuda"):
            events.append(("prepare", None))

        def fit(self, train_images, category):
            events.append(("fit", category, len(list(train_images))))

        def predict(self, image, category):
            events.append(("predict", category))
            return Prediction(image_score=0.5, anomaly_map=np.zeros((8, 8), dtype=np.float32))

    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _FullShot(), store, {}, categories=["alpha"])

    assert events[0] == ("prepare", None)
    assert events[1] == ("fit", "alpha", 3)          # FakeDataset yields 3 train images per category
    assert all(e[0] == "predict" for e in events[2:])  # every fit precedes its predicts


def test_runner_raises_instead_of_silently_writing_an_inf_map(tmp_path, fake_dataset):
    """A finite float32 anomaly map with a value above float16's max (65504) becomes
    `inf` under the runner's float16 cast with no error, silently corrupting every
    pixel metric computed from that map. The runner must fail loud instead."""
    class _HugeScoreMethod(AnomalyMethod):
        name = "huge"
        zero_shot = True

        def prepare(self, device="cuda"):
            pass

        def predict(self, image, category):
            huge = np.full((8, 8), 1e6, dtype=np.float32)
            return Prediction(image_score=0.5, anomaly_map=huge)

    store = ResultStore(tmp_path / "results")
    maps = tmp_path / "maps"
    with pytest.raises(ValueError, match="65504|1000000|1e\\+06|float16"):
        run_evaluation(
            fake_dataset, _HugeScoreMethod(), store, {},
            categories=["alpha"], maps_dir=maps,
        )
    saved = list(maps.rglob("*.npy"))
    assert not any(np.isinf(np.load(p)).any() for p in saved), (
        "an inf map was written to disk instead of the runner raising"
    )


def test_runner_does_not_fit_a_zero_shot_method(tmp_path, fake_dataset, counting_method):
    """counting_method is zero_shot; its fit() must never be called."""
    calls = []
    original = counting_method.fit

    def _spy(train_images, category):
        calls.append(category)
        return original(train_images, category)

    counting_method.fit = _spy
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    assert calls == []


class _SeededMethod(AnomalyMethod):
    """Declares a seed the way a real stochastic adapter does."""

    name = "seeded"
    zero_shot = True
    seed = 7

    def prepare(self, device="cuda"):
        pass

    def predict(self, image, category):
        return Prediction(
            image_score=0.5, anomaly_map=np.full((8, 8), 0.5, dtype=np.float32)
        )


def test_runner_stamps_the_seed_the_method_declares(tmp_path, fake_dataset):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _SeededMethod(), store, {}, categories=["alpha"])
    df = store.load_all()
    assert list(df["seed"]) == [7, 7, 7]


def test_runner_stamps_none_for_a_method_that_seeded_nothing(
    tmp_path, fake_dataset, counting_method
):
    """None is a real value, not a missing one: it says no seed was applied, which is exactly
    what a deterministic run's provenance should say."""
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, counting_method, store, {}, categories=["alpha"])
    df = store.load_all()
    assert "seed" in df.columns and df["seed"].isna().all()


def test_runner_refuses_a_seed_supplied_by_the_caller(tmp_path, fake_dataset, counting_method):
    """The original defect: run_meta took a seed from the CLI and stamped it while the method
    applied nothing. There must be exactly one writer of this field."""
    store = ResultStore(tmp_path)
    with pytest.raises(ValueError, match="method.seed"):
        run_evaluation(fake_dataset, counting_method, store, {"seed": 0}, categories=["alpha"])


class _OtherSeedMethod(_SeededMethod):
    """Same method, different seed, and a DISTINGUISHABLE map — otherwise "the first run's maps
    survived" passes trivially because both runs wrote identical arrays."""

    seed = 8

    def predict(self, image, category):
        return Prediction(
            image_score=0.5, anomaly_map=np.full((8, 8), 0.99, dtype=np.float32)
        )


def test_a_second_seed_is_not_mistaken_for_a_finished_run(tmp_path, fake_dataset):
    store = ResultStore(tmp_path)
    run_evaluation(fake_dataset, _SeededMethod(), store, {}, categories=["alpha"])
    written = run_evaluation(fake_dataset, _OtherSeedMethod(), store, {}, categories=["alpha"])
    assert [p.name for p in written] == ["fake__seeded__alpha__seed8.parquet"]
    assert sorted(store.load_all()["seed"].unique()) == [7, 8]


def test_two_seeds_do_not_share_a_map_directory(tmp_path, fake_dataset):
    """run_id is config_hash, which carries no seed, so two seeds of one config resolved to one
    map directory: the second run overwrote the first run's maps while the first run's finished
    shard still pointed at them."""
    maps = tmp_path / "maps"
    store = ResultStore(tmp_path / "r")
    run_evaluation(fake_dataset, _SeededMethod(), store, {"config_hash": "cfg"},
                   categories=["alpha"], maps_dir=maps)
    first = {p: np.load(p).copy() for p in store.load_all()["map_path"]}
    assert first and not np.allclose(list(first.values())[0], 0.99)

    run_evaluation(fake_dataset, _OtherSeedMethod(), store, {"config_hash": "cfg"},
                   categories=["alpha"], maps_dir=maps)

    for path, data in first.items():
        assert Path(path).exists(), "the second seed deleted the first seed's maps"
        assert np.array_equal(np.load(path), data), "the second seed overwrote them"
