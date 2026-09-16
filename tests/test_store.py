from pathlib import Path

import pandas as pd
import pytest

from vlmab.eval.store import ResultStore


def _rows():
    return [
        {"image_path": "a.png", "label": 0, "image_score": 0.1},
        {"image_path": "b.png", "label": 1, "image_score": 0.9},
    ]


def test_is_done_false_before_write(tmp_path):
    store = ResultStore(tmp_path)
    assert store.is_done("mvtec_ad2", "winclip", "can", 0) is False


def test_write_then_is_done_true(tmp_path):
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    assert store.is_done("mvtec_ad2", "winclip", "can", 0) is True


def test_write_stamps_meta_onto_every_row(tmp_path):
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0, "gpu": "T4"})
    df = pd.read_parquet(path)
    assert list(df["seed"]) == [0, 0]
    assert list(df["gpu"]) == ["T4", "T4"]
    assert list(df["category"]) == ["can", "can"]
    assert list(df["dataset"]) == ["mvtec_ad2", "mvtec_ad2"]
    assert list(df["method"]) == ["winclip", "winclip"]


def test_is_done_ignores_stray_temp_file(tmp_path):
    """is_done() must ignore an unrelated .tmp file left behind on disk.

    This does not by itself prove write() is crash-safe: no final shard is ever
    created in this test, so is_done() would return False here for any
    implementation, including one that writes straight to the final path with
    no tmp+rename at all. See test_failed_write_leaves_no_final_shard and
    test_failed_write_preserves_previous_shard below for the actual
    crash-safety property.
    """
    store = ResultStore(tmp_path)
    tmp = store.path_for("mvtec_ad2", "winclip", "can", 0).with_suffix(".parquet.tmp")
    tmp.write_bytes(b"half a parquet file")
    assert store.is_done("mvtec_ad2", "winclip", "can", 0) is False


def _crashing_to_parquet(self, path, *args, **kwargs):
    """Stand-in for DataFrame.to_parquet that simulates a kill mid-write.

    Writes some bytes to the target path (as a real interrupted parquet write
    would leave a truncated file behind) and then raises, so the exception
    propagates out of ResultStore.write() before tmp.replace(final) can run.
    """
    Path(path).write_bytes(b"CORRUPT-PARTIAL-PARQUET")
    raise OSError("simulated Colab disconnect mid-write")


def test_failed_write_leaves_no_final_shard(tmp_path, monkeypatch):
    """If the parquet write itself fails, no final shard must appear.

    This is the property a direct (no tmp+rename) implementation would fail:
    it would either write partial data straight to the final path or leave a
    final path that is_done() wrongly trusts.
    """
    store = ResultStore(tmp_path)
    monkeypatch.setattr(pd.DataFrame, "to_parquet", _crashing_to_parquet)

    with pytest.raises(OSError):
        store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})

    final = store.path_for("mvtec_ad2", "winclip", "can", 0)
    assert not final.is_file()
    assert store.is_done("mvtec_ad2", "winclip", "can", 0) is False


def test_failed_write_preserves_previous_shard(tmp_path, monkeypatch):
    """A crashed re-run must not corrupt or lose an already-completed shard."""
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    original = store.load_all().reset_index(drop=True)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _crashing_to_parquet)
    with pytest.raises(OSError):
        store.write(
            "mvtec_ad2",
            "winclip",
            "can",
            [{"image_path": "z.png", "label": 1, "image_score": 0.5}],
            {"seed": 0},
        )

    final = store.path_for("mvtec_ad2", "winclip", "can", 0)
    assert final.is_file()
    reloaded = store.load_all().reset_index(drop=True)
    pd.testing.assert_frame_equal(reloaded, original)


def test_successful_write_leaves_no_tmp_file_behind(tmp_path):
    """A normal write must not leave a .tmp file lying around in the store dir.

    This is what distinguishes the tmp+rename implementation from a direct
    write, and would fail if write() were later "simplified" to skip renaming.
    """
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    assert list(tmp_path.glob("*.tmp")) == []


def test_is_done_is_scoped_per_method_and_category(tmp_path):
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    assert store.is_done("mvtec_ad2", "winclip", "fabric", 0) is False
    assert store.is_done("mvtec_ad2", "saa", "can", 0) is False


def test_load_all_concatenates_every_shard(tmp_path):
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    store.write("mvtec_ad2", "winclip", "fabric", _rows(), {"seed": 0})
    df = store.load_all()
    assert len(df) == 4
    assert set(df["category"]) == {"can", "fabric"}


def test_load_all_empty_store_returns_empty_frame(tmp_path):
    assert ResultStore(tmp_path).load_all().empty


def test_write_rejects_empty_rows(tmp_path):
    with pytest.raises(ValueError):
        ResultStore(tmp_path).write("mvtec_ad2", "winclip", "can", [], {"seed": 0})


@pytest.mark.parametrize("key", ["dataset", "method", "category"])
def test_write_rejects_meta_keys_that_shadow_identity_columns(tmp_path, key):
    """meta was applied after the identity columns, so a meta key named `category`
    silently relabelled every row of the shard — the file name still said `can`
    while the column said something else, and load_all() would then group results
    under the wrong category with nothing to flag it."""
    store = ResultStore(tmp_path)
    with pytest.raises(ValueError, match=key):
        store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0, key: "wrong"})


def test_write_still_accepts_non_colliding_meta(tmp_path):
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0, "commit": "abc"})
    df = pd.read_parquet(path)
    assert list(df["category"]) == ["can", "can"]
    assert list(df["commit"]) == ["abc", "abc"]


@pytest.mark.parametrize("key", ["label", "image_score", "mask_path", "map_path", "split"])
def test_write_rejects_meta_keys_that_shadow_row_schema_columns(tmp_path, key):
    """meta is stamped on after the rows, so a meta key named `label` (or image_score,
    mask_path, map_path, split) silently overwrote real per-sample data with one constant.

    That is worse than the identity-column case it already guarded: a shard whose `label`
    column is all 0 still aggregates, and reports metrics computed against the wrong ground
    truth, with nothing anywhere to flag it.
    """
    store = ResultStore(tmp_path)
    with pytest.raises(ValueError, match=key):
        store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0, key: "wrong"})


def test_write_rejects_meta_keys_that_shadow_any_column_the_rows_carry(tmp_path):
    """The guard must not be limited to a hard-coded list: rows also carry meta_* columns
    (e.g. meta_lighting, the whole point of this dataset) and whatever a future Sample adds."""
    store = ResultStore(tmp_path)
    rows = [{"image_path": "a.png", "label": 0, "image_score": 0.1, "meta_lighting": "regular"}]
    with pytest.raises(ValueError, match="meta_lighting"):
        store.write("mvtec_ad2", "winclip", "can", rows, {"seed": 0, "meta_lighting": "wrong"})


from vlmab.eval.store import seed_tag


def test_seed_tag_is_readable_for_both_kinds_of_run():
    assert seed_tag(0) == "seed0"
    assert seed_tag(12) == "seed12"
    assert seed_tag(None) == "unseeded"


def test_two_seeds_write_two_shards(tmp_path):
    store = ResultStore(tmp_path)
    a = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 0})
    b = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 1})
    assert a != b
    assert a.name == "mvtec_ad2__patchcore_ref__vial__seed0.parquet"
    assert b.name == "mvtec_ad2__patchcore_ref__vial__seed1.parquet"
    assert a.is_file() and b.is_file()


def test_a_shard_is_named_for_the_seed_it_records(tmp_path):
    """write() takes the path's seed from the meta it is about to stamp, so the name and the
    column cannot disagree — there is no second argument to get out of step."""
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 3})
    assert "seed3" in path.name
    assert set(pd.read_parquet(path)["seed"]) == {3}


def test_a_run_with_no_seed_is_named_unseeded(tmp_path):
    store = ResultStore(tmp_path)
    path = store.write("mvtec_ad2", "intensity_baseline", "vial", _rows(), {"seed": None})
    assert path.name == "mvtec_ad2__intensity_baseline__vial__unseeded.parquet"


def test_is_done_is_scoped_per_seed(tmp_path):
    """The resume trap: without this, a second seed's run finds the first seed's shard, calls
    the category done, skips it, and leaves a root that reads as two seeds and is one."""
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "patchcore_ref", "vial", _rows(), {"seed": 0})
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", 0) is True
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", 1) is False
    assert store.is_done("mvtec_ad2", "patchcore_ref", "vial", None) is False
