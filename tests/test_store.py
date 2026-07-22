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
    assert store.is_done("mvtec_ad2", "winclip", "can") is False


def test_write_then_is_done_true(tmp_path):
    store = ResultStore(tmp_path)
    store.write("mvtec_ad2", "winclip", "can", _rows(), {"seed": 0})
    assert store.is_done("mvtec_ad2", "winclip", "can") is True


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
    tmp = store.path_for("mvtec_ad2", "winclip", "can").with_suffix(".parquet.tmp")
    tmp.write_bytes(b"half a parquet file")
    assert store.is_done("mvtec_ad2", "winclip", "can") is False


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

    final = store.path_for("mvtec_ad2", "winclip", "can")
    assert not final.is_file()
    assert store.is_done("mvtec_ad2", "winclip", "can") is False


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
            {"seed": 999},
        )

    final = store.path_for("mvtec_ad2", "winclip", "can")
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
    assert store.is_done("mvtec_ad2", "winclip", "fabric") is False
    assert store.is_done("mvtec_ad2", "saa", "can") is False


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
