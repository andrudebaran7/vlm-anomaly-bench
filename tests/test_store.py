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


def test_stray_temp_file_is_not_treated_as_done(tmp_path):
    """A session killed mid-write must not look like completed work."""
    store = ResultStore(tmp_path)
    tmp = store.path_for("mvtec_ad2", "winclip", "can").with_suffix(".parquet.tmp")
    tmp.write_bytes(b"half a parquet file")
    assert store.is_done("mvtec_ad2", "winclip", "can") is False


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
