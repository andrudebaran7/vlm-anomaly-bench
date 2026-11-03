"""scripts/export_image_scores.py: what goes public, and that the analyses can read it back."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("export_image_scores", ROOT / "scripts" / "export_image_scores.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)
boot = ex.sys.modules["bootstrap_iauroc"]


def _shards():
    return pd.DataFrame({
        "split": ["test_public"] * 3,
        "label": [0, 1, 1],
        "image_path": ["/content/data/secret/good/000_regular.png", "/content/data/secret/bad/001_regular.png",
                       "/content/data/secret/bad/001_shift_1.png"],
        "meta_lighting": ["regular", "regular", "shift_1"],
        "seed": [0, 0, 0],
        "image_score": [0.1234567890123456789, 12.5, 1e-7],
    })


def test_the_export_carries_no_path_and_no_extra_column(tmp_path):
    path = tmp_path / "antialiased" / "vial.csv"
    ex.write_scores(boot.image_table(_shards(), "antialiased"), path)
    text = path.read_text()
    assert text.splitlines()[0] == "folder,index,condition,label,seed,score"
    assert "/content" not in text and "secret" not in text and "antialiased" not in text


def test_the_analyses_read_the_export_back_exactly(tmp_path):
    table = boot.image_table(_shards(), "antialiased")
    ex.write_scores(table, tmp_path / "antialiased" / "vial.csv")
    back = boot.load(tmp_path, "vial", "antialiased")
    keys = ["folder", "index", "condition", "seed"]
    a = table.sort_values(keys).reset_index(drop=True)
    b = back.sort_values(keys).reset_index(drop=True)
    assert list(b["index"]) == list(a["index"]), "scene indices keep their leading zeros"
    assert list(b["score"]) == list(a["score"]), "scores keep full precision"
    assert (b["config"] == "antialiased").all() and list(b["label"]) == list(a["label"])


def test_a_missing_file_is_absent_and_not_an_error_of_another_kind(tmp_path):
    with pytest.raises(FileNotFoundError):
        boot.load(tmp_path, "fruit_jelly", "aliased")


def test_the_scores_are_tracked_and_licensed_apart():
    import subprocess
    probe = subprocess.run(["git", "-C", str(ROOT), "check-ignore", "-q", "results/image_scores/aliased/can.csv"])
    assert probe.returncode == 1, "the CSVs must not be gitignored"
    assert "CC BY-NC-SA 4.0" in (ROOT / "results" / "image_scores" / "README.md").read_text()
