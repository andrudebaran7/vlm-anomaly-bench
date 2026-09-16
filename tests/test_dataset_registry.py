"""The dataset registry is part of the result format, not a convenience.

The name selected here is stamped into every shard's `dataset` column and is what
`scripts/reproduction_gate.py` matches its pre-registered targets against. A rename would
silently orphan every existing shard from its targets.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_eval import main as run_eval_main  # noqa: E402

from vlmab.datasets.mvtec_ad import MVTecAD  # noqa: E402
from vlmab.datasets.mvtec_ad2 import MVTecAD2  # noqa: E402
from vlmab.datasets.registry import DATASETS, available, build_dataset  # noqa: E402
from vlmab.datasets.visa import VisA  # noqa: E402


def test_every_registered_name_is_the_class_name_attribute():
    """The key and `AnomalyDataset.name` must agree: the runner stamps `dataset.name`, not the
    key, so a disagreement would write shards the CLI cannot find again."""
    for key, cls in DATASETS.items():
        assert key == cls.name


def test_the_three_datasets_are_registered_under_their_stamped_names(tmp_path):
    # A real directory: MVTecAD2 validates its root at construction, so a fake path would
    # fail here for a reason that has nothing to do with the registry.
    assert available() == ["mvtec_ad", "mvtec_ad2", "visa"]
    assert build_dataset("mvtec_ad", tmp_path).__class__ is MVTecAD
    assert build_dataset("mvtec_ad2", tmp_path).__class__ is MVTecAD2
    assert build_dataset("visa", tmp_path).__class__ is VisA


def test_an_unknown_dataset_names_the_alternatives():
    with pytest.raises(KeyError, match="mvtec_ad2"):
        build_dataset("mvtec-ad", "/nowhere")


def test_the_cli_defaults_to_the_primary_benchmark(tmp_path, capsys):
    """Changing this default would silently repoint every documented command."""
    code = run_eval_main([
        "--method", "intensity_baseline", "--root", str(tmp_path),
        "--results", str(tmp_path / "out"), "--dataset", "nope",
    ])
    assert code == 1
    assert "available datasets" in capsys.readouterr().out
    import inspect
    import run_eval
    src = inspect.getsource(run_eval.main)
    assert '"--dataset", default="mvtec_ad2"' in src
