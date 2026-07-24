import pandas as pd

from mvtec_tree import build_category
from run_eval import main
from vlmab.eval.store import ResultStore


def test_run_eval_writes_shards_for_a_real_method(tmp_path):
    build_category(tmp_path, "vial")
    results = tmp_path / "results"
    code = main(["--method", "intensity_baseline", "--root", str(tmp_path),
                 "--split", "test_public", "--results", str(results), "--seed", "0"])
    assert code == 0
    df = ResultStore(results).load_all()
    assert len(df) > 0
    assert set(df["method"]) == {"intensity_baseline"}
    assert "latency_ms" in df.columns


def test_run_eval_rejects_an_unknown_method(tmp_path, capsys):
    build_category(tmp_path, "vial")
    code = main(["--method", "nope", "--root", str(tmp_path), "--results", str(tmp_path / "r")])
    assert code == 1
    assert "intensity_baseline" in capsys.readouterr().out
