import pytest

from mvtec_tree import build_category
from run_eval import main
from vlmab.eval.store import ResultStore
from vlmab.methods.base import AnomalyMethod


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


class _PrepareForbidden(AnomalyMethod):
    """Stands in for a real method on a fully-resumed run: `run_eval` must not call
    `prepare()` (or `predict()`) at all once every category's shard already exists."""

    name = "intensity_baseline"  # must match the shard already on disk to be considered done
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        raise AssertionError("prepare() must not be called when a run is fully resumed")

    def predict(self, image, category):
        raise AssertionError("predict() must not be called when a run is fully resumed")


def test_run_eval_does_not_prepare_a_method_on_a_fully_resumed_run(tmp_path, monkeypatch):
    # root and results are siblings, not nested, so a second run's dataset.categories()
    # scan of `root` never picks up the (now pre-existing) `results` directory as a bogus
    # category -- see the comment in run_eval.main about why that ordering matters.
    root = tmp_path / "root"
    build_category(root, "vial")
    results = tmp_path / "results"
    argv = ["--method", "intensity_baseline", "--root", str(root),
            "--split", "test_public", "--results", str(results), "--seed", "0"]

    # First run: does the real work and writes the shard.
    assert main(argv) == 0

    # Second run against the same --results: every category is already done, so
    # run_eval must not touch prepare()/predict() at all -- swap in a method that
    # raises AssertionError if either is called, and confirm the run still succeeds.
    monkeypatch.setattr("run_eval.build_method", lambda name: _PrepareForbidden())
    assert main(argv) == 0


def test_run_eval_reports_a_method_that_cannot_run_here_cleanly(tmp_path, capsys):
    build_category(tmp_path, "vial")
    code = main(["--method", "mllm_qwen", "--root", str(tmp_path),
                 "--split", "test_public", "--results", str(tmp_path / "results"), "--seed", "0"])
    assert code == 1
    assert "mllm_qwen" in capsys.readouterr().out


class _CudaFailure(AnomalyMethod):
    """Stands in for a real GPU adapter that hits a genuine runtime error deep inside
    predict() (e.g. a CUDA error, a shape-mismatch bug) -- as opposed to the "cannot run
    here" signal a method raises from prepare(). This must NOT be mistaken for that signal."""

    name = "intensity_baseline"  # match the registry name run_eval resolves via --method
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        pass

    def predict(self, image, category):
        raise RuntimeError("simulated CUDA failure")


def test_run_eval_does_not_mask_a_genuine_runtime_error_from_predict(tmp_path, monkeypatch):
    """A plain RuntimeError raised from inside a real predict() is a genuine bug -- it must
    propagate out of main(), not be caught and misreported as "cannot run here"."""
    build_category(tmp_path, "vial")
    monkeypatch.setattr("run_eval.build_method", lambda name: _CudaFailure())
    with pytest.raises(RuntimeError, match="simulated CUDA failure"):
        main(["--method", "intensity_baseline", "--root", str(tmp_path),
              "--split", "test_public", "--results", str(tmp_path / "results"), "--seed", "0"])
