import numpy as np
import pytest
import yaml

from vlmab.eval.store import ResultStore


def _validation_run(tmp_path, split="validation", n=6, size=12):
    """A shard of defect-free rows with real maps on disk, laid out as the runner lays them out.

    The map directory name matters: `run_id` is not a shard column, so
    `<dataset>__<method>__<run_id>__<category>` is the only place the run is recorded and the
    only place the artifact's provenance can come from.
    """
    maps = tmp_path / "maps" / "mvtec_ad2__intensity_baseline__cafe1234__vial"
    maps.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        path = maps / f"{i}.npy"
        np.save(path, rng.random((size, size)).astype(np.float16))
        rows.append({
            "image_path": f"/fake/{i}.png",
            "label": 0,
            "image_score": 0.5,
            "split": split,
            "mask_path": None,
            "map_path": str(path),
        })
    results = tmp_path / "shards"
    ResultStore(results).write("mvtec_ad2", "intensity_baseline", "vial", rows, {"seed": 0})
    return results


def _run(argv):
    from calibrate_threshold import main

    return main(argv)


def test_calibrate_refuses_a_shard_that_is_not_the_validation_split(tmp_path):
    # The failure this entire piece of work exists to prevent. Shards are keyed
    # (dataset, method, category) with no split, so pointing this at a test_public run is an
    # easy mistake and a silent one -- unless it raises here.
    results = _validation_run(tmp_path, split="test_public")
    with pytest.raises(ValueError, match="test_public"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_writes_an_artifact_for_every_category_it_found(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "intensity_baseline", "--out", str(out)]) == 0
    raw = yaml.safe_load(out.read_text())
    assert set(raw["categories"]) == {"vial"}
    assert raw["categories"]["vial"]["global_quantile"]["threshold"] > 0
    assert raw["categories"]["vial"]["per_image_robust_z"]["k"] > 0
    assert raw["categories"]["vial"]["transductive_quantile"] == {}


def test_calibrate_defaults_to_the_pre_registered_alpha_and_designated_rule(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    _run(["--results", str(results), "--dataset", "mvtec_ad2",
          "--method", "intensity_baseline", "--out", str(out)])
    raw = yaml.safe_load(out.read_text())
    assert raw["alpha"] == 1e-3
    assert raw["designated_for_submission"] == "per_image_robust_z"


def test_calibrate_records_the_run_id_and_image_count_it_calibrated_from(tmp_path):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    _run(["--results", str(results), "--dataset", "mvtec_ad2",
          "--method", "intensity_baseline", "--out", str(out)])
    raw = yaml.safe_load(out.read_text())
    assert raw["run_id"] == "cafe1234"
    assert raw["calibrated_on"]["n_images"] == 6
    assert raw["calibrated_on"]["split"] == "validation"


def test_calibrate_raises_when_the_method_has_no_shard(tmp_path):
    results = _validation_run(tmp_path)
    with pytest.raises(ValueError, match="winclip"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "winclip", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_raises_when_the_run_saved_no_maps(tmp_path):
    results = _validation_run(tmp_path)
    import pandas as pd

    shard = results / "mvtec_ad2__intensity_baseline__vial.parquet"
    pd.read_parquet(shard).drop(columns=["map_path"]).to_parquet(shard, index=False)
    with pytest.raises(ValueError, match="map_path"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_runs_end_to_end_on_a_synthetic_mvtec_tree(tmp_path):
    """The whole chain CI can run: dataset -> runner -> shard -> calibration artifact.

    The other tests here hand-build a shard, which cannot catch a mismatch between what the
    runner writes and what calibration reads. This goes through `run_evaluation` on the real
    loader over a synthetic tree.

    The method has to produce non-constant maps: `build_category` fills validation images with
    a uniform value, so `intensity_baseline` would return an all-zero deviation map, MAD would
    be 0 on every image and `per_image_robust_z` would have nothing to calibrate on. That is a
    real degenerate path, covered in test_threshold_rules.py; it is not the path this test is
    for.
    """
    from mvtec_tree import build_category
    from vlmab.datasets.mvtec_ad2 import MVTecAD2
    from vlmab.eval.runner import run_evaluation
    from vlmab.methods.base import AnomalyMethod, Prediction

    class NoisyMethod(AnomalyMethod):
        name = "noisy"
        zero_shot = True

        def prepare(self, device="cuda"):
            pass

        def predict(self, image, category):
            h, w = np.asarray(image).shape[:2]
            amap = np.random.default_rng(abs(hash((h, w))) % 2**32).random((h, w)).astype(np.float32)
            return Prediction(image_score=float(amap.max()), anomaly_map=amap)

    root = tmp_path / "data"
    build_category(root, "vial", n_good=3, n_bad=1, size=(12, 10))
    results, maps = tmp_path / "shards", tmp_path / "maps"
    run_evaluation(
        MVTecAD2(root), NoisyMethod(), ResultStore(results), {"seed": 0},
        split="validation", maps_dir=maps, device="cpu",
    )

    out = tmp_path / "cal.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "noisy", "--out", str(out)]) == 0
    raw = yaml.safe_load(out.read_text())
    assert raw["calibrated_on"] == {"split": "validation", "n_images": 3, "lighting": ["regular"]}
    assert np.isfinite(raw["categories"]["vial"]["global_quantile"]["threshold"])
    assert np.isfinite(raw["categories"]["vial"]["per_image_robust_z"]["k"])
    assert raw["run_id"]  # recovered from the runner's map directory name, not a column


def _shard_without_split(tmp_path, category, run_id="cafe1234", n=3, size=12):
    """A shard whose rows never had a `split` key at all -- distinct from a shard that has
    the column but the wrong value. ResultStore.write only creates the columns the rows carry,
    so this shard's DataFrame genuinely has no `split` column."""
    maps = tmp_path / "maps" / f"mvtec_ad2__intensity_baseline__{run_id}__{category}"
    maps.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        path = maps / f"{i}.npy"
        np.save(path, rng.random((size, size)).astype(np.float16))
        rows.append({
            "image_path": f"/fake-nosplit/{i}.png",
            "label": 0,
            "image_score": 0.5,
            "mask_path": None,
            "map_path": str(path),
        })
    results = tmp_path / "shards"
    ResultStore(results).write("mvtec_ad2", "intensity_baseline", category, rows, {"seed": 0})
    return results


def test_calibrate_raises_a_domain_error_when_splits_mix_valid_and_nan(tmp_path):
    # A shard for the same dataset/method that never had a `split` column at all is a real
    # path: load_all() concatenates every shard under the root, and pandas fills the column a
    # given shard lacks with NaN for that shard's rows. Sorting a set that mixes a str and a
    # float NaN raises TypeError -- the guard must turn that into the domain ValueError and
    # name what it actually found, NaN included.
    results = _validation_run(tmp_path)  # writes category "vial" with split="validation"
    _shard_without_split(tmp_path, "cable")  # same results root, no split column at all
    with pytest.raises(ValueError, match="NaN"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_raises_when_the_shard_has_no_split_column(tmp_path):
    # A shard with no `split` column at all must refuse, not raise a bare KeyError -- no split
    # column means the run cannot be shown to be a validation run, so it is never assumed to be
    # one.
    results = _shard_without_split(tmp_path, "vial")
    with pytest.raises(ValueError, match="split column"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml")])


def test_calibrate_refuses_to_designate_the_transductive_rule(tmp_path):
    # transductive_quantile adapts to the split it is scoring (protocol §4), so it is
    # explicitly not eligible for the private-split submission -- reject it before any file
    # is even read, rather than merely relying on the default being correct.
    results = _validation_run(tmp_path)
    with pytest.raises(ValueError, match="transductive_quantile"):
        _run(["--results", str(results), "--dataset", "mvtec_ad2",
              "--method", "intensity_baseline", "--out", str(tmp_path / "a.yaml"),
              "--designate", "transductive_quantile"])


def test_calibrate_warns_on_stderr_when_alpha_is_not_preregistered(tmp_path, capsys):
    # The alpha sweep (protocol §4's sensitivity grid) must never be mistaken for the
    # pre-registration: a non-default alpha gets a stderr warning and no "commit it" line,
    # even when --out points at the committed path.
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "intensity_baseline", "--out", str(out),
                 "--alpha", "1e-2"]) == 0
    captured = capsys.readouterr()
    assert "sensitivity-sweep" in captured.err
    assert "must not be committed" in captured.err
    assert "commit it" not in captured.out


def test_calibrate_prints_the_commit_reminder_at_the_preregistered_alpha(tmp_path, capsys):
    results = _validation_run(tmp_path)
    out = tmp_path / "a.yaml"
    assert _run(["--results", str(results), "--dataset", "mvtec_ad2",
                 "--method", "intensity_baseline", "--out", str(out)]) == 0
    captured = capsys.readouterr()
    assert "commit it" in captured.out
    assert captured.err == ""
