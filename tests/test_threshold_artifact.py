import pytest
import yaml

from vlmab.threshold.artifact import (
    ARTIFACT_PROTOCOL_VERSION,
    load_artifact,
    write_artifact,
)
from vlmab.threshold.rules import Calibration


def _categories():
    return {
        "vial": {
            "global_quantile": Calibration("global_quantile", 12.34, 1e-3, 1000, 0),
            "per_image_robust_z": Calibration("per_image_robust_z", 8.7, 1e-3, 1000, 2),
            "transductive_quantile": Calibration("transductive_quantile", float("nan"), 1e-3, 0, 0),
        }
    }


def _write(tmp_path, **overrides):
    kwargs = dict(
        dataset="mvtec_ad2",
        method="intensity_baseline",
        alpha=1e-3,
        calibrated_on={"split": "validation", "n_images": 41, "lighting": ["regular"]},
        run_id="4880865ba2ed",
        categories=_categories(),
        designated_for_submission="per_image_robust_z",
    )
    kwargs.update(overrides)
    return write_artifact(tmp_path / "a.yaml", **kwargs)


def test_write_artifact_records_everything_a_reader_needs_to_audit_it(tmp_path):
    raw = yaml.safe_load(_write(tmp_path).read_text())
    assert raw["dataset"] == "mvtec_ad2"
    assert raw["method"] == "intensity_baseline"
    assert raw["alpha"] == 1e-3
    assert raw["protocol_version"] == ARTIFACT_PROTOCOL_VERSION
    assert raw["calibrated_on"]["split"] == "validation"
    assert raw["run_id"] == "4880865ba2ed"
    assert raw["designated_for_submission"] == "per_image_robust_z"
    assert raw["categories"]["vial"]["global_quantile"]["threshold"] == 12.34
    assert raw["categories"]["vial"]["per_image_robust_z"]["k"] == 8.7
    assert raw["categories"]["vial"]["per_image_robust_z"]["degenerate_mad"] == 2
    assert raw["categories"]["vial"]["transductive_quantile"] == {}


def test_write_artifact_stamps_the_calibration_date(tmp_path):
    raw = yaml.safe_load(_write(tmp_path).read_text())
    assert len(raw["calibrated_at"]) == 10 and raw["calibrated_at"].count("-") == 2


def test_load_artifact_round_trips_what_was_written(tmp_path):
    loaded = load_artifact(_write(tmp_path))
    assert loaded["categories"]["vial"]["per_image_robust_z"]["k"] == 8.7


def test_load_artifact_rejects_a_stale_protocol_version(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["protocol_version"] = "0.2.10"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="protocol_version"):
        load_artifact(path)


def test_load_artifact_rejects_a_designated_rule_missing_from_a_category(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["categories"]["vial"]["per_image_robust_z"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="designated_for_submission"):
        load_artifact(path)


def test_write_artifact_rejects_a_designated_rule_that_is_not_a_rule(tmp_path):
    with pytest.raises(ValueError, match="designated_for_submission"):
        _write(tmp_path, designated_for_submission="vibes")


def test_write_artifact_rejects_an_alpha_outside_the_unit_interval(tmp_path):
    with pytest.raises(ValueError, match="alpha"):
        _write(tmp_path, alpha=0.0)


def test_load_artifact_rejects_an_unknown_rule_name(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["categories"]["vial"]["magic"] = {"threshold": 1.0}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="unknown rule"):
        load_artifact(path)


def test_load_artifact_rejects_missing_categories(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["categories"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="no categories key"):
        load_artifact(path)


def test_load_artifact_rejects_empty_categories(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["categories"] = {}
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="empty categories"):
        load_artifact(path)


def test_load_artifact_rejects_categories_not_a_mapping(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["categories"] = ["vial"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="categories must be a mapping"):
        load_artifact(path)


def test_load_artifact_rejects_missing_alpha(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["alpha"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="alpha"):
        load_artifact(path)


def test_load_artifact_rejects_non_numeric_alpha(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["alpha"] = "not_a_number"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="alpha"):
        load_artifact(path)


def test_load_artifact_rejects_out_of_range_alpha(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["alpha"] = 0.0
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="alpha"):
        load_artifact(path)


def test_load_artifact_rejects_designated_for_submission_not_in_rules(tmp_path):
    path = _write(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["designated_for_submission"] = "nonexistent_rule"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="designated_for_submission"):
        load_artifact(path)
