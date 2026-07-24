import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import AnomalyMethod, Prediction


def test_accepts_a_map_outside_zero_one():
    """Raw anomaly scores are not bounded to [0,1]; the contract must accept them."""
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    pred = Prediction(image_score=42.0, anomaly_map=np.full((6, 8), 7.5, dtype=np.float32))
    assert_valid_prediction(pred, img)  # must not raise


def test_still_rejects_a_non_native_shape():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    pred = Prediction(image_score=1.0, anomaly_map=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(AssertionError):
        assert_valid_prediction(pred, img)


def test_still_rejects_a_non_finite_map():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    bad = np.zeros((6, 8), dtype=np.float32)
    bad[0, 0] = np.nan
    with pytest.raises(AssertionError):
        assert_valid_prediction(Prediction(image_score=1.0, anomaly_map=bad), img)


def test_still_rejects_a_non_float32_map():
    img = np.zeros((6, 8, 3), dtype=np.uint8)
    with pytest.raises(AssertionError):
        assert_valid_prediction(Prediction(1.0, np.zeros((6, 8), dtype=np.float64)), img)


def test_fit_is_a_noop_on_the_base_contract():
    """Zero-shot methods inherit fit() and it does nothing."""
    AnomalyMethod().fit(iter([np.zeros((4, 4, 3), dtype=np.uint8)]), "vial")  # must not raise
