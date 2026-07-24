"""Reused by every adapter's tests: the one definition of a valid Prediction.

A method that fails this cannot be aggregated — `aggregate._load_pair` requires the map to match
the mask's (native) shape, and the pixel metrics require finite float32 values. The map is on the
method's own consistent scale, NOT per-image normalised to [0,1] (protocol v0.2.6): the pixel
metrics rank pooled pixels across images, so a per-image rescale would destroy that ranking.
"""
import math

import numpy as np

from vlmab.methods.base import Prediction


def assert_valid_prediction(pred: Prediction, image: np.ndarray) -> None:
    assert isinstance(pred, Prediction), type(pred)
    assert isinstance(pred.image_score, float), type(pred.image_score)
    assert math.isfinite(pred.image_score), pred.image_score

    amap = pred.anomaly_map
    assert isinstance(amap, np.ndarray), type(amap)
    assert amap.dtype == np.float32, amap.dtype
    assert amap.shape == image.shape[:2], f"{amap.shape} != native {image.shape[:2]}"
    assert np.isfinite(amap).all(), "anomaly map has non-finite values"
