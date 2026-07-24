import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.baseline import IntensityBaseline


@pytest.fixture
def method():
    m = IntensityBaseline()
    m.prepare(device="cpu")
    return m


def _image(shape=(40, 60), fill=100, blob=None):
    img = np.full((*shape, 3), fill, dtype=np.uint8)
    if blob is not None:
        (y0, y1, x0, x1), value = blob
        img[y0:y1, x0:x1] = value
    return img


def test_prediction_satisfies_the_contract(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    assert_valid_prediction(method.predict(img, "vial"), img)


def test_map_is_native_resolution(method):
    img = _image(shape=(31, 47))
    assert method.predict(img, "vial").anomaly_map.shape == (31, 47)


def test_a_uniform_image_scores_lower_than_one_with_a_bright_blob(method):
    plain = _image()
    spotted = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(plain, "vial").image_score < method.predict(spotted, "vial").image_score


def test_the_blob_is_the_brightest_region_of_the_map(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    amap = method.predict(img, "vial").anomaly_map
    assert amap[7, 7] > amap[30, 50]  # inside the blob vs. the plain background


def test_a_perfectly_flat_image_gives_an_all_zero_map(method):
    amap = method.predict(_image(), "vial").anomaly_map
    assert (amap == 0.0).all()


def test_prepare_is_a_noop_and_needs_no_gpu():
    IntensityBaseline().prepare(device="cpu")  # must not raise
