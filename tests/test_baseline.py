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


def test_a_flat_image_scores_below_a_spotted_one(method):
    plain = _image()
    spotted = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(plain, "vial").image_score < method.predict(spotted, "vial").image_score


def test_two_images_with_different_contrast_get_different_scores(method):
    """The bug this fixes: `normalise_to_unit` made every varied image score exactly 1.0, so
    I-AUROC was 0.5 by ties. Raw scores must differ with contrast."""
    faint = _image(blob=((5, 10, 5, 10), 150))
    strong = _image(blob=((5, 10, 5, 10), 255))
    assert method.predict(faint, "vial").image_score != method.predict(strong, "vial").image_score


def test_a_strong_blob_scores_above_one(method):
    """Proves the map is raw, not per-image normalised to [0,1]: a 255-on-100 blob deviates by
    ~140, far above 1.0."""
    assert method.predict(_image(blob=((5, 10, 5, 10), 255)), "vial").image_score > 1.0


def test_the_blob_is_the_brightest_region_of_the_map(method):
    img = _image(blob=((5, 10, 5, 10), 255))
    amap = method.predict(img, "vial").anomaly_map
    assert amap[7, 7] > amap[30, 50]


def test_a_perfectly_flat_image_gives_an_all_zero_map(method):
    amap = method.predict(_image(), "vial").anomaly_map
    assert (amap == 0.0).all()


def test_prepare_is_a_noop_and_needs_no_gpu():
    IntensityBaseline().prepare(device="cpu")
