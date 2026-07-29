import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.base import MethodNotRunnable


class _FakeBackend:
    """Stands in for the real AdaCLIP backend. The seam takes a category (superset choice: AdaCLIP
    may or may not consume it — verified on Colab), and returns a raw score plus a coarse map like
    the model's patch anomaly map before upsampling."""

    def __init__(self):
        self.calls = []

    def score(self, image, category):
        self.calls.append(category)
        m = np.zeros((16, 16), dtype=np.float32)
        m[4, 7] = 1.3
        return 2.5, m


def test_is_zero_shot():
    assert AdaClipRef().zero_shot is True


def test_name_is_the_registry_name():
    assert AdaClipRef().name == "adaclip"


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        AdaClipRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    """The seam takes a category, so a category-consuming backend is possible without a rewrite."""
    backend = _FakeBackend()
    m = AdaClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    assert backend.calls == ["vial"]


def test_predict_returns_a_native_resolution_raw_map():
    m = AdaClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    # score() returns 2.5 but the map's hot cell is 1.3: these must differ so this assertion can
    # only pass if the backend's score is passed through unchanged, not recomputed from the map.
    assert pred.image_score == pytest.approx(2.5)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(1.3)


def test_predict_records_the_category_in_extras():
    m = AdaClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "can")
    assert pred.extras == {"category": "can"}


def test_predict_without_a_backend_is_not_runnable():
    m = AdaClipRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
