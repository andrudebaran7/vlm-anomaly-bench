import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.winclip import WinClipRef


class _FakeBackend:
    """Stands in for the real anomalib WinCLIP backend. Records the category, returns a raw
    score plus a coarse map (like CLIP's windowed anomaly map before upsampling).

    The score and the map's hot cell are deliberately DIFFERENT values: if they matched, the
    image-score assertion below would still pass under an adapter bug that ignored the backend's
    score and returned the map's maximum instead. Passing the backend's score through untouched is
    the one substantive thing this adapter decides, so the fake has to be able to catch that."""

    def __init__(self):
        self.seen = []

    def score(self, image, category):
        self.seen.append(category)
        m = np.zeros((15, 15), dtype=np.float32)
        m[3, 4] = 1.4
        return 2.7, m


def test_is_zero_shot():
    assert WinClipRef().zero_shot is True


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        WinClipRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    backend = _FakeBackend()
    m = WinClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "fruit_jelly")
    assert backend.seen == ["fruit_jelly"]


def test_predict_returns_a_native_resolution_raw_map():
    m = WinClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((75, 60, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(2.7)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (75, 60)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(1.4)   # the hot window survives upsampling


def test_predict_without_a_backend_is_not_runnable():
    m = WinClipRef()  # no backend, prepare() not called
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
