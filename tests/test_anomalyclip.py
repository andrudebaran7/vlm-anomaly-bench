import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.base import MethodNotRunnable


class _FakeBackend:
    """Stands in for the real AnomalyCLIP backend. Object-agnostic, so score() takes no category;
    returns a raw score plus a coarse map (like the model's patch anomaly map before upsampling).

    The score and the map's hot cell are deliberately DIFFERENT values: if they matched, the
    image-score assertion below would still pass under an adapter bug that ignored the backend's
    score and returned the map's maximum instead. Passing the backend's score through untouched is
    the one substantive thing this adapter decides, so the fake has to be able to catch that."""

    def __init__(self):
        self.calls = 0

    def score(self, image):
        self.calls += 1
        m = np.zeros((16, 16), dtype=np.float32)
        m[5, 6] = 1.8
        return 3.1, m


def test_is_zero_shot():
    assert AnomalyClipRef().zero_shot is True


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        AnomalyClipRef().prepare(device="cpu")


def test_predict_calls_the_backend_with_the_image_only():
    """Object-agnostic: the backend seam takes no category (unlike WinCLIP)."""
    backend = _FakeBackend()
    m = AnomalyClipRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    assert backend.calls == 1


def test_predict_returns_a_native_resolution_raw_map():
    m = AnomalyClipRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(3.1)    # raw score passed through, not rescaled
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native
    assert pred.anomaly_map.max() == pytest.approx(1.8)


def test_predict_without_a_backend_is_not_runnable():
    m = AnomalyClipRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
