import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.saa import SaaRef


class _FakeBackend:
    """Stands in for the real SAA+ cascade. The category is substantive here — it selects the
    per-object domain prompts — so the fake records it. The returned map mimics the real output
    shape: a near-binary field built from a couple of mask regions, not a smooth field.

    The score and the map's maximum are deliberately DIFFERENT values: if they matched, the
    image-score assertion below would still pass under an adapter bug that ignored the backend's
    score and returned the map's maximum instead. Passing the backend's score through untouched is
    the one substantive thing this adapter decides, so the fake has to be able to catch that."""

    def __init__(self):
        self.calls = []

    def score(self, image, category):
        self.calls.append(category)
        m = np.zeros((16, 16), dtype=np.float32)
        m[2:5, 2:5] = 0.81       # one high-confidence mask region
        m[10:12, 9:13] = 0.44    # one lower-confidence region
        return 0.93, m           # NOT the map max — see the class docstring


def test_is_zero_shot_and_training_free():
    m = SaaRef()
    assert m.zero_shot is True
    assert m.name == "saa"


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        SaaRef().prepare(device="cpu")


def test_predict_passes_the_category_to_the_backend():
    """The category is substantive: it selects SAA+'s per-object domain prompts."""
    backend = _FakeBackend()
    m = SaaRef(backend=backend)
    m.prepare(device="cpu")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "vial")
    m.predict(np.zeros((30, 30, 3), dtype=np.uint8), "can")
    assert backend.calls == ["vial", "can"]


def test_predict_returns_a_native_resolution_raw_map():
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((80, 50, 3), dtype=np.uint8)
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)               # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(0.93)   # the BACKEND's score, not the map's max (0.81)
    assert pred.anomaly_map.shape == (80, 50)        # upsampled to native


def test_predict_does_not_smooth_or_renormalise_the_map():
    """The repo's own map is taken verbatim: upsampling only. A post-process no other method in the
    study receives would break comparability (protocol §3)."""
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((16, 16, 3), dtype=np.uint8), "vial")
    # Same size in and out, so upsample_to is identity and the values must survive untouched.
    assert pred.anomaly_map.max() == pytest.approx(0.81)
    assert pred.anomaly_map.min() == pytest.approx(0.0)
    assert sorted(np.unique(pred.anomaly_map).tolist()) == pytest.approx([0.0, 0.44, 0.81], abs=1e-6)


def test_predict_records_the_category_in_extras():
    m = SaaRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    pred = m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "walnuts")
    assert pred.extras == {"category": "walnuts"}


def test_predict_without_a_backend_is_not_runnable():
    m = SaaRef()
    with pytest.raises(MethodNotRunnable):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
