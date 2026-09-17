import numpy as np
import pytest

from method_contract import assert_valid_prediction
from vlmab.methods.base import MethodNotRunnable
from vlmab.methods.patchcore_ref import PatchCoreRef


class _FakeBackend:
    """Stands in for the real anomalib-backed backend. Records fit, returns a canned score+map."""

    def __init__(self, coarse=(7, 7)):
        self.fitted_images = None
        self._coarse = coarse

    def fit(self, train_images):
        self.fitted_images = list(train_images)

    def score(self, image):
        # A raw (unbounded) image score plus a coarse map with one hot cell — like anomalib.
        m = np.zeros(self._coarse, dtype=np.float32)
        m[1, 2] = 4.2
        return 4.2, m


def test_is_full_shot():
    assert PatchCoreRef().zero_shot is False


def test_prepare_without_a_backend_is_not_runnable():
    with pytest.raises(MethodNotRunnable):
        PatchCoreRef().prepare(device="cpu")


def test_fit_delegates_the_training_images_to_the_backend():
    backend = _FakeBackend()
    m = PatchCoreRef(backend=backend)
    m.prepare(device="cpu")
    imgs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    m.fit(iter(imgs), "vial")
    assert len(backend.fitted_images) == 3


def test_predict_returns_a_native_resolution_raw_map():
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    img = np.zeros((70, 70, 3), dtype=np.uint8)
    m.fit(iter([img]), "vial")
    pred = m.predict(img, "vial")
    assert_valid_prediction(pred, img)              # finite float32 native map, not [0,1]
    assert pred.image_score == pytest.approx(4.2)   # raw score passed through, not rescaled
    assert pred.anomaly_map[15, 25] == pytest.approx(4.2)  # the hot cell, upsampled to native
    assert pred.anomaly_map[0, 0] == 0.0


def test_predict_before_fit_is_a_real_error_not_a_not_runnable():
    """Predicting a category that was never fitted is an orchestration bug — it must surface,
    not be swallowed as 'cannot run here'."""
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    with pytest.raises(RuntimeError) as exc:
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "vial")
    assert not isinstance(exc.value, MethodNotRunnable)


def test_predict_for_a_different_category_than_fitted_is_an_error():
    m = PatchCoreRef(backend=_FakeBackend())
    m.prepare(device="cpu")
    m.fit(iter([np.zeros((8, 8, 3), dtype=np.uint8)]), "vial")
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((8, 8, 3), dtype=np.uint8), "cable")


class _SeededBackend(_FakeBackend):
    """A backend that applied a seed, like the real PatchCoreBackend(seed=...)."""

    def __init__(self, seed=7):
        super().__init__()
        self.seed = seed


def test_declares_the_seed_its_backend_applied():
    assert PatchCoreRef(backend=_SeededBackend(seed=7)).seed == 7


def test_a_backend_that_applied_no_seed_declares_none():
    assert PatchCoreRef(backend=_FakeBackend()).seed is None


def test_a_seed_disagreeing_with_the_backend_is_refused():
    """Two callers each believing they set the seed is the defect this contract removes; it is
    not resolved by a precedence rule nobody will remember."""
    with pytest.raises(ValueError, match="disagree"):
        PatchCoreRef(backend=_SeededBackend(seed=0), seed=1)


class _PreprocessingBackend(_FakeBackend):
    """A backend that declares which pre-processing it applied, as the real one does."""

    def __init__(self, preprocess):
        super().__init__()
        self.preprocess = preprocess


def test_the_adapter_declares_the_preprocessing_its_backend_applied():
    """The seed's lesson, applied to the other half of the configuration: the backend is what
    builds the transform, so the backend is what declares it and the adapter only passes it on.
    A value the adapter invented would put a pre-processing into provenance that nothing ran."""
    assert PatchCoreRef(backend=_PreprocessingBackend("classic")).preprocess == "classic"


def test_a_backend_that_declares_nothing_leaves_the_adapter_declaring_nothing():
    """The fakes in this file predate the seam. None is the honest answer for them — not a
    default that would make an unknown pre-processing look like the anomalib one."""
    assert PatchCoreRef(backend=_FakeBackend()).preprocess is None


def test_asking_for_a_preprocessing_the_backend_does_not_apply_refuses():
    """Exactly the seed guard's shape. The adapter cannot make the backend use the other one,
    so the only safe outcome is to refuse rather than record the wrong one."""
    with pytest.raises(ValueError, match="classic"):
        PatchCoreRef(backend=_PreprocessingBackend("anomalib"), preprocess="classic")


def test_without_a_backend_the_requested_preprocessing_is_kept_for_later_checking():
    """Mirrors the seed: nothing has been constructed that could apply anything, and prepare()
    refuses before this method scores — so keep what was asked for."""
    assert PatchCoreRef(preprocess="classic").preprocess == "classic"
