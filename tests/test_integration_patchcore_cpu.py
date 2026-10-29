"""The real anomalib path, fit then score, on CPU (review #25, 2026-10-29).

Every other PatchCore test injects a fake backend, so the one path that produced every anchor
number -- anomalib 2.6.0's Patchcore through `PatchCoreBackend` -- had no test outside Colab, and
the five ways anomalib drifted from its docs (memory: anomalib 2.x API drift) were each found
there, one session at a time. This runs that path end to end on synthetic images.

Skipped unless anomalib and torch are installed; CI's `integration` job installs the pinned
versions from requirements-gpu.txt (CPU wheels). A small backbone keeps the download and the fit
short: this checks that the integration runs and separates the obvious, not the anchor's numbers.
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("anomalib")

from vlmab.methods.patchcore_backend import PatchCoreBackend  # noqa: E402


def _normal(rng, size=96):
    """A smooth, textured, defect-free image."""
    y, x = np.mgrid[:size, :size]
    base = 128 + 40 * np.sin(x / 7.0) * np.cos(y / 9.0)
    img = base[..., None] + rng.normal(0, 3, (size, size, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def test_patchcore_fits_and_scores_through_anomalib_on_cpu():
    rng = np.random.default_rng(0)
    backend = PatchCoreBackend(backbone="resnet18", device="cpu", num_workers=0, seed=0)
    assert backend.resize_antialias is True   # protocol §3 v0.2.21, set by the real pre-processor

    backend.fit(_normal(rng) for _ in range(8))

    good = _normal(rng)
    bad = good.copy()
    bad[30:60, 30:60] = 255                    # a bright square no training image has

    good_score, good_map = backend.score(good)
    bad_score, bad_map = backend.score(bad)

    assert np.isfinite(good_score) and np.isfinite(bad_score)
    assert good_map.ndim == 2 and good_map.shape == bad_map.shape
    assert np.isfinite(good_map).all()
    assert bad_score > good_score
    # The map has to point at the defect, not only the score rise somewhere.
    h, w = bad_map.shape
    inside = bad_map[int(h * 30 / 96):int(h * 60 / 96), int(w * 30 / 96):int(w * 60 / 96)]
    assert inside.mean() > bad_map.mean()


def test_a_second_fit_on_one_backend_is_refused():
    """The shared-backend defect of 2026-10-21, against the real library this time."""
    rng = np.random.default_rng(1)
    backend = PatchCoreBackend(backbone="resnet18", device="cpu", num_workers=0, seed=0)
    backend.fit(_normal(rng) for _ in range(4))
    with pytest.raises(RuntimeError, match="already fitted"):
        backend.fit(_normal(rng) for _ in range(4))
