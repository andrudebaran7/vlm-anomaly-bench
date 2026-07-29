"""AdaCLIP zero-shot adapter (Cao et al., ECCV 2024) — hybrid learnable prompts.

AdaCLIP adapts CLIP with hybrid prompts: a static component learned on auxiliary anomaly data and
fixed in the checkpoint, plus a dynamic component generated per test image. It scores any image
zero-shot. It is auxiliary-trained, so which checkpoint (trained on which dataset) is used for which
test set is a pre-registration decision audited in docs/adaclip-overlap-audit.md — the auxiliary data
must not overlap the test set.

The backend seam takes a category:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

This is the superset choice. Whether AdaCLIP's inference path actually consumes a class name is
verified on Colab against the official repo; if it does not, the backend ignores the argument and
says so in its own docstring. An inert argument costs nothing (AnomalyMethod.predict() receives
`category` unconditionally), while a missing one would force an adapter rewrite mid-GPU-session.

anomalib does not ship AdaCLIP, so provenance is the official repo at a pinned commit (protocol §3
priority 1). The real inference runs on Colab behind the injected backend; everything here is tested
with a fake backend.

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class AdaClipRef(AnomalyMethod):
    name = "adaclip"
    zero_shot = True

    def __init__(self, backend=None):
        self._backend = backend

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real backend is
        Colab-only (it needs the official repo, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the official repo + CLIP + GPU
            raise MethodNotRunnable(
                "AdaClipRef needs an AdaCLIP backend; build it in a GPU session from the official "
                "repo at the pinned commit and inject it (see the AdaCLIP Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("AdaClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
