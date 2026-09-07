"""AnomalyCLIP zero-shot adapter (Zhou et al., ICLR 2024) — object-agnostic learned prompts.

AnomalyCLIP learns object-AGNOSTIC text prompts on auxiliary anomaly data, then scores any image
zero-shot without per-object prompts. So, unlike WinCLIP, the backend seam takes no category:
score(image) -> (raw_score, raw_map). It is auxiliary-trained, so which checkpoint (trained on
which dataset) is used for which test set is a pre-registration decision audited in
docs/anomalyclip-overlap-audit.md — the auxiliary data must not overlap the test set.

anomalib does not ship AnomalyCLIP, so provenance is the official repo at a pinned commit
(protocol §3 priority 1). The real inference runs on Colab behind the injected backend; everything
here is tested with a fake backend.

The backend is any object with:
    score(image: np.ndarray) -> tuple[float, np.ndarray]   # (raw image score, raw anomaly map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import BackendSeeded, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class AnomalyClipRef(BackendSeeded):
    name = "anomalyclip"
    zero_shot = True

    def __init__(self, backend=None, seed: int | None = None):
        self._backend = backend
        # The backend is what calls seed_everything, so it is what declares the seed.
        self._init_seed(backend, seed)

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real backend is
        Colab-only (it needs the official repo, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the official repo + CLIP + GPU
            raise MethodNotRunnable(
                "AnomalyClipRef needs an AnomalyCLIP backend; build it in a GPU session from the "
                "official repo at the pinned commit and inject it (see the AnomalyCLIP Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        # `category` is part of the AnomalyMethod contract but unused: AnomalyCLIP's prompts are
        # object-agnostic, so the backend scores the image without an object name.
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("AnomalyClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(image_score=float(raw_score), anomaly_map=amap)
