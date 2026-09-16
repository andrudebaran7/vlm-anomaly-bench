"""WinCLIP zero-shot adapter (Jeong et al., CVPR 2023) — CLIP with handcrafted prompt ensembles.

WinCLIP's substance is the anomalib call: a pre-trained CLIP scores each image against text
embeddings of normal/anomalous state prompts, with multi-scale windows for localisation. That
needs anomalib, CLIP weights and a GPU, so it lives behind an injected backend written on Colab
(the WinCLIP Colab phases in docs/superpowers/plans/2026-07-24-winclip-zeroshot-adapter.md). Everything here — passing the per-category class name into
the prompts, the native-resolution map, the raw pass-through — is tested with a fake backend.

WinCLIP is zero-shot: no fit, no training. The category is passed to the backend because it is the
object noun in the prompt ensemble ("a photo of a {damaged} {vial}"), taken verbatim from the
paper (protocol §3), which changes the text embeddings.

The backend is any object with:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import BackendSeeded, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class WinClipRef(BackendSeeded):
    name = "winclip"
    zero_shot = True

    def __init__(self, backend=None, seed: int | None = None):
        self._backend = backend
        # The backend is what calls seed_everything, so it is what declares the seed.
        self._init_seed(backend, seed)

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real anomalib WinCLIP
        backend is Colab-only (it needs anomalib, CLIP weights and a GPU), reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs anomalib + CLIP + GPU
            raise MethodNotRunnable(
                "WinClipRef needs an anomalib WinCLIP backend; build it in a GPU session with "
                "anomalib installed and inject it (see the WinCLIP Colab plan under docs/superpowers/plans/)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("WinClipRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
