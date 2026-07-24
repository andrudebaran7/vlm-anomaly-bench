"""IntensityBaseline: the honest floor.

Not a serious detector — it flags pixels whose brightness deviates from the image mean, which
catches nothing subtle. It exists so the full pipeline (method -> runner -> shard -> aggregate)
runs in CI against a real `AnomalyMethod`, and so every results table has a trivial baseline to
sit above: a zero-shot/VLM method that cannot beat "the odd bright pixel" is telling you
something. Pure numpy, no weights, no GPU.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction


class IntensityBaseline(AnomalyMethod):
    name = "intensity_baseline"
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        """No weights to load."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)
        deviation = np.abs(gray - gray.mean()).astype(np.float32)
        # Raw deviation, NOT normalised to [0,1]: per-image rescaling would make every varied
        # image's score identical (the old bug: I-AUROC 0.5 by ties) and break the cross-image
        # ranking the pixel metrics depend on (protocol v0.2.6).
        return Prediction(image_score=float(deviation.max()), anomaly_map=deviation)
