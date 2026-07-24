"""IntensityBaseline: the honest floor.

Not a serious detector — it flags pixels whose brightness deviates from the image mean, which
catches nothing subtle. It exists so the full pipeline (method -> runner -> shard -> aggregate)
runs in CI against a real `AnomalyMethod`, and so every results table has a trivial baseline to
sit above: a zero-shot/VLM method that cannot beat "the odd bright pixel" is telling you
something. Pure numpy, no weights, no GPU.
"""
import numpy as np

from vlmab.methods.base import AnomalyMethod, Prediction
from vlmab.methods.postprocess import normalise_to_unit


class IntensityBaseline(AnomalyMethod):
    name = "intensity_baseline"
    zero_shot = True

    def prepare(self, device: str = "cuda") -> None:
        """No weights to load."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        gray = np.asarray(image, dtype=np.float32).mean(axis=2)
        deviation = np.abs(gray - gray.mean())
        amap = normalise_to_unit(deviation)
        return Prediction(image_score=float(amap.max()), anomaly_map=amap)
