"""Common method contract: everything is an AnomalyMethod.

predict() returns an image-level score plus a pixel anomaly map in [0,1],
resized to the input resolution. Efficiency instrumentation wraps predict().
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Prediction:
    image_score: float
    anomaly_map: np.ndarray  # HxW float32 in [0,1]
    extras: dict | None = None  # tokens/cost for API methods, etc.


class AnomalyMethod:
    """Adapters: winclip.py, anomalyclip.py, adaclip.py, saa.py, mllm.py, patchcore_ref.py."""

    name: str = "base"
    zero_shot: bool = True

    def prepare(self, device: str = "cuda") -> None:
        """Load weights/checkpoints. Called once."""
        raise NotImplementedError

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError
