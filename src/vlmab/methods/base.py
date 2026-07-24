"""Common method contract: everything is an AnomalyMethod.

predict() returns an image-level score plus a pixel anomaly map at the input resolution, on the
method's own consistent scale (NOT per-image normalised — protocol v0.2.6). Efficiency
instrumentation wraps predict(). Full-shot methods build per-category state in fit().
"""
from dataclasses import dataclass
from typing import Iterable
import numpy as np


class MethodNotRunnable(RuntimeError):
    """Raised by an adapter that cannot run in the current environment (e.g. a model
    wrapper with no weights/GPU/client available). Distinct from a genuine runtime error
    inside predict(), so callers can surface real bugs while handling this cleanly."""


@dataclass
class Prediction:
    image_score: float
    anomaly_map: np.ndarray  # HxW float32, native resolution, method's own consistent scale
    extras: dict | None = None  # tokens/cost for API methods, etc.


class AnomalyMethod:
    """Adapters: winclip.py, anomalyclip.py, adaclip.py, saa.py, mllm.py, patchcore_ref.py."""

    name: str = "base"
    zero_shot: bool = True

    def prepare(self, device: str = "cuda") -> None:
        """Load weights/checkpoints. Called once."""
        raise NotImplementedError

    def fit(self, train_images: Iterable[np.ndarray], category: str) -> None:
        """Build per-category state (e.g. a memory bank) from defect-free training images.

        The runner calls this once per category, before scoring that category, and only when
        `zero_shot` is False. Zero-shot methods do not override it — it is a no-op for them."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError
