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
    #: The seed this method applied to its own stochasticity, or None if it applied none.
    #
    # NOT "the seed the caller asked for". `run_evaluation` stamps this value into every shard's
    # provenance, so a method that declares a seed it did not apply produces a record that reads
    # true and is not — which is worse than no record. A deterministic method declares None and
    # refuses a seed; a method whose stochasticity lives in a backend declares the backend's.
    seed: int | None = None

    def prepare(self, device: str = "cuda") -> None:
        """Load weights/checkpoints. Called once."""
        raise NotImplementedError

    def fit(self, train_images: Iterable[np.ndarray], category: str) -> None:
        """Build per-category state (e.g. a memory bank) from defect-free training images.

        The runner calls this once per category, before scoring that category, and only when
        `zero_shot` is False. Zero-shot methods do not override it — it is a no-op for them."""

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        raise NotImplementedError


class BackendSeeded(AnomalyMethod):
    """An adapter whose stochasticity lives in an injected backend.

    The backend is what calls into the library that samples, so the backend is what applies the
    seed and therefore what declares it. The adapter only passes the declaration on.

    Call `_init_seed(backend, seed)` from the adapter's `__init__`.
    """

    _declared_seed: int | None = None

    def _init_seed(self, backend, seed: int | None) -> None:
        if backend is None:
            # Nothing has been constructed that could apply anything, and prepare() will raise
            # MethodNotRunnable before this method scores a sample. Keep what the caller asked
            # for so a backend built from it later has something to be checked against.
            self._declared_seed = seed
            return
        # getattr, not attribute access: the adapter tests inject fake backends written before
        # seeds existed, and those declare None rather than exploding.
        backend_seed = getattr(backend, "seed", None)
        if seed is not None and seed != backend_seed:
            raise ValueError(
                f"{type(self).__name__} was given seed={seed!r} but its backend applies "
                f"seed={backend_seed!r}; the two disagree and this adapter cannot make the "
                "backend use the other one. Seed the backend at construction and leave this "
                "argument out."
            )
        self._declared_seed = backend_seed

    @property
    def seed(self) -> int | None:
        return self._declared_seed
