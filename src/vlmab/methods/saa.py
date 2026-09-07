"""SAA+ training-free adapter (Cao et al., "Segment Any Anomaly without Training").

SAA+ is a cascade of two frozen foundation models: GroundingDINO proposes regions from defect
language prompts, SAM refines them into masks. Nothing is trained by the method, so — unlike
AnomalyCLIP and AdaCLIP — there is no auxiliary-training overlap audit. Two consequences shape this
adapter:

1. The category is substantive. SAA+'s contribution is hybrid prompt regularization: per-object domain
   knowledge (defect language expressions, object-specific area/size constraints). The category
   selects those prompts, so the seam takes it. Those prompts are taken VERBATIM from the official
   repo at the pinned commit (configs/methods/saa_prompts.yaml) — protocol §3 amendment v0.2.9, which
   exempts SAA+ from the no-per-category-prompts rule precisely because they are the method, not a
   tuning knob of ours.

2. The map is not dense. SAA+ emits region masks with confidence scores, so its anomaly map has very
   few distinct levels. We take the repo's OWN final map verbatim and only upsample it: compositing
   masks ourselves would be a re-implementation (protocol §3 priority 3, flagged), and smoothing it
   would be a post-process no other method in the study receives. Map granularity is reported instead,
   via postprocess.distinct_levels, and carried as a stated limitation in the paper's §5.2.

The backend is any object with:
    score(image: np.ndarray, category: str) -> tuple[float, np.ndarray]   # (raw score, raw map)

Scores and maps are RAW (protocol v0.2.6): passed through and only upsampled to native resolution.
"""
import numpy as np

from vlmab.methods.base import BackendSeeded, MethodNotRunnable, Prediction
from vlmab.methods.postprocess import upsample_to


class SaaRef(BackendSeeded):
    name = "saa"
    zero_shot = True

    def __init__(self, backend=None, seed: int | None = None):
        self._backend = backend
        # The backend is what calls seed_everything, so it is what declares the seed.
        self._init_seed(backend, seed)

    def prepare(self, device: str = "cuda") -> None:
        """With an injected backend there is nothing to load. Otherwise the real cascade is
        Colab-only (it needs the official repo plus GroundingDINO and SAM weights and a GPU),
        reported cleanly."""
        if self._backend is None:  # pragma: no cover - needs the repo + two checkpoints + GPU
            raise MethodNotRunnable(
                "SaaRef needs an SAA+ backend; build it in a GPU session from the official repo at "
                "the pinned commit, with both the GroundingDINO and SAM checkpoints, and inject it "
                "(see the SAA+ Colab plan)"
            )

    def predict(self, image: np.ndarray, category: str) -> Prediction:
        if self._backend is None:  # pragma: no cover
            raise MethodNotRunnable("SaaRef has no backend; inject one or build it on GPU")
        raw_score, raw_map = self._backend.score(image, category)
        amap = upsample_to(np.asarray(raw_map, dtype=np.float32), image.shape[:2])
        return Prediction(
            image_score=float(raw_score),
            anomaly_map=amap,
            extras={"category": category},
        )
