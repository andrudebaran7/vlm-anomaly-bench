"""patchcore_ref adapter. TODO(M2/M3): wrap official implementation (pinned commit) — see docs/protocol.md §3."""
from .base import AnomalyMethod, Prediction


class Patchcore_ref(AnomalyMethod):
    name = "patchcore_ref"
    # TODO: prepare(), predict()
