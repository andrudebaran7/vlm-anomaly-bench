"""adaclip adapter. TODO(M2/M3): wrap official implementation (pinned commit) — see docs/protocol.md §3."""
from .base import AnomalyMethod, Prediction


class Adaclip(AnomalyMethod):
    name = "adaclip"
    # TODO: prepare(), predict()
