"""saa adapter. TODO(M2/M3): wrap official implementation (pinned commit) — see docs/protocol.md §3."""
from .base import AnomalyMethod, Prediction


class Saa(AnomalyMethod):
    name = "saa"
    # TODO: prepare(), predict()
