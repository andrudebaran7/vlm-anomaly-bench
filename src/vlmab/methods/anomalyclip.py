"""anomalyclip adapter. TODO(M2/M3): wrap official implementation (pinned commit) — see docs/protocol.md §3."""
from .base import AnomalyMethod, Prediction


class Anomalyclip(AnomalyMethod):
    name = "anomalyclip"
    # TODO: prepare(), predict()
