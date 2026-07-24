"""Name -> adapter, for the CLI and the Colab notebook.

Only GPU-free adapters are registered here today. Each deferred model wrapper (WinCLIP,
AnomalyCLIP, AdaCLIP, SAA+, PatchCore) adds its own entry when its plan lands, so an unknown
name fails with the list of what actually runs rather than a promise.
"""
from typing import Callable

from vlmab.methods.base import AnomalyMethod
from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.mllm import QwenMLLM

_REGISTRY: dict[str, Callable[[], AnomalyMethod]] = {
    "intensity_baseline": IntensityBaseline,
    "mllm_qwen": QwenMLLM,   # CPU-usable only with an injected client; prepare() gates the rest
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_method(name: str) -> AnomalyMethod:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    return _REGISTRY[name]()
