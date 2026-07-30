"""Name -> adapter, for the CLI and the Colab notebook.

Adapters are buildable by name here even when they need a GPU backend they cannot construct on
CPU (mllm_qwen, patchcore_ref, winclip, anomalyclip, adaclip, saa): they build, but their
prepare() raises MethodNotRunnable without an injected backend. As of the SAA+ plan every planned
method is registered, so an unknown name fails with the list of what actually runs rather than a
promise.
"""
from typing import Callable

from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.base import AnomalyMethod
from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.mllm import QwenMLLM
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.saa import SaaRef
from vlmab.methods.winclip import WinClipRef

_REGISTRY: dict[str, Callable[[], AnomalyMethod]] = {
    "adaclip": AdaClipRef,          # CPU-usable only with an injected backend; prepare() gates the rest
    "anomalyclip": AnomalyClipRef,  # CPU-usable only with an injected backend; prepare() gates the rest
    "intensity_baseline": IntensityBaseline,
    "mllm_qwen": QwenMLLM,          # CPU-usable only with an injected client; prepare() gates the rest
    "patchcore_ref": PatchCoreRef,  # CPU-usable only with an injected backend; prepare() gates the rest
    "saa": SaaRef,                  # CPU-usable only with an injected backend; prepare() gates the rest
    "winclip": WinClipRef,          # CPU-usable only with an injected backend; prepare() gates the rest
}


def available() -> list[str]:
    return sorted(_REGISTRY)


def build_method(name: str) -> AnomalyMethod:
    if name not in _REGISTRY:
        raise KeyError(f"unknown method {name!r}; available: {available()}")
    return _REGISTRY[name]()
