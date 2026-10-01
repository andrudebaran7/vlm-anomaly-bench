"""Attach a GPU backend to a CPU adapter, lazily, one builder per method.

`registry.build_method` deliberately returns an adapter with NO backend: that is what lets CI
construct every method on a machine with no torch. This module is the other half — the place a
backend is wired on — and so every import in it is function-local. A module-scope `import torch`
here would break both CI jobs, which is why a test asserts their absence rather than trusting
review.

A method in the registry without a builder here is refused BY NAME, before any archive is
fetched or any image is read. The four unbenchmarked methods (protocol v0.2.18) are exactly that
case, and an operator who types one should be told so in a millisecond rather than after a
download.
"""
from typing import Callable

from vlmab.methods.base import AnomalyMethod

#: Pinned 2026-09-17: `classic` was tested and refuted, and every number in this repo comes from
#: the anomalib transform. The factory does not expose it as a parameter — the probe does.
PREPROCESS = "anomalib"


def _patchcore(seed: int | None) -> AnomalyMethod:
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    return PatchCoreRef(backend=PatchCoreBackend(seed=seed, preprocess=PREPROCESS))


def _winclip(seed: int | None) -> AnomalyMethod:
    from vlmab.methods.winclip import WinClipRef
    from vlmab.methods.winclip_backend import WinClipBackend

    return WinClipRef(backend=WinClipBackend(seed=seed))


_GPU_BUILDERS: dict[str, Callable[[int | None], AnomalyMethod]] = {
    "patchcore_ref": _patchcore,
    "winclip": _winclip,
}


def gpu_builders() -> list[str]:
    """Method names that can be built with a GPU backend."""
    return sorted(_GPU_BUILDERS)


def build_runnable(name: str, seed: int | None = None) -> AnomalyMethod:
    """An adapter wired to its GPU backend, ready for `prepare()`.

    Raises KeyError naming the available builders. The import of the backend happens inside the
    builder, so a wrong name costs nothing and a right one costs the import only once.
    """
    if name not in _GPU_BUILDERS:
        raise KeyError(
            f"no GPU backend for {name!r}; methods with one: {gpu_builders()}. "
            "The four methods scoped out in protocol v0.2.18 keep their adapters and gates but "
            "have no backend, deliberately."
        )
    return _GPU_BUILDERS[name](seed)
