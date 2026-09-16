"""The seed contract: a method declares the seed it applied, and nothing else may claim one.

`seed` never means "the seed I was asked for". A method that seeds nothing declares None, which
is why None is the default rather than 0 — recording 0 for a method that never seeded anything is
the exact falsehood this contract exists to remove.
"""
import pytest

from vlmab.methods.base import AnomalyMethod, BackendSeeded
from vlmab.methods.baseline import IntensityBaseline


class _Backend:
    """A backend that applies a seed and says so, like PatchCoreBackend."""

    def __init__(self, seed=None):
        self.seed = seed


class _SeedlessBackend:
    """A backend from before seeds existed — e.g. the fake backends the adapter tests inject.
    It must declare None rather than raise AttributeError."""


class _Adapter(BackendSeeded):
    name = "adapter"

    def __init__(self, backend=None, seed=None):
        self._init_seed(backend, seed)


def test_a_method_declares_no_seed_by_default():
    assert AnomalyMethod.seed is None


def test_an_adapter_declares_the_seed_its_backend_applied():
    assert _Adapter(backend=_Backend(seed=7)).seed == 7


def test_an_adapter_with_a_seedless_backend_declares_none():
    assert _Adapter(backend=_SeedlessBackend()).seed is None


def test_an_adapter_keeps_its_own_seed_when_no_backend_is_injected():
    assert _Adapter(seed=3).seed == 3


def test_a_seed_that_disagrees_with_the_backend_is_refused():
    with pytest.raises(ValueError, match="disagree"):
        _Adapter(backend=_Backend(seed=0), seed=1)


def test_a_seed_matching_the_backend_is_accepted():
    assert _Adapter(backend=_Backend(seed=1), seed=1).seed == 1


def test_the_deterministic_baseline_declares_no_seed():
    assert IntensityBaseline().seed is None


def test_the_deterministic_baseline_refuses_a_seed():
    with pytest.raises(ValueError, match="deterministic"):
        IntensityBaseline(seed=0)


from vlmab.methods.adaclip import AdaClipRef
from vlmab.methods.anomalyclip import AnomalyClipRef
from vlmab.methods.mllm import QwenMLLM
from vlmab.methods.patchcore_ref import PatchCoreRef
from vlmab.methods.saa import SaaRef
from vlmab.methods.winclip import WinClipRef

#: Every adapter whose stochasticity will live in an injected backend. They are listed
#: explicitly rather than discovered, so an adapter added without the seed contract fails here
#: instead of silently opting out of it.
_BACKEND_ADAPTERS = [AdaClipRef, AnomalyClipRef, PatchCoreRef, QwenMLLM, SaaRef, WinClipRef]


@pytest.mark.parametrize("cls", _BACKEND_ADAPTERS, ids=lambda c: c.__name__)
def test_every_backend_adapter_accepts_a_seed_and_declares_it(cls):
    assert cls(seed=5).seed == 5


@pytest.mark.parametrize("cls", _BACKEND_ADAPTERS, ids=lambda c: c.__name__)
def test_every_backend_adapter_declares_none_by_default(cls):
    assert cls().seed is None
