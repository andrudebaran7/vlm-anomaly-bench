import pytest

from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.registry import available, build_method


def test_builds_a_known_method():
    assert isinstance(build_method("intensity_baseline"), IntensityBaseline)


def test_unknown_method_lists_the_known_ones():
    with pytest.raises(KeyError) as exc:
        # a name that is deliberately not a method and never will be: every planned method is now
        # registered, so borrowing a real one would make this test fail as each plan lands.
        build_method("not_a_method_sentinel")
    assert "intensity_baseline" in str(exc.value)


def test_available_is_sorted_and_nonempty():
    names = available()
    assert names == sorted(names) and "intensity_baseline" in names


def test_builds_the_patchcore_anchor():
    from vlmab.methods.patchcore_ref import PatchCoreRef

    assert isinstance(build_method("patchcore_ref"), PatchCoreRef)
    assert "patchcore_ref" in available()


def test_builds_winclip():
    from vlmab.methods.winclip import WinClipRef

    assert isinstance(build_method("winclip"), WinClipRef)
    assert "winclip" in available()


def test_builds_anomalyclip():
    from vlmab.methods.anomalyclip import AnomalyClipRef

    assert isinstance(build_method("anomalyclip"), AnomalyClipRef)
    assert "anomalyclip" in available()


def test_builds_adaclip():
    from vlmab.methods.adaclip import AdaClipRef

    assert isinstance(build_method("adaclip"), AdaClipRef)
    assert "adaclip" in available()


def test_builds_saa():
    from vlmab.methods.saa import SaaRef

    assert isinstance(build_method("saa"), SaaRef)
    assert "saa" in available()


def test_every_planned_method_is_registered():
    """Pins the full set of currently-registered method names, so an accidental removal fails. This is
    not every method the project ever plans to run — WinCLIP+ (few-shot) is noted as a separate plan in
    configs/methods/winclip.yaml and listed in README.md's methods table — only every method with a
    landed CPU-adapter plan, which as of SAA+ is all seven of these."""
    assert set(available()) == {
        "adaclip",
        "anomalyclip",
        "intensity_baseline",
        "mllm_qwen",
        "patchcore_ref",
        "saa",
        "winclip",
    }


def test_build_method_forwards_the_seed_to_the_adapter():
    from vlmab.methods.patchcore_ref import PatchCoreRef

    method = build_method("patchcore_ref", seed=3)
    assert isinstance(method, PatchCoreRef) and method.seed == 3


def test_build_method_lets_a_deterministic_method_refuse_a_seed():
    """The registry does not know which methods are stochastic and must not decide: it forwards,
    and the adapter that knows raises."""
    with pytest.raises(ValueError, match="deterministic"):
        build_method("intensity_baseline", seed=0)


def test_every_registered_method_builds_without_a_seed():
    for name in available():
        assert build_method(name).seed is None
