import pytest

from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.registry import available, build_method


def test_builds_a_known_method():
    assert isinstance(build_method("intensity_baseline"), IntensityBaseline)


def test_unknown_method_lists_the_known_ones():
    with pytest.raises(KeyError) as exc:
        # a deferred wrapper, not registered yet — swap for another unregistered name
        # when saa itself gets registered, or this test starts failing for the wrong reason.
        build_method("saa")
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
