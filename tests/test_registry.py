import pytest

from vlmab.methods.baseline import IntensityBaseline
from vlmab.methods.registry import available, build_method


def test_builds_a_known_method():
    assert isinstance(build_method("intensity_baseline"), IntensityBaseline)


def test_unknown_method_lists_the_known_ones():
    with pytest.raises(KeyError) as exc:
        build_method("winclip")           # a deferred wrapper: not registered yet
    assert "intensity_baseline" in str(exc.value)


def test_available_is_sorted_and_nonempty():
    names = available()
    assert names == sorted(names) and "intensity_baseline" in names
