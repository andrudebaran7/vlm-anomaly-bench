"""Stage 15 (the v0.2.21 guard) must be runnable ALONE, for the same reason as stage 14.

The notebook cell that runs it calls `run(only=...)`, which skips stage 2. CI has no anomalib,
so the stage cannot complete here; what is checked is that it gets past its own setup.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

STAGE = "15. score() resizes with antialiasing (v0.2.21 guard)"
STAGE14 = "14. which resize score() applies (H1/H2, check b)"


def test_stage_15_is_registered_under_the_name_the_notebook_uses():
    import probe_patchcore

    assert STAGE in [name for name, _, _ in probe_patchcore._STAGES]


def test_stage_15_builds_its_own_images_when_run_alone(capsys):
    import probe_patchcore

    ctx = probe_patchcore.run(only=STAGE)
    assert "images" in ctx and len(ctx["images"]) == 20
    assert "KeyError" not in capsys.readouterr().out


def test_superseded_stages_13_and_14_skip_but_still_build_images(capsys):
    import probe_patchcore

    for name in ("13. the classic pre-processor", STAGE14):
        ctx = probe_patchcore.run(only=name)
        out = capsys.readouterr().out
        assert len(ctx["images"]) == 20
        assert "PASS  skipped — superseded by protocol v0.2.21 (classic refused)" in out
        assert ctx["_failed_at"] is None


# --- M1 (2026-10-19 whole-branch review): the guard must be able to fail ----------------------
# After v0.2.21 routes export_transform to transform, "export path == train path" holds by
# construction, so a PASS proved nothing about antialiasing. Stage 15 now also scores through an
# antialias-OFF copy of the transform and asserts it DIFFERS from the train path. The copy is
# built by `_antialias_off_copy`, tested here with fake transform classes (CI has no torchvision).


class Resize:  # named as torchvision names it: the helper matches on type name
    def __init__(self, size, interpolation="bilinear", antialias=True):
        self.size, self.interpolation, self.antialias = size, interpolation, antialias


class Normalize:
    pass


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)


def test_the_antialias_off_copy_keeps_size_interpolation_and_normalize():
    import probe_patchcore

    norm = Normalize()
    original = Compose([Resize([256, 256], interpolation="bicubic", antialias=True), norm])
    off = probe_patchcore._antialias_off_copy(original, Compose, Resize)

    resize, normalize = off.transforms
    assert isinstance(resize, Resize)
    assert resize.size == [256, 256] and resize.interpolation == "bicubic"
    assert resize.antialias is False
    assert normalize is norm, "the transform's own Normalize must be carried over"
    assert original.transforms[0].antialias is True, "the model's transform must not be mutated"


def test_the_antialias_off_copy_refuses_a_transform_without_resize_or_normalize():
    import probe_patchcore
    import pytest

    with pytest.raises(RuntimeError, match="Resize"):
        probe_patchcore._antialias_off_copy(Compose([Normalize()]), Compose, Resize)
    with pytest.raises(RuntimeError, match="Normalize"):
        probe_patchcore._antialias_off_copy(Compose([Resize([256, 256])]), Compose, Resize)


def test_stage_15_labels_the_routed_export_path_honestly_and_discriminates():
    import inspect

    import probe_patchcore

    src = inspect.getsource(probe_patchcore._antialiased_guard)
    assert "export_transform (now routed to transform)" in src
    assert "_antialias_off_copy(" in src
    assert "torchvision.transforms.v2" in src
    assert "not same(off, trn)" in src
