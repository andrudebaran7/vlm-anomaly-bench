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
