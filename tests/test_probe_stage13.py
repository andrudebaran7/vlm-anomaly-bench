"""Stage 13 must be runnable ALONE, because that is the only way it is ever run.

Notebook cell 3b.0 calls `probe_patchcore.run(only="13. the classic pre-processor")`, and
`only=` skips every other stage — including stage 2, which is what normally fills `ctx` with
the synthetic images. A stage that silently depended on stage 2 would fail with a KeyError
that reads like "the stage was renamed", and the one check standing between the author and a
40-minute wrong run would be the thing that broke.

Importing the probe never imports anomalib, so this runs in CI. The stage cannot COMPLETE here
(constructing the backend needs anomalib and a GPU) — what is checked is that it gets past its
own setup, which is the part that was broken.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))


def test_stage_13_builds_its_own_images_when_run_alone(capsys):
    import probe_patchcore

    ctx = probe_patchcore.run(only="13. the classic pre-processor")
    assert "images" in ctx and len(ctx["images"]) == 20, (
        "stage 13 did not reach the point of having images to fit on; run(only=...) skips "
        "stage 2, so it has to build them itself"
    )
    assert "KeyError" not in capsys.readouterr().out
