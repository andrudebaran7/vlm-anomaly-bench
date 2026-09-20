"""A result that is not in the repository is a result nobody has.

`.gitignore` excludes `results/*`, because anomalib writes its own run tree there and the shards
and native-resolution maps are tens of gigabytes. The report files are the exception — but an
exception has to be maintained, and `git add -A` skips an ignored file **silently**, so a commit
can succeed carrying a message that describes a result it does not contain.

That is not hypothetical: on 2026-09-21 `results/mvtec_ad2/vial.md` was written, added, committed
and reported as committed, and none of it was true, because only `results/reproduction/` had been
negated. This test is the guard, and it is about the repository rather than the code.
"""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = sorted(ROOT.glob("results/**/*.md"))


def _ignored(path: Path) -> bool:
    """git check-ignore exits 0 when the path IS ignored, 1 when it is not."""
    return subprocess.run(
        ["git", "-C", str(ROOT), "check-ignore", "-q", str(path)],
    ).returncode == 0


@pytest.mark.skipif(not REPORTS, reason="no result reports on this checkout")
@pytest.mark.parametrize("report", REPORTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_a_result_report_is_not_silently_ignored(report):
    assert not _ignored(report), (
        f"{report.relative_to(ROOT)} is gitignored, so `git add -A` will skip it without a word "
        "and any commit claiming to record it will be wrong. Add a narrow negation in "
        ".gitignore — narrow, because the shards and maps beside it must stay out."
    )


def test_the_shards_and_maps_beside_a_report_stay_out():
    """The negation has to stay narrow. A whole-directory negation would sweep in the
    native-resolution anomaly maps, which are 24.9 GB across the grid."""
    for probe in ("results/mvtec_ad2/vial/shards/x.parquet",
                  "results/mvtec_ad2/vial/maps/x.npy"):
        assert _ignored(ROOT / probe), f"{probe} would be committable; it must not be"
