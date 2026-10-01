"""A run that produced shards and no report is a result nobody has.

Fruit Jelly ran on 2026-09-20 at commit a7bc7bd and its numbers existed only on a Drive folder
until 2026-09-29 — nine days in which the study did not know it had them. `write_report` closes
the common case; this closes the one where a session died between the shards and the summary.

Shards are gitignored, so this guard necessarily runs against the working tree rather than the
repository. On a clean checkout it finds nothing and skips, which is correct: there is nothing to
have lost.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "results" / "mvtec_ad2"
SHARD_DIRS = sorted(p for p in GRID.glob("*/*/shards") if any(p.glob("*.parquet")))


@pytest.mark.skipif(not SHARD_DIRS, reason="no shards in this working tree")
@pytest.mark.parametrize("shards", SHARD_DIRS, ids=lambda p: str(p.relative_to(GRID)))
def test_a_shard_directory_has_a_report_beside_it(shards):
    category_dir = shards.parent
    report = category_dir.parent / f"{category_dir.name}.md"
    assert report.exists(), (
        f"{shards.relative_to(ROOT)} holds results and {report.relative_to(ROOT)} does not "
        "exist. Run `--summarise-only` for that category before the session ends: a result that "
        "only lives in a shard is one nobody has, which cost nine days on 2026-09-20."
    )
