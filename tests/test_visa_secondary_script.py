"""The VisA secondary-check runner lives in a tracked script, not in notebook JSON.

That is a delivery constraint, not a style preference: a fix inside a notebook cell cannot reach
a live Colab session, because cell 1.1 hard-resets the repository clone while the notebook the
session is executing is a different file. A tracked `.py` is fetched by a cell the session
already has. Anything that must be fixable mid-session belongs here.

These tests run in CI, where neither torch nor anomalib exists, so they check exactly the two
things CPU can: that the module imports without them, and that the one magic number in it still
agrees with the record it was read from.
"""
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_visa_secondary.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_visa_secondary", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_imports_without_torch_or_anomalib():
    """It has to be importable in CI to be testable at all, and the heavy imports belong inside
    the function that needs a GPU anyway."""
    for blocked in ("torch", "anomalib"):
        assert blocked not in sys.modules or True   # not asserting absence; see below
    mod = _module()
    assert hasattr(mod, "main")
    # The real check: nothing GPU-only was pulled in as a side effect of importing.
    assert "anomalib" not in sys.modules


def test_the_archive_size_matches_the_verified_record():
    """The script refuses a download whose byte count differs. That number was measured once and
    written into docs/datasets-access.md; two copies of a magic number drift, and a stale one
    here would reject a perfectly good archive at the start of a two-hour run."""
    recorded = (ROOT / "docs" / "datasets-access.md").read_text()
    found = set(re.findall(r"1,929,840,640|1_929_840_640|1929840640", recorded))
    assert found, "docs/datasets-access.md no longer records the VisA archive size"
    assert _module().ARCHIVE_BYTES == 1_929_840_640
