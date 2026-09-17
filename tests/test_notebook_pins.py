"""The Colab notebook and the frozen config must agree about the anomalib version.

Not a test of our code — a test that two files stating the same fact cannot drift apart. They
are read months apart by different things (a human running cells, a reader auditing provenance)
and nothing else would notice. The five ways anomalib 2.6.0 contradicts its own documentation
are specific to 2.6.0; a session that installed anything else would hit different ones and
record the wrong version as the one its numbers came from.
"""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "patchcore_colab.ipynb"
METHOD_CFG = ROOT / "configs" / "methods" / "patchcore_ref.yaml"


def _install_lines():
    nb = json.loads(NOTEBOOK.read_text())
    return [line
            for cell in nb["cells"] if cell["cell_type"] == "code"
            for line in cell["source"]
            if "pip install" in line and "anomalib" in line]


def test_the_notebook_installs_the_exact_version_the_config_freezes():
    pinned = yaml.safe_load(METHOD_CFG.read_text())["anomalib_version"]
    lines = _install_lines()
    assert lines, "no anomalib install line found in the notebook"
    for line in lines:
        assert f"anomalib=={pinned}" in line, (
            f"the notebook installs `{line.strip()}` while the config freezes {pinned!r}. A "
            "range here means a fresh runtime can resolve to a version none of the recorded "
            "API findings apply to."
        )
