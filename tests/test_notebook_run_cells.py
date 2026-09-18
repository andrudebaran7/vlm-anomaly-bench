"""Every cell that runs an evaluation must stand on its own.

Not a test of our code -- a test of the notebook as an executable document. The session
handoffs name a *subset* of cells to run (phase 0, 1.1, 3.1, then one run cell), because the
phases above are done and re-running a 40-minute fit to reach the cell below it is not free.
A run cell that inherits an import or a `dataset` from the cell above therefore NameErrors
after the setup has already been paid for -- fifteen minutes of install, Drive mount and
extraction, spent to reach a typo-shaped failure. That happened to 3b.3 and is what this test
exists to stop.

The one name a run cell may inherit is `MVTEC_AD_CATEGORIES`, which cell 3.1 defines beside
the Drive extraction that puts those categories on disk. It is in every session path that
reaches a run cell, and splitting it from the extraction would be the worse coupling.
"""
import ast
import builtins
import json
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "patchcore_colab.ipynb"

# Defined by cell 3.1, which no session that reaches a run cell can skip: it is the cell that
# extracts those categories from Drive.
INHERITABLE = {"MVTEC_AD_CATEGORIES"}


def _code_cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [(i, "".join(c["source"]))
            for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]


def _strip_magics(source: str) -> str:
    """Drop IPython's `!shell` and `%magic` lines, which are not Python."""
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith(("!", "%")))


def _free_names(source: str) -> set[str]:
    """Names the cell reads without binding them first, ignoring builtins."""
    tree = ast.parse(_strip_magics(source))
    bound, used = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, (ast.Store, ast.Del)) else used).add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.comprehension,)):
            pass  # its targets are Name/Store nodes and are caught above
    return used - bound - set(dir(builtins))


RUN_CELLS = [(i, src) for i, src in _code_cells() if "run_evaluation(" in src]


def test_the_notebook_still_has_run_cells():
    """A guard on the guard: a renamed runner would make the list below silently empty."""
    assert len(RUN_CELLS) >= 4, f"only {len(RUN_CELLS)} cells call run_evaluation()"


@pytest.mark.parametrize("index, source", RUN_CELLS, ids=lambda v: v if isinstance(v, int) else "")
def test_a_run_cell_imports_everything_it_uses(index, source):
    orphans = _free_names(source) - INHERITABLE
    assert not orphans, (
        f"notebook cell {index} calls run_evaluation() but reads {sorted(orphans)} without "
        "defining them. A session told to skip the cells above this one would NameError here, "
        "after paying for the setup. Repeat the imports in the cell."
    )
