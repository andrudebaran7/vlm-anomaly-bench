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


# --- The VisA session path -----------------------------------------------------------------
# Its handoff is phase 0 -> 1.1 -> 3.4 -> 3.5, and it skips cell 3.1 entirely because VisA needs
# no Drive upload. So these cells may not inherit even MVTEC_AD_CATEGORIES -- the one name the
# rule above allows. They are checked separately and more strictly for that reason.

VISA_CELLS = [(i, src) for i, src in _code_cells()
              if "run_visa_secondary" in src or "reproduction_visa" in src]


def test_the_visa_session_path_still_has_its_cells():
    """A guard on the guard, as above: a renamed script or folder would empty this list and the
    check below would pass by vacuity."""
    assert len(VISA_CELLS) == 2, (
        f"expected cells 3.4 and 3.5, found {[i for i, _ in VISA_CELLS]}"
    )


@pytest.mark.parametrize("index, source", VISA_CELLS, ids=lambda v: v if isinstance(v, int) else "")
def test_a_visa_cell_imports_everything_it_uses(index, source):
    orphans = _free_names(source)
    assert not orphans, (
        f"notebook cell {index} is on the VisA session path and reads {sorted(orphans)} without "
        "defining them. That path skips cell 3.1, so it cannot inherit even "
        "MVTEC_AD_CATEGORIES. Repeat the imports in the cell."
    )


def test_the_visa_result_is_copied_off_the_runtime_before_anything_can_delete_it():
    """Cell 1.1 does `git reset --hard`, and the Colab filesystem dies with the runtime. The
    handoff calls a result left in a closed session's scrollback the same as no result, so the
    notebook has to carry the step that gets it out -- not just the step that produces it."""
    copy_cells = [src for _, src in VISA_CELLS if "reproduction_visa" in src]
    assert len(copy_cells) == 1
    src = copy_cells[0]
    assert "drive.mount" in src, "the copy step has to mount Drive itself"
    assert "patchcore_visa.md" in src and "shards" in src, (
        "both the report and the shards go to Drive: the shards are what let the gate be "
        "re-scored without repeating the two-hour fit"
    )
    assert "os.path.isfile(REPORT)" in src, (
        "an absent report means the gate refused, which is a different problem from a FAIL and "
        "must not be copied over in silence"
    )


# --- The M3 session path -------------------------------------------------------------------
# Phase 0 -> 1.1 -> the grid cell, skipping everything else, so it inherits nothing at all.

M3_CELLS = [(i, src) for i, src in _code_cells() if "run_mvtec_ad2" in src]


def test_the_m3_grid_cell_exists():
    assert len(M3_CELLS) == 1, f"expected one M3 cell, found {[i for i, _ in M3_CELLS]}"


@pytest.mark.parametrize("index, source", M3_CELLS, ids=lambda v: v if isinstance(v, int) else "")
def test_the_m3_cell_imports_everything_it_uses(index, source):
    orphans = _free_names(source)
    assert not orphans, (
        f"notebook cell {index} runs the M3 grid and reads {sorted(orphans)} without defining "
        "them. Its session path skips every cell between 1.1 and it."
    )


def test_the_m3_cell_names_the_category_rather_than_hardcoding_it_in_the_shell_line():
    """Eight categories go through this one cell. A category spelled into the `!python` line
    directly is one that has to be edited inside a shell string, which is where a typo stops
    looking like a typo."""
    source = M3_CELLS[0][1]
    assert "CATEGORY =" in source
    assert "--category {CATEGORY}" in source
