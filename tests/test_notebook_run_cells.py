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
    """Drop IPython's `!shell` and `%magic` lines, which are not Python, along with any
    backslash-continued lines that belong to them."""
    kept, continuing = [], False
    for line in source.splitlines():
        if continuing or line.lstrip().startswith(("!", "%")):
            continuing = line.rstrip().endswith("\\")
            continue
        kept.append(line)
    return "\n".join(kept)


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
    assert "patchcore_ref_visa.md" in src and "shards" in src, (
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


def _phase6_start() -> int:
    nb = json.loads(NOTEBOOK.read_text())
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("## Phase 6"):
            return i
    raise AssertionError("the notebook has no '## Phase 6' markdown cell")


# Phase 6 (v0.2.21) is its own session path: phase 0 -> 1.1 -> 6.0, so it inherits nothing, not
# even MVTEC_AD_CATEGORIES.
PHASE6_CELLS = [(i, src) for i, src in _code_cells() if i > _phase6_start()]
# Cell 52 (phase 5) is the original grid cell; phase 6 re-uses it in 6.4 and 6.5.
M3_PHASE5 = [(i, src) for i, src in M3_CELLS if i < _phase6_start()]


def test_the_m3_grid_cell_exists():
    assert len(M3_PHASE5) == 1, f"expected one phase-5 M3 cell, found {[i for i, _ in M3_PHASE5]}"


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
    source = M3_PHASE5[0][1]
    assert "CATEGORY =" in source
    assert "--category {CATEGORY}" in source


# --- Phase 6: the v0.2.21 corrected-anchor session ---------------------------------------------

def test_the_v0221_session_cells_exist():
    joined = "\n".join(src for _, src in PHASE6_CELLS)
    assert "15. score() resizes with antialiasing" in joined
    assert "patchcore_mvtec_ad_v0.2.21.md" in joined
    assert "--summarise-only" in joined


def _cell_with(text):
    hits = [src for _, src in PHASE6_CELLS if text in src]
    assert len(hits) == 1, f"expected one phase-6 cell with {text!r}, found {len(hits)}"
    return hits[0]


def test_the_v0221_gate_cell_writes_to_the_new_report_not_the_old():
    for i, src in PHASE6_CELLS:
        assert "--out results/reproduction/patchcore_mvtec_ad.md" not in src, f"cell {i}"
    gate = _cell_with("scripts/reproduction_gate.py")
    assert "--out results/reproduction/patchcore_mvtec_ad_v0.2.21.md" in gate


def test_6_0_asserts_the_marker_stage_15_sets():
    assert 'ctx.get("v0221_guard") is True' in _cell_with("probe_patchcore.run(")
    import sys
    sys.path.insert(0, str(NOTEBOOK.parent))
    import inspect
    import probe_patchcore
    stage = [fn for name, _, fn in probe_patchcore._STAGES if name.startswith("15.")][0]
    assert 'ctx["v0221_guard"] = True' in inspect.getsource(stage)


def test_6_2_runs_one_seed_at_a_time_and_copies_each_to_drive_before_the_next():
    cell = _cell_with("scripts/run_mvtec_ad.py")
    assert "drive.mount" in cell
    assert "for seed in (0, 1, 2)" in cell
    assert "--seeds" in cell and "returncode" in cell and "SystemExit" in cell, \
        "a failed seed must stop the cell before the gate"
    assert cell.index("for seed in") < cell.rindex("copy2") < cell.index("--all-seeds"), \
        "the Drive copy must happen inside the seed loop, before the gate"
    assert "patchcore_ref_v0.2.21" in cell and "mvtec_ad_results" in cell
    assert "never overwrit" in cell.lower() or "not overwrit" in cell.lower()


def test_6_5_resummarises_from_drive_and_survives_one_failed_category():
    cell = _cell_with("--summarise-only")
    assert "drive.mount" in cell
    assert "isdir(shards)" not in cell and "no shards on this VM" not in cell
    assert "returncode" in cell
    assert "failed" in cell


@pytest.mark.parametrize("index, source", PHASE6_CELLS, ids=lambda v: v if isinstance(v, int) else "")
def test_a_phase6_cell_imports_everything_it_uses(index, source):
    orphans = _free_names(source)
    assert not orphans, (
        f"notebook cell {index} is in phase 6 and reads {sorted(orphans)} without defining them. "
        "Its session path is phase 0 -> 1.1 -> 6.0 and skips every other phase."
    )


def test_the_v0221_gate_scores_all_three_seeds_from_the_labelled_root():
    src = "\n".join(s for _, s in PHASE6_CELLS)
    assert "--all-seeds" in src
    assert "patchcore_ref_v0.2.21/mvtec_ad/shards" in src


# --- 2026-10-19 whole-branch review: old phases cannot write v0.2.21 numbers into old files ----
# After v0.2.21 the default PatchCoreBackend IS the corrected configuration, so an old run cell
# re-run by mistake would write corrected-config numbers into the OLD roots and overwrite the
# annotated old reports (patchcore_mvtec_ad.md / _3seed.md / _classic.md, patchcore_ref_visa.md).
# Every such cell now starts with a guard.

GUARD = 'raise SystemExit("superseded by protocol v0.2.21 — use phase 6")'


def _nb_cells():
    return json.loads(NOTEBOOK.read_text())["cells"]


def _writes_old_results(src: str) -> bool:
    return ("reproduction_gate.py" in src or "run_visa_secondary" in src
            or "reproduction_visa" in src
            or ("run_evaluation(" in src and "results/reproduction" in src))


OLD_RESULT_CELLS = [(i, src) for i, src in _code_cells()
                    if i < _phase6_start() and _writes_old_results(src)]


def test_the_old_result_cells_are_the_ones_the_review_named():
    """3.2, 3.3, 3.4, 3.5, 3b.1, 3b.2, 3b.3 run and gate. A guard on the selector above."""
    assert [i for i, _ in OLD_RESULT_CELLS] == [30, 32, 34, 36, 41, 43, 46, 47]


@pytest.mark.parametrize("index, source", OLD_RESULT_CELLS,
                         ids=lambda v: v if isinstance(v, int) else "")
def test_an_old_result_cell_starts_with_the_v0221_guard(index, source):
    assert source.splitlines()[0] == GUARD, (
        f"notebook cell {index} writes PatchCore results or reports under old paths and must "
        f"start with {GUARD!r}"
    )


def test_phase_3_and_3b_markdown_say_they_are_superseded():
    cells = _nb_cells()
    phase3 = "".join(cells[27]["source"])
    assert phase3.startswith("## Phase 3 —")
    assert "Superseded by protocol v0.2.21" in phase3 and "phase 6" in phase3
    phase3b = "".join(cells[37]["source"])
    assert "Nothing in 3b runs any more" not in phase3b
    assert "SystemExit" in phase3b


def test_6_5_restores_the_gate_report_from_drive_before_asserting():
    cell = _cell_with("--summarise-only")
    assert "MyDrive/mvtec_ad_results/patchcore_ref_v0.2.21" in cell
    assert "copy2" in cell
    assert cell.index("copy2") < cell.index("assert os.path.isfile")
    assert '"run 6.2 first"' not in cell


def test_6_4_and_next_steps_owe_the_by_hand_anchor_criterion():
    cells = _nb_cells()
    md64 = next("".join(c["source"]) for c in cells
                if c["cell_type"] == "markdown" and "".join(c["source"]).startswith("### 6.4"))
    for text in (md64, (NOTEBOOK.parents[1] / "docs" / "next-steps.md").read_text()):
        assert "usable/unusable-anchor criterion (random baseline)" in " ".join(text.split())


# --- 2026-10-21: the first v0.2.21 gate reused one backend across categories ------------------
# Its shards (commit ee78196) sit at the very paths 6.2 restores and resumes from; a re-run would
# skip every seed and re-gate the defective scores. 6.1b moves them aside; 6.2 refuses them.

DEFECTIVE_COMMIT = "ee7819602ee2b75793ce86b70d8da1d90646e594"


def test_6_1b_moves_the_defective_gate_aside_and_never_deletes_it():
    cell = _cell_with("shared_backend_defect_2026-10-20")
    assert DEFECTIVE_COMMIT in cell
    assert "shutil.move" in cell and "drive.mount" in cell
    for verb in ("rmtree", "os.remove", "unlink", "rmdir"):
        assert verb not in cell, f"6.1b must move evidence, never {verb} it"
    assert "not dest.exists()" in cell, "a second move must not overwrite the first"
    assert "patchcore_mvtec_ad_v0.2.21.md" in cell, "the defective report is moved too"


def test_6_1b_comes_before_the_gate():
    idx = {text: next(i for i, src in PHASE6_CELLS if text in src)
           for text in ("shared_backend_defect_2026-10-20", "scripts/run_mvtec_ad.py")}
    assert idx["shared_backend_defect_2026-10-20"] < idx["scripts/run_mvtec_ad.py"]


def test_6_2_refuses_shards_of_the_defective_gate_before_running_a_seed():
    cell = _cell_with("scripts/run_mvtec_ad.py")
    assert DEFECTIVE_COMMIT in cell and "6.1b" in cell
    assert cell.index("restored") < cell.index(DEFECTIVE_COMMIT) < cell.index("for seed in"), \
        "the check must see the shards restored from Drive, and run before any seed"


@pytest.mark.parametrize("text", ["scripts/run_mvtec_ad.py", "--summarise-only"])
def test_6_2_and_6_5_stream_the_script_output_into_the_cell(text):
    """2026-10-20: `subprocess.run` without capture sends a child's output to the VM log, not
    the cell. 6.2 sat silent for 40 minutes a seed, and 6.5 told the author to read an output
    that was never shown."""
    cell = _cell_with(text)
    assert "Popen" in cell and "subprocess.run(" not in cell


# --- 2026-10-21: vial's v0.2.21 maps stayed on the VM that ran it ------------------------------
# Its shards resume and skip every seed, so a report regenerated elsewhere has no AU-PRO. The
# author chose to re-fit (2026-10-21): 6.4b moves the shards aside and runs vial afresh.

def test_6_4b_moves_vials_shards_aside_and_never_deletes_them():
    cell = _cell_with("no_maps_2026-10-20")
    assert "shutil.move" in cell and "drive.mount" in cell
    for verb in ("rmtree", "os.remove", "unlink", "rmdir"):
        assert verb not in cell, f"6.4b must move evidence, never {verb} it"
    assert "not dest.exists()" in cell, "a second run must not move the fresh shards over the old"
    assert "patchcore_ref_v0.2.21/vial/shards" in cell
    assert '"--category", "vial"' in cell and "Popen" in cell


def test_6_4b_sits_between_the_grid_and_the_re_summarise():
    idx = {text: next(i for i, src in PHASE6_CELLS if text in src)
           for text in ('CATEGORY = "', "no_maps_2026-10-20", "--summarise-only")}
    assert idx['CATEGORY = "'] < idx["no_maps_2026-10-20"] < idx["--summarise-only"]


def test_7_1_measures_au_pro_on_its_own_root_and_touches_no_registered_result():
    """Review #24's real-category check: one seed into results/au_pro_check, --no-save, then the
    check script. It must never write into a patchcore_ref* root or to Drive."""
    cells = [src for _, src in _code_cells() if "au_pro_exact_check.py" in src]
    assert len(cells) == 1
    cell = cells[0]
    assert "results/au_pro_check/" in cell and '"--no-save"' in cell and '"--seeds", "0"' in cell
    assert "patchcore_ref" not in cell
    assert not _free_names(cell), _free_names(cell)


def test_7_2_and_7_3_apply_the_preregistered_selection_and_touch_no_registered_result():
    cells = [src for _, src in _code_cells() if "figure_inputs.py" in src]
    assert len(cells) == 2
    model = next(c for c in cells if '"model"' in c)
    assert "results/figure_check/" in model and '"--no-save"' in model and '"--seeds", "0"' in model
    assert "--committed" in model and "patchcore_ref_v0.2.21" in model   # read, for the check
    for cell in cells:
        assert not _free_names(cell), _free_names(cell)
