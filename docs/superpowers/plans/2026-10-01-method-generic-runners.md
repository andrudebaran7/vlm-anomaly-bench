# Method-Generic Session Runners Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the three session runners work for any registered method, so WinCLIP can use the Drive/resume/guard/report machinery PatchCore's eight-category grid proved, and close the four defects M3 logged.

**Architecture:** A new `src/vlmab/methods/gpu.py` holds one lazy factory per method that wires an adapter to its GPU backend; the runners call it instead of importing `PatchCoreBackend` directly. Result paths gain a method component so two methods cannot pool into one table. `scripts/run_visa_secondary.py` is generalised the same way, and the MVTec AD classic gate — which has only ever existed as notebook cells — becomes `scripts/run_mvtec_ad.py`.

**Tech Stack:** Python 3.11/3.13, pytest, pandas/pyarrow. anomalib and torch are imported lazily and never by CI.

**Spec:** `docs/superpowers/specs/2026-10-01-short-paper-descope-design.md`

## Global Constraints

- **Item A must not touch the scoring path.** Only `scripts/*.py` and the new `src/vlmab/methods/gpu.py`. `patchcore_backend.py`, `src/vlmab/eval/runner.py`, `src/vlmab/eval/aggregate.py` and `src/vlmab/metrics/` stay byte-identical, or the regenerated maps the figures need are no longer the maps that produced the reported numbers (spec §5).
- **anomalib and torch must never be imported at module scope.** CI runs on Python 3.11 and 3.13 without either. Every GPU import goes inside a function.
- **`PREPROCESS = "anomalib"`** stays the only pre-processor any run uses; `classic` was refuted 2026-09-17.
- **`DEFAULT_SEEDS = (0, 1, 2)`** stays the default for every method (protocol §6). The existing `⚠️ N seed(s)` warning already covers fewer; no new seed logic.
- **Exit codes stay:** 0 ran, 1 a run failed, 2 could not start.
- **Protocol §6:** metrics are computed per seed and combined afterwards, never pooled. Existing guards enforce it; do not weaken them.
- Tests run with `.venv/bin/pytest`. The suite is 559 tests green at `93628d7`.
- **Task 9 is the single exception to the frozen-scoring-path rule**, and it carries a stop step
  that refuses to proceed until the figures exist. Every other task touches only `scripts/` and
  the new `src/vlmab/methods/gpu.py`.

## Review Focus

1. **A method registered for CPU but with no GPU builder** (`saa`, `adaclip`, `anomalyclip`, `mllm_qwen`): the runner must refuse by name before any extraction or fit, not fail inside a lazy import after the expensive part. → Task 1.
2. **Shards from two methods in one directory** (any pre-migration directory): `summarise` must refuse rather than average PatchCore and WinCLIP into one row. → Task 3.
3. **A shard with no `commit` column** (written before provenance existed): the new `write_report` must say "unknown" rather than raise. → Task 4.
4. **`--max-bytes` below the computed requirement**: the guard's suggestion must never print a budget smaller than the peak it just measured. → Task 5.
5. **A category directory that exists but is empty** (interrupted extraction): must exit 2, not proceed to a fit over zero images. → Task 2.

---

### Task 1: A lazy GPU-backend factory

**Files:**
- Create: `src/vlmab/methods/gpu.py`
- Test: `tests/test_gpu_factory.py`

**Interfaces:**
- Consumes: `vlmab.methods.registry.available`, the adapters `PatchCoreRef` and `WinClipRef`.
- Produces: `build_runnable(name: str, seed: int | None = None) -> AnomalyMethod` — an adapter already wired to its GPU backend, ready for `prepare()`. Raises `KeyError` for a name with no GPU builder, listing the names that have one. Tasks 2, 6 and 7 call it.

- [ ] **Step 1: Write the failing tests**

```python
"""The GPU-backend factory: one lazy builder per method, refusing by name before any cost.

The registry builds CPU adapters with no backend on purpose, so CI never imports torch. This
module is where a backend gets attached, which means every import here is function-local.
"""
import sys
from pathlib import Path

import pytest

from vlmab.methods import gpu


def test_a_method_with_no_gpu_builder_is_refused_by_name():
    with pytest.raises(KeyError) as exc:
        gpu.build_runnable("saa")
    message = str(exc.value)
    assert "saa" in message
    assert "patchcore_ref" in message and "winclip" in message, (
        "the refusal must list the names that DO have a builder, so an operator who typed one "
        "of the four unbenchmarked methods learns what is available without reading source"
    )


def test_an_unknown_name_is_refused_too():
    with pytest.raises(KeyError):
        gpu.build_runnable("not_a_method")


def test_importing_the_factory_imports_neither_torch_nor_anomalib():
    """CI runs without either. A module-scope import here would break both jobs."""
    source = (Path(gpu.__file__)).read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "torch" not in stripped and "anomalib" not in stripped, (
                f"module-scope GPU import: {stripped!r}"
            )
    assert "torch" not in sys.modules or True  # the repo may be running under a torch-bearing venv


def test_every_builder_name_is_a_registered_method():
    from vlmab.methods.registry import available
    for name in gpu.gpu_builders():
        assert name in available(), f"{name} has a GPU builder but is not in the registry"
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_gpu_factory.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlmab.methods.gpu'`

- [ ] **Step 3: Write the module**

```python
"""Attach a GPU backend to a CPU adapter, lazily, one builder per method.

`registry.build_method` deliberately returns an adapter with NO backend: that is what lets CI
construct every method on a machine with no torch. This module is the other half — the place a
backend is wired on — and so every import in it is function-local. A module-scope `import torch`
here would break both CI jobs, which is why a test asserts their absence rather than trusting
review.

A method in the registry without a builder here is refused BY NAME, before any archive is
fetched or any image is read. The four unbenchmarked methods (protocol v0.2.18) are exactly that
case, and an operator who types one should be told so in a millisecond rather than after a
download.
"""
from typing import Callable

from vlmab.methods.base import AnomalyMethod

#: Pinned 2026-09-17: `classic` was tested and refuted, and every number in this repo comes from
#: the anomalib transform. The factory does not expose it as a parameter — the probe does.
PREPROCESS = "anomalib"


def _patchcore(seed: int | None) -> AnomalyMethod:
    from vlmab.methods.patchcore_backend import PatchCoreBackend
    from vlmab.methods.patchcore_ref import PatchCoreRef

    return PatchCoreRef(backend=PatchCoreBackend(seed=seed, preprocess=PREPROCESS))


def _winclip(seed: int | None) -> AnomalyMethod:
    from vlmab.methods.winclip import WinClipRef
    from vlmab.methods.winclip_backend import WinClipBackend

    return WinClipRef(backend=WinClipBackend(seed=seed))


_GPU_BUILDERS: dict[str, Callable[[int | None], AnomalyMethod]] = {
    "patchcore_ref": _patchcore,
    "winclip": _winclip,
}


def gpu_builders() -> list[str]:
    """Method names that can be built with a GPU backend."""
    return sorted(_GPU_BUILDERS)


def build_runnable(name: str, seed: int | None = None) -> AnomalyMethod:
    """An adapter wired to its GPU backend, ready for `prepare()`.

    Raises KeyError naming the available builders. The import of the backend happens inside the
    builder, so a wrong name costs nothing and a right one costs the import only once.
    """
    if name not in _GPU_BUILDERS:
        raise KeyError(
            f"no GPU backend for {name!r}; methods with one: {gpu_builders()}. "
            "The four methods scoped out in protocol v0.2.18 keep their adapters and gates but "
            "have no backend, deliberately."
        )
    return _GPU_BUILDERS[name](seed)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_gpu_factory.py -v`
Expected: PASS, 4 tests. `test_every_builder_name_is_a_registered_method` proves `winclip` is registered even though its backend module does not exist yet — the builder is never called in these tests.

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 563 passed (559 + 4)

- [ ] **Step 6: Commit**

```bash
git add src/vlmab/methods/gpu.py tests/test_gpu_factory.py
git commit -m "feat: a lazy GPU-backend factory, refusing a backendless method by name

The registry builds CPU adapters with no backend so CI never imports torch.
This is the other half, and every import in it is function-local — asserted by
a test that reads the source rather than trusting review.

A registered method with no builder is refused by name before any archive is
fetched. That is the four methods protocol v0.2.18 scopes out, and an operator
who types one learns so in a millisecond instead of after a download."
```

---

### Task 2: `run_mvtec_ad2.py --method`, with method-scoped paths

**Files:**
- Modify: `scripts/run_mvtec_ad2.py` (docstring, `run_one_seed`, `main`'s arguments and path defaults, `verify_layout`)
- Modify: `tests/test_run_mvtec_ad2.py` (harness patches the factory instead of the backend module)
- Test: `tests/test_run_mvtec_ad2.py`

**Interfaces:**
- Consumes: `vlmab.methods.gpu.build_runnable` (Task 1).
- Produces: `--method` (default `patchcore_ref`); `default_results(method, category) -> Path` = `results/mvtec_ad2/<method>/<category>`; `default_drive_results(method, category) -> Path` = `/content/drive/MyDrive/mvtec_ad2_results/<method>/<category>`. Task 3 reads shards from the first.

**Why the paths change.** Shards land in `<results>/shards` and `summarise` loads *every* parquet in that directory. Two methods in one directory would average PatchCore and WinCLIP into one row, and the report `<category>.md` would be overwritten by whichever ran last. The shard *filenames* already carry the method, so nothing would collide on disk — the pooling would be silent, which is the failure class this repo keeps finding. PatchCore's existing Drive shards at `mvtec_ad2_results/<category>/` are **not migrated**: M3 is closed and its numbers live in committed reports, so the old location is historical. A PatchCore re-run for figures therefore re-fits rather than restoring, which is what figure regeneration wants anyway.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_methods_do_not_share_a_shard_directory(harness):
    mod = harness["module"]
    a = mod.default_results("patchcore_ref", "vial")
    b = mod.default_results("winclip", "vial")
    assert a != b
    assert a.parts[-2:] == ("patchcore_ref", "vial")
    assert b.parts[-2:] == ("winclip", "vial")


def test_drive_results_are_method_scoped_too(harness):
    mod = harness["module"]
    assert mod.default_drive_results("winclip", "vial").parts[-2:] == ("winclip", "vial")


def test_the_method_comes_from_the_factory_not_a_hardcoded_backend(harness):
    """The runner must not know PatchCore's name. Task 1's factory is the only way in."""
    built = harness["built"]
    harness["main"](["--category", "vial"])
    assert [name for name, _ in built] == ["patchcore_ref"] * 3


def test_a_method_without_a_gpu_backend_exits_2_before_extracting(harness, capsys):
    rc = harness["main"](["--category", "vial", "--method", "saa"])
    assert rc == 2
    out = capsys.readouterr()
    assert "saa" in (out.out + out.err)
    assert harness["extracted"] == [], "it must refuse before touching the archive"


def test_preprocess_is_stamped_only_when_the_method_has_one(harness):
    """WinCLIP has no pre-processing identity. A null column records that; a fabricated
    "anomalib" would claim a property the method does not have."""
    harness["main"](["--category", "vial", "--method", "winclip"])
    assert harness["calls"][0]["meta"]["preprocess"] is None


def test_an_empty_category_directory_exits_2_rather_than_fitting_over_nothing(harness, capsys):
    """An interrupted extraction leaves the directory present and useless. Review Focus #5."""
    (harness["root"] / "vial").mkdir(parents=True, exist_ok=True)
    harness["layout_ok"] = False
    rc = harness["main"](["--category", "vial"])
    assert rc == 2
    assert harness["calls"] == []
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v -k "method or empty_category or preprocess_is_stamped"`
Expected: FAIL — `default_results` does not exist; `--method` is an unrecognised argument.

- [ ] **Step 3: Extend the harness fixture**

Replace the `backend_mod` patch with a factory patch, and record what was built:

```python
@pytest.fixture
def harness(monkeypatch, tmp_path):
    """A category already on disk, a fake factory, and a recorder for run_evaluation."""
    import vlmab.datasets.mvtec_ad2 as ds_mod
    import vlmab.eval.runner as runner_mod
    import vlmab.methods.gpu as gpu_mod

    calls: list[dict] = []
    built: list[tuple[str, int | None]] = []
    extracted: list[str] = []
    state = {"layout_ok": True}

    class _FakeMethod:
        def __init__(self, name, seed):
            self.name, self.seed = name, seed
            self.zero_shot = name != "patchcore_ref"
            if name == "patchcore_ref":
                self.preprocess = "anomalib"

    def _build(name, seed=None):
        if name not in ("patchcore_ref", "winclip"):
            raise KeyError(f"no GPU backend for {name!r}; methods with one: "
                           "['patchcore_ref', 'winclip']")
        built.append((name, seed))
        return _FakeMethod(name, seed)

    monkeypatch.setattr(gpu_mod, "build_runnable", _build)
    monkeypatch.setattr(ds_mod, "MVTecAD2", lambda root: types.SimpleNamespace(root=root))
    monkeypatch.setattr(runner_mod, "run_evaluation",
                        lambda *a, **kw: calls.append(kw) or [])

    mod = _module()
    root = tmp_path / "data" / "mvtec_ad2"
    (root / "vial" / "train" / "good").mkdir(parents=True)

    # The archive is "already extracted", so fetch() short-circuits; a test that wants the
    # missing-archive path removes this directory itself.
    monkeypatch.setattr(mod, "verify_layout",
                        lambda category, root: None if state["layout_ok"]
                        else (_ for _ in ()).throw(RuntimeError("layout rejected")))
    monkeypatch.setattr(mod, "copy_to_drive", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "restore_from_drive", lambda *a, **kw: None)
    monkeypatch.setattr(mod, "summarise", lambda *a, **kw: None)

    def _fetch(category, root_, archives):
        extracted.append(category)

    monkeypatch.setattr(mod, "fetch", _fetch)

    def _main(argv):
        return mod.main([*argv, "--root", str(root),
                         "--results", str(tmp_path / "out" / "vial")])

    return {"module": mod, "main": _main, "calls": calls, "built": built,
            "extracted": extracted, "root": root, **state}
```

`state["layout_ok"]` is mutated by the empty-directory test; `verify_layout` raising is what the
script already turns into exit code 2.

Keep every existing test passing: they call `harness["main"](["--category", "vial"])` with no
`--method`, which defaults to `patchcore_ref`. The two tests that assert on default paths call
`mod.default_results` directly rather than through `_main`, since `_main` pins `--results`.

- [ ] **Step 4: Change the script**

```python
#: Shards land in <results>/shards and summarise loads every parquet there, so the method has to
#: be in the path. Without it two methods pool into one row silently — the failure class this
#: repo keeps finding. PatchCore's pre-2026-10-01 Drive shards stay where they are: M3 is closed
#: and its numbers are in committed reports.
def default_results(method: str, category: str) -> Path:
    return Path("results/mvtec_ad2") / method / category


def default_drive_results(method: str, category: str) -> Path:
    return DRIVE_RESULTS / method / category
```

In `run_one_seed`, replace the two hardcoded imports with the factory and stamp `preprocess`
defensively:

```python
def run_one_seed(category: str, root: Path, results: Path, seed: int, device: str,
                 method_name: str) -> None:
    from vlmab.datasets.mvtec_ad2 import MVTecAD2
    from vlmab.eval.provenance import run_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.gpu import build_runnable

    # A fresh adapter per seed: the memory bank is what the seed changes, and reusing one would
    # carry the previous seed's bank into the next seed's scores while the shard claimed
    # otherwise. This is the defect the seed-provenance work already had to undo once.
    method = build_runnable(method_name, seed=seed)
    store = ResultStore(results / "shards")
    # `preprocess` decides a run's identity for PatchCore and does not exist for WinCLIP. None
    # records that honestly; "anomalib" would claim a property the method does not have.
    preprocess = getattr(method, "preprocess", None)
    meta = run_meta({"method": method_name, "split": "test_public", "preprocess": preprocess})
    meta = {**meta, "preprocess": preprocess}

    run_evaluation(
        MVTecAD2(root), method, store, meta,
        categories=[category], split="test_public", fit_split="train",
        maps_dir=results / "maps", device=device,
    )
    print(f"  {category} seed {seed}: done (preprocess {preprocess})")
```

In `main`, add the argument, refuse a backendless method before `fetch`, and default the paths:

```python
    parser.add_argument("--method", default="patchcore_ref",
                        help="a method with a GPU backend; see vlmab.methods.gpu")
```

and change the two path defaults from constants to `None`, so the method-scoped helpers decide:

```python
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--drive-results", type=Path, default=None)
```

Then, immediately after `args = parser.parse_args(argv)` and **before** `fetch` — a wrong method
must cost nothing:

```python
    from vlmab.methods.gpu import gpu_builders
    if args.method not in gpu_builders():
        print(f"\ncannot start: no GPU backend for {args.method!r}; "
              f"methods with one: {gpu_builders()}", file=sys.stderr)
        return 2
    results = args.results or default_results(args.method, args.category)
    drive_results = args.drive_results or default_drive_results(args.method, args.category)
```

Update the module docstring's first line to name the method argument, and keep every other
paragraph: they document decisions that have not changed.

- [ ] **Step 5: Run the runner's tests**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v`
Expected: PASS, all existing tests plus the six new ones.

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 569 passed

- [ ] **Step 7: Migrate the eight committed reports to the method-scoped layout**

`git mv` so history follows the files:

```bash
mkdir -p results/mvtec_ad2/patchcore_ref
for c in vial sheet_metal fruit_jelly wallplugs can walnuts rice fabric; do
  git mv "results/mvtec_ad2/$c.md" "results/mvtec_ad2/patchcore_ref/$c.md"
done
```

Then update `.gitignore`'s negation so the moved reports stay tracked, and confirm with the
guard that already exists:

Run: `.venv/bin/pytest tests/test_results_are_tracked.py -v`
Expected: PASS — eight reports, none gitignored, and shards/maps beside them still ignored.

- [ ] **Step 8: Update the paths the docs and the paper point at**

```bash
grep -rln 'results/mvtec_ad2/[a-z_]*\.md' docs/ ../vlm-anomaly-paper/sections/ ../vlm-anomaly-paper/docs/
```

Rewrite each hit to `results/mvtec_ad2/patchcore_ref/<category>.md`. These are SOURCE comments
and provenance references; a stale path in a provenance line is the defect this repo treats most
seriously.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "feat: run_mvtec_ad2 --method, with method-scoped result paths

summarise loads every parquet in <results>/shards, so two methods in one
directory would average PatchCore and WinCLIP into one row and overwrite each
other's report. The filenames already carry the method, so the pooling would
have been silent — the failure class this repo keeps finding.

The eight committed reports move under patchcore_ref/ with git mv so history
follows, and the docs and the paper's SOURCE comments are repointed. PatchCore's
old Drive shards are not migrated: M3 is closed, its numbers are in the reports,
and a figure re-run wants a fresh fit anyway.

preprocess is stamped from the method or null. WinCLIP has no pre-processing
identity and a fabricated \"anomalib\" would claim a property it does not have."
```

---

### Task 3: `summarise` refuses a shard directory holding two methods

**Files:**
- Modify: `scripts/run_mvtec_ad2.py` (`summarise`)
- Test: `tests/test_run_mvtec_ad2.py`

**Interfaces:**
- Consumes: `ResultStore.load_all` (unchanged).
- Produces: nothing new; `summarise` raises `ValueError` naming the methods found.

Review Focus #2. Task 2 prevents this for new runs; this is the guard for a directory that
already holds both, which is what a hand-set `--results` or a restored Drive folder can produce.

- [ ] **Step 1: Write the failing test**

```python
def test_summarise_refuses_a_directory_holding_two_methods(monkeypatch, tmp_path, capsys):
    """Shard filenames carry the method, so two methods coexist on disk without colliding.
    Averaging them would produce a row describing no method at all."""
    import pandas as pd
    mod = _module()
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    for name in ("patchcore_ref", "winclip"):
        pd.DataFrame({
            "method": [name] * 2, "seed": [0, 0], "meta_lighting": ["regular"] * 2,
            "label": [0, 1], "image_score": [0.1, 0.9], "map_path": [None, None],
        }).to_parquet(shards / f"mvtec_ad2__{name}__vial__seed0.parquet")

    with pytest.raises(ValueError) as exc:
        mod.summarise(tmp_path)
    message = str(exc.value)
    assert "patchcore_ref" in message and "winclip" in message
    assert "pool" in message.lower() or "two methods" in message.lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py::test_summarise_refuses_a_directory_holding_two_methods -v`
Expected: FAIL — `summarise` averages both and returns without raising.

- [ ] **Step 3: Add the guard at the top of `summarise`**

Immediately after `df = ResultStore(results / "shards").load_all()`:

```python
    # Shard filenames carry the method, so two methods coexist here without colliding on disk.
    # Averaging them produces a row that describes no method at all, and nothing downstream
    # would notice. Same shape as the per-seed guard in aggregate, same reason.
    methods = (sorted(str(m) for m in df["method"].dropna().unique())
               if "method" in df.columns else [])
    if len(methods) > 1:
        raise ValueError(
            f"the shards under {results / 'shards'} hold {len(methods)} methods "
            f"({methods}): summarising them would pool two methods into one row. Point "
            "--results at one method's directory (see default_results)."
        )
    # Task 5's recovery suggestion needs the method name. It comes from the shards, which this
    # guard has just proved hold exactly one — no new parameter, and nothing to pass wrongly.
    method_name = methods[0] if methods else "patchcore_ref"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v -k summarise`
Expected: PASS, every summarise test including the new one.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_mvtec_ad2.py tests/test_run_mvtec_ad2.py
git commit -m "fix: summarise refuses a shard directory holding two methods

The filenames carry the method, so two coexist without colliding and the
average would describe no method at all. Task 2's paths prevent it for new
runs; this covers a hand-set --results or a restored Drive folder."
```

---

### Task 4: `write_report` takes the commit from the shards, not from the machine

**Files:**
- Modify: `scripts/run_mvtec_ad2.py` (`summarise`'s report block)
- Test: `tests/test_run_mvtec_ad2.py`

**Interfaces:**
- Produces: nothing new. The report's provenance line reads the shards' `commit` column; with
  more than one distinct value it lists them all rather than printing one.

Defect #2, and it is not hypothetical: on 2026-09-29 a Fruit Jelly report claimed `61704b4` for
numbers computed nine days earlier at `a7bc7bd`. Review Focus #3 covers a shard with no column.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_report_records_the_shards_commit_not_the_machines(monkeypatch, tmp_path):
    import pandas as pd
    mod = _module()
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    pd.DataFrame({
        "method": ["patchcore_ref"] * 2, "seed": [0, 0], "meta_lighting": ["regular"] * 2,
        "label": [0, 1], "image_score": [0.1, 0.9], "map_path": [None, None],
        "commit": ["a7bc7bd7d13b28c7b548c3dd50997f34e0161147"] * 2,
    }).to_parquet(shards / "mvtec_ad2__patchcore_ref__vial__seed0.parquet")

    mod.summarise(tmp_path)
    report = (tmp_path.parent / f"{tmp_path.name}.md").read_text()
    assert "a7bc7bd" in report
    assert "HEAD" not in report


def test_a_report_from_shards_at_two_commits_lists_both(monkeypatch, tmp_path):
    """A resumed run mixes commits. One of them printed as THE commit is a false claim."""
    import pandas as pd
    mod = _module()
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    for seed, sha in ((0, "a" * 40), (1, "b" * 40)):
        pd.DataFrame({
            "method": ["patchcore_ref"] * 2, "seed": [seed] * 2,
            "meta_lighting": ["regular"] * 2, "label": [0, 1],
            "image_score": [0.1, 0.9], "map_path": [None, None], "commit": [sha] * 2,
        }).to_parquet(shards / f"mvtec_ad2__patchcore_ref__vial__seed{seed}.parquet")

    mod.summarise(tmp_path)
    report = (tmp_path.parent / f"{tmp_path.name}.md").read_text()
    assert "aaaaaaa" in report and "bbbbbbb" in report
    assert "MIXED" in report or "2 commits" in report


def test_shards_without_a_commit_column_report_unknown_rather_than_raising(tmp_path):
    """Review Focus #3: shards predating provenance must not crash the report."""
    import pandas as pd
    mod = _module()
    shards = tmp_path / "shards"
    shards.mkdir(parents=True)
    pd.DataFrame({
        "method": ["patchcore_ref"] * 2, "seed": [0, 0], "meta_lighting": ["regular"] * 2,
        "label": [0, 1], "image_score": [0.1, 0.9], "map_path": [None, None],
    }).to_parquet(shards / "mvtec_ad2__patchcore_ref__vial__seed0.parquet")

    mod.summarise(tmp_path)
    report = (tmp_path.parent / f"{tmp_path.name}.md").read_text()
    assert "unknown" in report
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v -k "commit"`
Expected: FAIL — the report contains the test machine's HEAD, not the shards' value.

- [ ] **Step 3: Replace the `head` computation**

Delete the `subprocess.run(["git", ..., "rev-parse", "--short", "HEAD"])` block and derive the
line from the frame:

```python
    # The commit that computed these numbers is in the shards, not on this machine. A resumed
    # run restored from Drive carries shards from an earlier session, and stamping the writing
    # machine's HEAD claimed 61704b4 for numbers computed at a7bc7bd (2026-09-29). Read it.
    if "commit" in combined.columns:
        commits = sorted({str(c)[:7] for c in combined["commit"].dropna().unique()})
    else:
        commits = []
    if not commits:
        commit_line = "`unknown` — the shards carry no `commit` column"
    elif len(commits) == 1:
        commit_line = f"`{commits[0]}`"
    else:
        commit_line = (f"**MIXED, {len(commits)} commits**: "
                       + ", ".join(f"`{c}`" for c in commits)
                       + " — these numbers do not come from one code state")
```

and use `commit_line` where `head` was used.

- [ ] **Step 4: Run them to verify they pass**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v -k "commit or summarise"`
Expected: PASS

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 573 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/run_mvtec_ad2.py tests/test_run_mvtec_ad2.py
git commit -m "fix: the report's commit comes from the shards, not the writing machine

On 2026-09-29 a Fruit Jelly report claimed 61704b4 for numbers computed nine
days earlier at a7bc7bd, because write_report called git rev-parse HEAD. The
shards carry the truth in a commit column stamped by run_meta.

Shards at two commits list both and say MIXED rather than picking one, and
shards with no column say unknown rather than raising."
```

---

### Task 5: The guard's recovery suggestion is usable

**Files:**
- Modify: `scripts/run_mvtec_ad2.py` (the `max_bytes` except branch, around lines 243-250)
- Test: `tests/test_run_mvtec_ad2.py`

**Interfaces:**
- Produces: nothing new. The printed command includes `--summarise-only` and a budget above the
  measured requirement.

Defect #3. On Wall Plugs it suggested 9 GB for a peak it had just measured at 10.03 GB, and
omitted `--summarise-only`, so following it verbatim fails twice over.

- [ ] **Step 1: Write the failing test**

```python
def test_the_recovery_suggestion_is_runnable(monkeypatch, tmp_path, capsys):
    """Review Focus #4: the suggested budget must never sit below the measured peak, and the
    command must carry --summarise-only or the retry reloads torch it does not need."""
    mod = _module()
    needed = 10_027_008_000

    def _boom(*a, **kw):
        raise ValueError(
            f"group meta_lighting='overexposed': needs about {needed:,} bytes "
            f"(10.0 GB) at peak, which exceeds max_bytes=6,000,000,000"
        )
    # summarise calls vlmab.eval.aggregate.aggregate; that is where the guard raises from.
    import vlmab.eval.aggregate as agg_mod
    monkeypatch.setattr(agg_mod, "aggregate", _boom)

    shards = (tmp_path / "vial" / "shards")
    shards.mkdir(parents=True)
    import pandas as pd
    pd.DataFrame({
        "method": ["patchcore_ref"] * 2, "seed": [0, 0], "meta_lighting": ["regular"] * 2,
        "label": [0, 1], "image_score": [0.1, 0.9], "map_path": [None, None],
        "commit": ["a" * 40] * 2,
    }).to_parquet(shards / "mvtec_ad2__patchcore_ref__vial__seed0.parquet")

    with pytest.raises(SystemExit):
        mod.summarise(tmp_path / "vial", max_bytes=6_000_000_000)
    err = capsys.readouterr().err

    assert "--summarise-only" in err, "without the flag the retry reloads torch it does not need"
    assert "--method" in err, "the retry must name the method or it defaults to the wrong one"
    suggested = int(err.split("--max-bytes")[1].split()[0])
    assert suggested >= needed, (
        f"suggested {suggested:,} for a peak of {needed:,} — following it fails again"
    )
```

The guard branch writes to **stderr**, not stdout, and ends in `raise SystemExit(1)`; both are
existing behaviour this task preserves.

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py::test_the_recovery_suggestion_is_runnable -v`
Expected: FAIL — the suggestion omits `--summarise-only` and the budget is derived from free RAM.

- [ ] **Step 3: Build the suggestion from the requirement**

```python
            # The budget has to cover the peak the guard just measured, not a fraction of free
            # RAM: on Wall Plugs this suggested 9 GB for a 10.03 GB peak and failed again. And
            # the retry needs --summarise-only, or the seed loop builds a backend per seed and
            # holds ~3 GB of torch while aggregating — the overhead the flag exists to shed.
            needed = _bytes_from_guard_message(str(exc))
            budget = int(needed * 1.15) if needed else None
            print(f"\n{exc}\n", file=sys.stderr)
            print("The shards are safe — this is the summary only, and nothing needs re-running.",
                  file=sys.stderr)
            if available:
                print(f"This machine reports {available / 1e9:.1f} GB available right now.",
                      file=sys.stderr)
            if budget:
                if available and budget > available:
                    print(f"⚠️  the requirement is above what this machine reports free. "
                          "A high-RAM runtime, or one lighting condition at a time.",
                          file=sys.stderr)
                print("Re-run with:\n"
                      f"    python scripts/run_mvtec_ad2.py --category {results.name} "
                      f"--method {method_name} --summarise-only --max-bytes {budget}\n",
                      file=sys.stderr)
            else:
                print("Re-run with --summarise-only and a budget above the requirement above.\n",
                      file=sys.stderr)
```

with a small helper that parses the byte count out of the guard's own message:

```python
def _bytes_from_guard_message(message: str) -> int | None:
    """The guard states the requirement in bytes; take it from there rather than re-deriving."""
    import re
    match = re.search(r"about ([\d,]+) bytes", message)
    return int(match.group(1).replace(",", "")) if match else None
```

- [ ] **Step 4: Run it to verify it passes**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad2.py -v -k "recovery or guard or memory"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/run_mvtec_ad2.py tests/test_run_mvtec_ad2.py
git commit -m "fix: the guard's recovery suggestion is runnable

On Wall Plugs it suggested 9 GB for a peak it had just measured at 10.03 GB and
omitted --summarise-only, so following it verbatim failed twice over. The budget
now comes from the requirement the guard states, with 15% headroom, and says so
when that exceeds what the machine reports free."
```

---

### Task 6: A guard against shards with no report

**Files:**
- Create: `tests/test_every_run_has_a_report.py`
- Test: itself

**Interfaces:** none. A repository-level guard, like `test_results_are_tracked.py`.

Defect #4. Fruit Jelly ran on 2026-09-20 and its result existed nowhere for nine days, because
nothing checks that a shard directory has a report beside it. The shards are gitignored, so this
can only check what is in the tree — which is exactly the case that bit: the directory was
present locally and the report was not written.

- [ ] **Step 1: Write the test**

```python
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
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/pytest tests/test_every_run_has_a_report.py -v`
Expected: SKIPPED on a clean checkout ("no shards in this working tree"). That is the correct
outcome here and the test's own docstring says why.

- [ ] **Step 3: Prove it catches the case it exists for**

Create a fake shard, confirm the test fails, then remove it:

```bash
mkdir -p results/mvtec_ad2/patchcore_ref/faketest/shards
touch results/mvtec_ad2/patchcore_ref/faketest/shards/x.parquet
.venv/bin/pytest tests/test_every_run_has_a_report.py -q   # expect 1 failed
rm -rf results/mvtec_ad2/patchcore_ref/faketest
.venv/bin/pytest tests/test_every_run_has_a_report.py -q   # expect 1 skipped
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_every_run_has_a_report.py
git commit -m "test: a shard directory without a report beside it fails

Fruit Jelly ran on 2026-09-20 and its numbers existed only on Drive until
2026-09-29. write_report covers the common case; this covers a session that
died between the shards and the summary. Shards are gitignored, so the guard
runs against the working tree and correctly skips on a clean checkout."
```

---

### Task 7: `run_visa_secondary.py --method`, and the target block looked up not hardcoded

**Files:**
- Create: `src/vlmab/eval/targets.py`
- Modify: `scripts/run_visa_secondary.py` (docstring, the backend construction at line 79, `main` at 103-133)
- Test: `tests/test_targets.py`, `tests/test_visa_secondary_script.py`

**Interfaces:**
- Consumes: `vlmab.methods.gpu.build_runnable` (Task 1).
- Produces: `vlmab.eval.targets.block_for(targets: Path, dataset: str) -> str` — the name of the
  single top-level block in a reproduction config whose `dataset` matches, raising `ValueError`
  for zero or several. Task 8 calls it too. Plus `--method` on the script (default
  `patchcore_ref`).

**Why a lookup rather than a map.** The block names differ per method and are not guessable:
`patchcore_ref.yaml` has `gate` (MVTec AD classic) and `secondary` (VisA, **non-gating**, because
its paper predates VisA), while `winclip.yaml` has `gate_mvtec_ad` and `gate_visa`, **both
gating** (§2 v0.2.14). A hardcoded per-method dict would need editing for every method added and
would silently pick the wrong block if a name changed. The config already states `dataset:` in
each block, so the right block is derivable. `not_the_target` — WinCLIP's Table 7 ablation, which
is higher than the real target and therefore the tempting number — declares no `dataset` and is
skipped by construction.

- [ ] **Step 1: Write the failing tests for the lookup**

```python
"""Finding the right target block in a reproduction config, instead of hardcoding its name.

The names differ per method: patchcore_ref has `gate` and `secondary`, winclip has
`gate_mvtec_ad` and `gate_visa`. Both configs state `dataset:` inside each block, so the block
is derivable and a per-method map is unnecessary.
"""
from pathlib import Path

import pytest

from vlmab.eval.targets import block_for

ROOT = Path(__file__).resolve().parents[1]


def test_patchcores_visa_block_is_its_non_gating_secondary():
    assert block_for(ROOT / "configs/reproduction/patchcore_ref.yaml", "visa") == "secondary"


def test_patchcores_classic_block_is_its_gate():
    assert block_for(ROOT / "configs/reproduction/patchcore_ref.yaml", "mvtec_ad") == "gate"


def test_winclip_has_a_gating_block_for_each_dataset():
    cfg = ROOT / "configs/reproduction/winclip.yaml"
    assert block_for(cfg, "mvtec_ad") == "gate_mvtec_ad"
    assert block_for(cfg, "visa") == "gate_visa"


def test_the_not_the_target_block_is_never_returned(tmp_path):
    """WinCLIP's Table 7 ablation is higher than the real target. It declares no dataset, and
    a lookup that returned it would hand the gate the tempting number."""
    cfg = tmp_path / "t.yaml"
    cfg.write_text(
        "method: winclip\n"
        "gate_visa:\n  dataset: visa\n  published_mean: 78.1\n"
        "not_the_target:\n  published_mean: 78.9\n"
    )
    assert block_for(cfg, "visa") == "gate_visa"


def test_a_dataset_with_no_block_is_refused_by_name(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text("method: winclip\ngate_visa:\n  dataset: visa\n")
    with pytest.raises(ValueError) as exc:
        block_for(cfg, "mvtec_ad")
    assert "mvtec_ad" in str(exc.value) and "gate_visa" in str(exc.value)


def test_two_blocks_for_one_dataset_are_refused_rather_than_guessed(tmp_path):
    cfg = tmp_path / "t.yaml"
    cfg.write_text(
        "method: m\na:\n  dataset: visa\nb:\n  dataset: visa\n"
    )
    with pytest.raises(ValueError) as exc:
        block_for(cfg, "visa")
    assert "a" in str(exc.value) and "b" in str(exc.value)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_targets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vlmab.eval.targets'`

- [ ] **Step 3: Write the lookup**

```python
"""Which block of a reproduction config holds a dataset's target.

Hardcoding the name per method would need editing for every method added and would pick the
wrong block silently if a name changed. Each block already states its `dataset`, so the answer
is in the file. Blocks without a `dataset` key — `not_the_target`, the counts at the top level —
are skipped by construction, which matters: WinCLIP's `not_the_target` holds Table 7's 78.9,
higher than the real 78.1 and therefore the number a wrong lookup would be tempted by.
"""
from pathlib import Path
from typing import Any

import yaml


def block_for(targets: Path, dataset: str) -> str:
    """The single top-level block in `targets` whose `dataset` is `dataset`.

    Raises ValueError for none or several, naming what was found. Guessing between two blocks
    would mean scoring a gate against a target nobody chose.
    """
    loaded: dict[str, Any] = yaml.safe_load(Path(targets).read_text()) or {}
    blocks = {
        name: body for name, body in loaded.items()
        if isinstance(body, dict) and "dataset" in body
    }
    matching = sorted(name for name, body in blocks.items() if body["dataset"] == dataset)
    if not matching:
        raise ValueError(
            f"{Path(targets).name} has no block for dataset {dataset!r}; "
            f"blocks that declare one: {sorted(blocks)}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"{Path(targets).name} has {len(matching)} blocks for dataset {dataset!r} "
            f"({matching}); a gate scored against a guessed block is scored against a target "
            "nobody chose"
        )
    return matching[0]
```

- [ ] **Step 4: Run them to verify they pass**

Run: `.venv/bin/pytest tests/test_targets.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Write the failing tests for the script**

```python
def test_the_method_is_selectable_and_comes_from_the_factory(visa_harness):
    visa_harness["main"](["--method", "winclip", "--skip-fetch", "--no-score"])
    assert visa_harness["built"][0][0] == "winclip"


def test_patchcore_is_still_the_default(visa_harness):
    visa_harness["main"](["--skip-fetch", "--no-score"])
    assert visa_harness["built"][0][0] == "patchcore_ref"


def test_the_targets_file_and_block_follow_the_method(visa_harness):
    """WinCLIP's VisA figure GATES; PatchCore's is a non-gating secondary. The script must not
    decide that — it passes the block the config declares for the dataset."""
    visa_harness["main"](["--method", "winclip", "--skip-fetch"])
    argv = visa_harness["gate_argv"][0]
    assert "configs/reproduction/winclip.yaml" in argv
    assert argv[argv.index("--which") + 1] == "gate_visa"


def test_patchcores_block_is_still_secondary(visa_harness):
    visa_harness["main"](["--skip-fetch"])
    argv = visa_harness["gate_argv"][0]
    assert argv[argv.index("--which") + 1] == "secondary"


def test_results_and_report_paths_are_method_scoped(visa_harness):
    mod = visa_harness["module"]
    assert mod.default_results("winclip").parts[-2:] == ("winclip", "visa")
    assert mod.default_out("winclip").name == "winclip_visa.md"


def test_a_method_without_a_gpu_backend_exits_2(visa_harness):
    assert visa_harness["main"](["--method", "saa", "--skip-fetch"]) == 2
```

The `visa_harness` fixture mirrors Task 2's: it patches `vlmab.methods.gpu.build_runnable`,
`vlmab.eval.runner.run_evaluation`, and `subprocess.run` (recording `gate_argv` and returning a
zero-returncode stub), then calls `mod.main`.

- [ ] **Step 6: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_visa_secondary_script.py -v -k "method or block or scoped"`
Expected: FAIL — unrecognised argument `--method`.

- [ ] **Step 7: Change the script**

```python
def default_results(method: str) -> Path:
    return Path("results/reproduction") / method / "visa"


def default_out(method: str) -> Path:
    return Path("results/reproduction") / f"{method}_visa.md"
```

In `run`, replace the `PatchCoreBackend` import with `build_runnable(method_name, seed=seed)` and
stamp `getattr(method, "preprocess", None)`, exactly as Task 2 does in `run_one_seed`.

In `main`:

```python
    parser.add_argument("--method", default="patchcore_ref")
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--which", default=None,
                        help="target block; derived from the config when omitted")
    args = parser.parse_args(argv)

    from vlmab.methods.gpu import gpu_builders
    if args.method not in gpu_builders():
        print(f"\ncannot start: no GPU backend for {args.method!r}; "
              f"methods with one: {gpu_builders()}", file=sys.stderr)
        return 2

    results = args.results or default_results(args.method)
    out = args.out or default_out(args.method)
    targets = Path("configs/reproduction") / f"{args.method}.yaml"

    if not args.skip_fetch:
        fetch(args.root)
    run(args.root, results, args.seed, args.device, args.method)

    if args.no_score:
        print("skipping the score step as asked")
        return 0

    from vlmab.eval.targets import block_for
    which = args.which or block_for(targets, "visa")

    gate = Path(__file__).resolve().parent / "reproduction_gate.py"
    return subprocess.run(
        [sys.executable, str(gate),
         "--results", str(results / "shards"),
         "--targets", str(targets),
         "--which", which,
         "--out", str(out)],
    ).returncode
```

Amend the docstring paragraph about the non-gating VisA figure to say it is **PatchCore's**
situation specifically — its paper predates VisA — and that whether a dataset's figure gates is
declared per method in `configs/reproduction/`, which `block_for` reads. Leave the rest of the
docstring: every other paragraph still holds.

- [ ] **Step 8: Run the tests**

Run: `.venv/bin/pytest tests/test_visa_secondary_script.py tests/test_targets.py -v`
Expected: PASS

- [ ] **Step 9: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 581 passed

- [ ] **Step 10: Commit**

```bash
git add src/vlmab/eval/targets.py tests/test_targets.py scripts/run_visa_secondary.py tests/test_visa_secondary_script.py
git commit -m "feat: run_visa_secondary --method, with the target block looked up

The block names are not guessable: patchcore_ref has gate/secondary, winclip has
gate_mvtec_ad/gate_visa, and WinCLIP's VisA figure GATES where PatchCore's is a
non-gating secondary because its paper predates VisA. Each block already states
its dataset, so block_for derives it and a per-method map is unnecessary.

not_the_target declares no dataset and is skipped by construction, which matters:
it holds Table 7's 78.9, higher than the real 78.1 and the number a wrong lookup
would reach for."

```

---

### Task 8: `scripts/run_mvtec_ad.py` — the MVTec AD classic gate, as a script

**Files:**
- Create: `scripts/run_mvtec_ad.py`
- Test: `tests/test_run_mvtec_ad_script.py`

**Interfaces:**
- Consumes: `vlmab.methods.gpu.build_runnable` (Task 1), `vlmab.datasets.mvtec_ad.MVTecAD`,
  `vlmab.eval.runner.run_evaluation`, `vlmab.eval.store.ResultStore`.
- Produces: `python scripts/run_mvtec_ad.py --method winclip [--categories ...] [--seeds ...]`,
  writing shards under `results/reproduction/<method>/mvtec_ad/shards`, scored afterwards by
  the existing `scripts/reproduction_gate.py --which`.

This path has never existed as a script. PatchCore's classic gate ran from notebook cells 3.2
and 3b.1, which is why the 15-category run had to be re-typed for phase 3b and why its fit times
were never recorded. WinCLIP needs it twice.

- [ ] **Step 1: Write the failing tests**

```python
"""The MVTec AD classic gate runner. PatchCore's ran from notebook cells; WinCLIP needs a script.

Fifteen categories at one or three seeds, writing the shards reproduction_gate.py scores. Same
fake-injection style as the other runner tests: none of this runs on CPU.
"""
import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mvtec_ad.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_mvtec_ad", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_fifteen_categories_are_the_default(harness):
    harness["main"](["--method", "winclip"])
    scored = [kw["categories"] for kw in harness["calls"]]
    assert len(scored[0]) == 15, "the published 91.8 is a mean over fifteen; a subset is not it"


def test_it_scores_the_test_split_and_fits_on_train(harness):
    harness["main"](["--method", "winclip"])
    kw = harness["calls"][0]
    assert kw["split"] == "test" and kw["fit_split"] == "train"


def test_a_subset_warns_that_the_gate_needs_all_fifteen(harness, capsys):
    harness["main"](["--method", "winclip", "--categories", "bottle", "cable"])
    out = capsys.readouterr().out
    assert "15" in out and ("not" in out.lower() or "⚠" in out)


def test_the_method_comes_from_the_factory(harness):
    harness["main"](["--method", "winclip"])
    assert harness["built"][0][0] == "winclip"


def test_a_method_without_a_backend_exits_2(harness, capsys):
    assert harness["main"](["--method", "saa"]) == 2


def test_shards_are_written_under_a_method_scoped_reproduction_path(harness):
    """run_evaluation takes a store, not a path, so assert on the store's own root."""
    harness["main"](["--method", "winclip"])
    store = harness["stores"][0]
    assert store.root.parts[-3:] == ("winclip", "mvtec_ad", "shards")


def test_the_target_block_is_derived_from_the_config(harness):
    """winclip's classic block is gate_mvtec_ad; patchcore_ref's is gate. Neither is hardcoded."""
    from vlmab.eval.targets import block_for
    assert block_for(Path("configs/reproduction/winclip.yaml"), "mvtec_ad") == "gate_mvtec_ad"
    assert block_for(Path("configs/reproduction/patchcore_ref.yaml"), "mvtec_ad") == "gate"
```

The `harness` fixture here is Task 2's shape with `MVTecAD` in place of `MVTecAD2`, no Drive
patches (there are none to make), and a `stores` list recording the `ResultStore` each call
received — `monkeypatch.setattr(runner_mod, "run_evaluation", lambda ds, m, store, meta, **kw:
calls.append(kw) or stores.append(store) or [])`.

- [ ] **Step 2: Run them to verify they fail**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad_script.py -v`
Expected: FAIL — the script does not exist.

- [ ] **Step 3: Write the script**

No Drive machinery: MVTec AD classic downloads directly, so there is no archive to fetch and
nothing to restore. No per-lighting aggregation either — classic has no lighting conditions, and
the gate is scored by `reproduction_gate.py`, not here.

```python
#!/usr/bin/env python3
"""Run one method over MVTec AD classic's 15 categories: the reproduction gate's data path.

    python scripts/run_mvtec_ad.py --method winclip
    python scripts/run_mvtec_ad.py --method winclip --categories bottle cable   # a probe

**This existed only as notebook cells until 2026-10-01.** PatchCore's gate ran from cells 3.2 and
3b.1, which is why phase 3b had to be re-typed by hand and why none of its fit times were
recorded. WinCLIP needs this path twice (its classic gate and, through
`run_visa_secondary.py`, its VisA one), and both of its gates bind under protocol §2 v0.2.14.

**All fifteen categories are the default** because the published figure every gate compares
against is their mean. A subset is a probe and says so.

Scoring is `scripts/reproduction_gate.py`, which this script does not call: the gate reads
shards, so running and scoring stay separable and a crashed run can be scored later.

Exit codes: 0 ran, 1 a run failed, 2 could not start.
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

#: Protocol §6. The existing warning covers fewer; nothing here decides a method is deterministic.
DEFAULT_SEEDS = (0, 1, 2)


def default_results(method: str) -> Path:
    return Path("results/reproduction") / method / "mvtec_ad"


def run_one_seed(root: Path, results: Path, seed: int, device: str, method_name: str,
                 categories: Sequence[str]) -> None:
    from vlmab.datasets.mvtec_ad import MVTecAD
    from vlmab.eval.provenance import run_meta
    from vlmab.eval.runner import run_evaluation
    from vlmab.eval.store import ResultStore
    from vlmab.methods.gpu import build_runnable

    # A fresh adapter per seed, for the reason the AD 2 runner documents: reusing one carries the
    # previous seed's state into the next seed's scores while the shard claims otherwise.
    method = build_runnable(method_name, seed=seed)
    store = ResultStore(results / "shards")
    preprocess = getattr(method, "preprocess", None)
    meta = run_meta({"method": method_name, "split": "test", "preprocess": preprocess})
    meta = {**meta, "preprocess": preprocess}

    run_evaluation(
        MVTecAD(root), method, store, meta,
        categories=list(categories), split="test", fit_split="train",
        maps_dir=results / "maps", device=device,
    )
    print(f"  seed {seed}: done over {len(categories)} categories (preprocess {preprocess})")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", default="patchcore_ref")
    parser.add_argument("--root", type=Path, default=Path("data/mvtec_ad"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    head = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent.parent),
                           "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip() or "unknown"
    print(f"{Path(__file__).name} running from commit {head}\n")

    from vlmab.methods.gpu import gpu_builders
    if args.method not in gpu_builders():
        print(f"\ncannot start: no GPU backend for {args.method!r}; "
              f"methods with one: {gpu_builders()}", file=sys.stderr)
        return 2

    from vlmab.datasets.mvtec_ad import MVTecAD
    categories = args.categories or MVTecAD(args.root).categories()
    if len(categories) != 15:
        print(f"⚠️  {len(categories)} categories, not 15. The published figure every gate "
              "compares against is the mean over all 15, so this is a probe and its numbers "
              "are not a gate verdict.")
    if len(args.seeds) != len(set(args.seeds)):
        print(f"repeated seed in {args.seeds}; each seed is one run", file=sys.stderr)
        return 2
    if len(args.seeds) < 3:
        print(f"⚠️  {len(args.seeds)} seed(s). Protocol §6 requires three for any REPORTED "
              "number wherever stochasticity exists.")

    results = args.results or default_results(args.method)
    for i, seed in enumerate(args.seeds, start=1):
        print(f"[seed {i}/{len(args.seeds)}] {args.method} over MVTec AD classic at seed {seed}")
        run_one_seed(args.root, results, seed, args.device, args.method, categories)

    print(f"\ndone at seeds {args.seeds}. Shards: {results / 'shards'}\n"
          f"Score it with:\n"
          f"    python scripts/reproduction_gate.py --results {results / 'shards'} "
          f"--targets configs/reproduction/{args.method}.yaml --which "
          f"<the block for mvtec_ad; see vlmab.eval.targets.block_for>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_run_mvtec_ad_script.py -v`
Expected: PASS, 6 tests

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -q`
Expected: 582 passed

- [ ] **Step 6: Commit**

```bash
git add scripts/run_mvtec_ad.py tests/test_run_mvtec_ad_script.py
git commit -m "feat: run_mvtec_ad.py — the classic gate, as a script at last

PatchCore's 15-category gate ran from notebook cells 3.2 and 3b.1, which is why
phase 3b had to be re-typed and why its fit times were never recorded. WinCLIP
needs this path twice. No Drive machinery: classic downloads directly."
```

---

### Task 9: Defect #1 — the coreset size reaches provenance

**Files:**
- Modify: `src/vlmab/methods/patchcore_ref.py`
- Test: `tests/test_patchcore_ref.py`

**Interfaces:**
- Produces: `PatchCoreRef.memory_bank_size -> int | None`, read by nothing yet; the runners pick
  it up through `run_meta` in a follow-up, not here.

**This task violates the global constraint and must be done LAST, after the figures exist.** It
modifies the scoring path's module. Measured on 2026-09-30, the coreset is exactly
`floor(N × 102.4)` and the `Selecting Coreset Indices` bar shows one less; that figure was read
off the bar for five of eight categories and put a wrong number in three reports. Fixing it is
right, but a plan that edits `patchcore_ref.py` before the figures are regenerated breaks spec §5.

- [ ] **Step 1: Confirm the figures are done**

Check that `docs/superpowers/plans/` holds a completed figures plan and that the figure files
exist. **If they do not, stop: this task is not ready.**

- [ ] **Step 2: Write the failing test**

```python
def test_the_memory_bank_size_is_exposed_for_provenance():
    """Read off a tqdm bar for five of eight categories, and the bar is one short of the bank."""
    class _Backend:
        seed = 0
        preprocess = "anomalib"
        memory_bank_size = 29798
    method = PatchCoreRef(backend=_Backend())
    assert method.memory_bank_size == 29798


def test_it_is_none_before_a_fit_rather_than_zero():
    """Zero is a measurement; None is "not fitted yet". They must not be the same value."""
    class _Backend:
        seed = 0
        preprocess = "anomalib"
        memory_bank_size = None
    assert PatchCoreRef(backend=_Backend()).memory_bank_size is None
```

- [ ] **Step 3: Add the property**

```python
    @property
    def memory_bank_size(self) -> int | None:
        """Entries in the fitted coreset, or None before a fit.

        Measured 2026-09-30: this is exactly floor(N x 102.4) and anomalib's progress bar shows
        one less, which is how 29797/14027/26930 reached three reports instead of
        29798/14028/26931. Provenance should not be read off a tqdm bar.
        """
        return getattr(self._backend, "memory_bank_size", None)
```

The backend side (`memory_bank_size` on `PatchCoreBackend`, from
`self._model.model.memory_bank.shape[0]`) belongs to the same task; it is GPU-only code and
cannot be tested here, so it carries the probe pattern's comment and is exercised in the next
Colab session.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/pytest tests/test_patchcore_ref.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vlmab/methods/patchcore_ref.py src/vlmab/methods/patchcore_backend.py tests/test_patchcore_ref.py
git commit -m "feat: the coreset size reaches provenance instead of a tqdm bar

Measured 2026-09-30: the bank is floor(N x 102.4) and the bar shows one less,
which is how three reports got 29797/14027/26930. Deliberately last in the
plan: it edits the scoring path, which spec section 5 freezes until the figures
are regenerated."
```

---

## What this plan does NOT cover

- **B and C** (the WinCLIP backend and its two gates) are a Colab playbook executed
  interactively with a VERIFY step per upstream call, which is how every GPU session in this
  project has run. `docs/superpowers/plans/2026-07-24-winclip-zeroshot-adapter.md` already holds
  phases A–D; it predates the two pre-registered gates (2026-09-18) and needs an amendment, not
  a rewrite.
- **D, E, F** (the grid, calibration, submission packaging) depend on C passing and on the
  server's own docs, which are three unticked checkboxes.
- **G** (figures) is its own plan: regeneration runs plus CPU plotting code.
