"""The M3 grid runner: one MVTec AD 2 category, every seed §6 requires.

Tested the way the VisA runner is, and for the same reason: none of this can run on CPU, and the
defects that hurt are the ones that appear after the expensive part has already been paid for.
The fixtures below are fakes injected through the function-local imports the script's structure
allows.
"""
import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mvtec_ad2.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_mvtec_ad2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeBackend:
    def __init__(self, seed=None, preprocess="anomalib", **kw):
        self.seed = seed
        self.preprocess = preprocess


@pytest.fixture
def harness(monkeypatch, tmp_path):
    """A category already on disk, a fake backend, and a recorder for run_evaluation."""
    import vlmab.datasets.mvtec_ad2 as ds_mod
    import vlmab.eval.runner as runner_mod
    import vlmab.methods.patchcore_backend as backend_mod

    calls: list[dict] = []
    monkeypatch.setattr(backend_mod, "PatchCoreBackend", _FakeBackend)
    monkeypatch.setattr(ds_mod, "MVTecAD2", lambda root: types.SimpleNamespace(root=root))
    monkeypatch.setattr(
        runner_mod, "run_evaluation",
        lambda dataset, method, store, meta, **kw: calls.append(
            {"method": method, "meta": dict(meta), **kw}) or [])

    root = tmp_path / "data"
    (root / "vial" / "train").mkdir(parents=True)

    mod = _module()
    monkeypatch.setattr(mod, "verify_layout", lambda *a, **k: None)
    monkeypatch.setattr(mod, "summarise", lambda *a, **k: None)
    return mod, calls, root, tmp_path


def _run(mod, root, tmp_path, *extra):
    return mod.main(["--category", "vial", "--root", str(root), "--no-save",
                     "--results", str(tmp_path / "results"), *extra])


def test_three_seeds_are_the_default_because_section_6_requires_them(harness):
    mod, calls, root, tmp = harness
    assert _run(mod, root, tmp) == 0
    assert [c["method"].seed for c in calls] == [0, 1, 2]


def test_each_seed_gets_its_own_backend(harness):
    """The memory bank is what the seed changes. A reused backend would carry the previous
    seed's bank into the next seed's scores while the shard claimed otherwise -- the defect the
    seed-provenance work already had to undo once."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp)
    assert len({id(c["method"]) for c in calls}) == 3


def test_the_shard_records_the_preprocess_the_method_applied(harness):
    """run_meta hashes its cfg rather than passing the keys through, so `preprocess` has to be
    added as a real column on top of it. Without this the value that decides a run's identity
    would be recoverable only by recomputing a hash."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp)
    for call in calls:
        assert call["meta"]["preprocess"] == "anomalib" == call["method"].preprocess
        assert "config_hash" in call["meta"] and "commit" in call["meta"]


def test_it_runs_the_public_test_split_and_fits_on_train(harness):
    """M3 is the public-test grid. test_private has no labels and validation is M4's."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp)
    assert {c["split"] for c in calls} == {"test_public"}
    assert {c["fit_split"] for c in calls} == {"train"}


def test_validation_is_deliberately_not_scored(harness):
    """Pinned so it is not quietly added later. run_evaluation re-fits per call, so a second
    split roughly doubles the fit cost per seed, and the threshold rule's seed semantics are not
    pre-registered yet -- producing those scores now risks producing the wrong ones. The cost of
    this decision is a second upload of each category at M4, recorded in the script's docstring."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp)
    assert "validation" not in {c["split"] for c in calls}


def test_a_missing_archive_exits_2_not_1(harness):
    """0 ran, 1 a run failed, 2 could not start. A category that was never uploaded to Drive must
    not look like one that ran and produced nothing."""
    mod, calls, root, tmp = harness
    rc = mod.main(["--category", "fabric", "--root", str(root), "--no-save",
                   "--archives", str(tmp / "empty"), "--results", str(tmp / "r")])
    assert rc == 2
    assert not calls


def test_an_already_extracted_category_is_not_re_extracted(harness, capsys):
    """Re-running after a disconnect must not spend minutes on 10 GB it already has."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp)
    assert "already extracted" in capsys.readouterr().out


def test_a_repeated_seed_is_refused(harness):
    """Each seed is one run, and the same seed twice would overwrite one shard with another
    that the store considers identical."""
    mod, calls, root, tmp = harness
    assert _run(mod, root, tmp, "--seeds", "0", "1", "0") == 2
    assert not calls


def test_fewer_than_three_seeds_warns_and_continues(harness, capsys):
    """A single seed is a legitimate measurement -- a cost probe, say -- but never a reported
    number, and the run has to say so at the point where it is chosen."""
    mod, calls, root, tmp = harness
    assert _run(mod, root, tmp, "--seeds", "0") == 0
    out = capsys.readouterr().out
    assert "1 seed(s)" in out and "§6" in out
    assert [c["method"].seed for c in calls] == [0]


def test_shards_reach_drive_after_every_seed_not_only_at_the_end(monkeypatch, harness):
    """The 2026-09-19 lesson. A multi-hour run that loses its VM at the last category loses
    everything if the copy waits for the end, and Colab serializes cells so nothing else can
    rescue it mid-run."""
    mod, calls, root, tmp = harness
    saved: list[int] = []
    monkeypatch.setattr(mod, "save_to_drive", lambda results, dest: saved.append(len(calls)))
    rc = mod.main(["--category", "vial", "--root", str(root),
                   "--results", str(tmp / "results"), "--drive-results", str(tmp / "drive")])
    assert rc == 0
    assert saved == [1, 2, 3], "one copy per finished seed"


# --- Mounting Drive is the notebook's job, and this is why ---------------------------------
# Added 2026-09-21 after the first live M3 run died here. `google.colab.drive.mount` sends an
# authentication request to the Colab frontend THROUGH the IPython kernel; a `!python`
# subprocess has no kernel, so `get_ipython()` returns None and the call dies inside Colab's own
# `_message.send_request` with an AttributeError naming nothing relevant.
#
# The tests above never caught it because every one of them passed --no-mount, which skipped the
# only line that mattered. A flag that turns off the code under test is not a test of it.


def test_the_script_never_tries_to_mount_drive():
    """It cannot, from where it runs. Asserted against the parsed source rather than by running
    it, because the failure only reproduces inside a real Colab subprocess."""
    import ast

    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    offenders = {m for m in imported if m.startswith("google")}
    assert not offenders, (
        f"{SCRIPT.name} imports {offenders}. Mounting is the notebook's job: this runs as a "
        "subprocess with no IPython kernel, and drive.mount() needs one."
    )


def test_an_unmounted_drive_is_reported_as_such_and_not_as_a_missing_upload(harness, capsys):
    """Two very different problems that would otherwise produce the same message: Drive not
    mounted, versus mounted but the category never uploaded. The first is fixed in ten seconds
    and the second needs a download behind a registration form."""
    mod, calls, root, tmp = harness
    rc = mod.main(["--category", "fabric", "--root", str(root), "--no-save",
                   "--archives", "/content/drive/MyDrive/mvtec_ad2",
                   "--results", str(tmp / "r")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not mounted" in err
    assert "drive.mount('/content/drive')" in err, "the message must carry the exact fix"
    assert not calls


def test_it_prints_the_commit_it_is_running_from(harness, capsys):
    """A stale clone is invisible in a traceback: the line numbers look plausible and the code is
    simply the old code. One live M3 run was lost to that — a fix sat on origin/master, the
    session had not synced, and the traceback named a call signature that no longer existed
    anywhere in the repo. Printed before anything can fail, so it is there even on a crash."""
    mod, calls, root, tmp = harness
    _run(mod, root, tmp, "--seeds", "0")
    out = capsys.readouterr().out
    assert "running from commit" in out
    assert out.index("running from commit") < out.index("seed 1/1")


# --- summarise() must aggregate ONE SEED AT A TIME -----------------------------------------
# The first version pooled all three seeds into one `aggregate` call and died on
# `aggregate._require_single_seed` -- after every seed had already been computed and saved. The
# guard was right and the caller was wrong: I-AUROC over 3N rows is not the mean of three
# I-AUROCs, it scores every image three times and treats the copies as independent samples.
#
# The fake below runs the REAL guard, so this test reproduces the original failure exactly
# rather than approximating it.


def _seeded_frame(seeds=(0, 1, 2), lightings=("regular", "overexposed")):
    import pandas as pd

    rows = [{"seed": s, "meta_lighting": light, "label": i % 2, "score": float(i)}
            for s in seeds for light in lightings for i in range(4)]
    return pd.DataFrame(rows)


def _patch_summarise_deps(monkeypatch, frame, calls):
    import pandas as pd

    import vlmab.eval.aggregate as agg_mod
    import vlmab.eval.store as store_mod

    def fake_aggregate(df, by=None, **kw):
        agg_mod._require_single_seed(df)          # the real guard, not a stand-in
        calls.append(sorted(df["seed"].unique()))
        return pd.DataFrame({by: sorted(df[by].unique()),
                             "i_auroc": [90.0 + len(calls)] * df[by].nunique()})

    monkeypatch.setattr(agg_mod, "aggregate", fake_aggregate)
    monkeypatch.setattr(store_mod.ResultStore, "load_all", lambda self: frame)


def test_summarise_never_hands_aggregate_a_pooled_seed_frame(monkeypatch, tmp_path, capsys):
    calls: list = []
    _patch_summarise_deps(monkeypatch, _seeded_frame(), calls)

    _module().summarise(tmp_path)

    assert calls == [[0], [1], [2]], f"aggregate saw {calls}; each call must be one seed"
    out = capsys.readouterr().out
    assert "3 seed(s)" in out and "§6" in out


def test_summarise_reports_a_spread_not_a_single_number(monkeypatch, tmp_path, capsys):
    """§6 reports mean ± std. A bare mean would hide a seed disagreement, which for PatchCore is
    the measurement the whole seed apparatus exists to produce."""
    calls: list = []
    _patch_summarise_deps(monkeypatch, _seeded_frame(), calls)

    _module().summarise(tmp_path)

    out = capsys.readouterr().out
    assert "±" in out
    for lighting in ("regular", "overexposed"):
        assert lighting in out


def test_summarise_handles_a_single_seed_and_says_it_is_not_reportable(monkeypatch, tmp_path,
                                                                       capsys):
    """std over one seed is NaN, which must print as 0.00 rather than leaking 'nan' into a
    table, and one seed has to be flagged at the point it is shown."""
    calls: list = []
    _patch_summarise_deps(monkeypatch, _seeded_frame(seeds=(0,)), calls)

    _module().summarise(tmp_path)

    out = capsys.readouterr().out
    assert calls == [[0]]
    assert "nan" not in out.lower()
    assert "1 seed(s)" in out and "requires three" in out


def test_summarise_writes_a_report_rather_than_only_printing(monkeypatch, tmp_path):
    """A result that only ever printed is a result nobody has — the project's own rule, and the
    first live M3 run's numbers existed solely in a console until they were transcribed by hand.
    The report also has to carry the caveats, because a bare per-lighting table reads as a
    zero-shot result when PatchCore is the full-shot ceiling."""
    calls: list = []
    _patch_summarise_deps(monkeypatch, _seeded_frame(), calls)
    results = tmp_path / "mvtec_ad2" / "vial"

    _module().summarise(results)

    report = tmp_path / "mvtec_ad2" / "vial.md"
    assert report.is_file(), "no report written"
    text = report.read_text()
    assert "patchcore_ref on vial" in text
    assert "full-shot anchor" in text, "a bare table reads as a zero-shot result"
    assert "One category is not a dataset result" in text
    assert "Resize([256, 256])" in text, "the square-resize property must travel with the number"
    assert "3 seed(s)" in text and "regular" in text and "overexposed" in text


# --- Restoring shards from Drive -----------------------------------------------------------
# The runner resumes on an existence check of a shard file, and those live on the VM that Colab
# reclaims. The first live Vial run re-fitted all three seeds on a fresh runtime while three
# perfectly good shards sat on Drive, because save_to_drive only ever copied outward.


def test_it_restores_shards_drive_has_and_this_machine_does_not(tmp_path, capsys):
    mod = _module()
    drive, results = tmp_path / "drive", tmp_path / "results"
    (drive / "shards").mkdir(parents=True)
    for name in ("a.parquet", "b.parquet"):
        (drive / "shards" / name).write_bytes(b"shard")

    mod.restore_from_drive(results, drive)

    assert sorted(p.name for p in (results / "shards").glob("*.parquet")) == ["a.parquet",
                                                                             "b.parquet"]
    assert "restored 2 shard(s)" in capsys.readouterr().out


def test_it_never_overwrites_a_local_shard_with_drives_copy(tmp_path):
    """The Drive copy came from the local one, so they agree — but a restore that can clobber is
    one nobody should run twice, and the local file is the one the current run just wrote."""
    mod = _module()
    drive, results = tmp_path / "drive", tmp_path / "results"
    (drive / "shards").mkdir(parents=True)
    (drive / "shards" / "a.parquet").write_bytes(b"from drive")
    (results / "shards").mkdir(parents=True)
    (results / "shards" / "a.parquet").write_bytes(b"local, newer")

    mod.restore_from_drive(results, drive)

    assert (results / "shards" / "a.parquet").read_bytes() == b"local, newer"


def test_restoring_from_a_drive_with_nothing_on_it_is_a_no_op(tmp_path, capsys):
    mod = _module()
    mod.restore_from_drive(tmp_path / "results", tmp_path / "never-used")
    assert "restored" not in capsys.readouterr().out


def test_summarise_skips_pixel_metrics_rather_than_dying_on_absent_maps(monkeypatch, tmp_path,
                                                                        capsys):
    """A shard restored from Drive references maps that never left the VM that computed them.
    pixel_metrics loads every map_path from disk, so a missing file would raise from inside
    np.load with no hint of why. The image-level table has to survive that."""
    import pandas as pd

    frame = _seeded_frame(seeds=(0,))
    frame = frame.assign(map_path=[str(tmp_path / "gone.npy")] * len(frame))
    calls: list = []
    _patch_summarise_deps(monkeypatch, frame, calls)

    _module().summarise(tmp_path / "results" / "vial")

    out = capsys.readouterr().out
    assert "anomaly maps are not on this machine" in out
    assert "Pixel metrics (AU-PRO, SegF1) are SKIPPED" in out
    assert calls == [[0]], "the image-level aggregation must still have run"
    seen = calls and True
    assert seen


def test_summarise_says_nothing_about_maps_when_they_are_all_present(monkeypatch, tmp_path,
                                                                     capsys):
    mod = _module()
    real = tmp_path / "map.npy"
    real.write_bytes(b"x")
    frame = _seeded_frame(seeds=(0,))
    frame = frame.assign(map_path=[str(real)] * len(frame))
    calls: list = []
    _patch_summarise_deps(monkeypatch, frame, calls)

    mod.summarise(tmp_path / "results" / "vial")

    assert "not on this machine" not in capsys.readouterr().out


def test_summarise_turns_the_memory_guard_into_a_recoverable_instruction(monkeypatch, tmp_path,
                                                                         capsys):
    """Four of eight categories are projected to exceed the 6 GB pixel-metric guard. The shards
    are already written and on Drive when it fires, so the only thing blocked is the summary —
    and the way out has to be an explicit budget checked against measured RAM, never a silent
    default, because the guard's own docstring says exactly that."""
    import vlmab.eval.aggregate as agg_mod
    import vlmab.eval.store as store_mod

    frame = _seeded_frame(seeds=(0,))
    monkeypatch.setattr(store_mod.ResultStore, "load_all", lambda self: frame)

    def exploding_aggregate(df, by=None, **kw):
        # The real wording, from aggregate.py:158 — the "about N bytes" clause is what the
        # recovery suggestion reads its budget out of, so a stub that omits it tests a path
        # the guard never takes.
        raise ValueError(
            "group meta_lighting='regular': 110,297,088 pooled pixels need about "
            "8,823,767,040 bytes (8.8 GB) at peak, which exceeds max_bytes=6,000,000,000 "
            "(6.0 GB). Aggregate a smaller group")

    monkeypatch.setattr(agg_mod, "aggregate", exploding_aggregate)

    with pytest.raises(SystemExit) as excinfo:
        _module().summarise(tmp_path / "results" / "rice")

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "shards are safe" in err
    assert "--max-bytes" in err and "--category rice" in err


def test_an_unrelated_value_error_is_not_swallowed_as_a_memory_problem(monkeypatch, tmp_path):
    """The guard is matched on its own message. A different ValueError — a pooled-seed frame,
    say — must keep its own traceback rather than being reported as a memory budget."""
    import vlmab.eval.aggregate as agg_mod
    import vlmab.eval.store as store_mod

    monkeypatch.setattr(store_mod.ResultStore, "load_all", lambda self: _seeded_frame(seeds=(0,)))
    monkeypatch.setattr(agg_mod, "aggregate",
                        lambda df, by=None, **kw: (_ for _ in ()).throw(ValueError("pools 3 seeds")))

    with pytest.raises(ValueError, match="pools 3 seeds"):
        _module().summarise(tmp_path / "results" / "rice")


def test_summarise_only_runs_no_seed_and_loads_no_backend(harness, monkeypatch, tmp_path):
    """The seed loop constructs a PatchCoreBackend per seed BEFORE the runner checks is_done, so
    even an all-done category pays for importing anomalib and building a model — roughly 3 GB
    resident. On the four categories projected to exceed the pixel-metric guard, that 3 GB is
    the difference between an aggregation that fits and one the kernel OOM-kills."""
    mod, calls, root, tmp = harness
    seen: list = []
    monkeypatch.setattr(mod, "summarise", lambda results, mb=None: seen.append(mb))

    rc = mod.main(["--category", "vial", "--root", str(root), "--no-save", "--summarise-only",
                   "--max-bytes", "8000000000", "--results", str(tmp / "results")])

    assert rc == 0
    assert calls == [], "no seed may run"
    assert seen == [8000000000], "the budget must reach summarise"


# --- Task 2 (2026-10-01): the runner works for any method with a GPU backend -----------------


class _FakeMethod:
    """A method as the runner sees it. WinCLIP has no `preprocess`; PatchCore does."""

    def __init__(self, name, seed):
        self.name, self.seed = name, seed
        self.zero_shot = name != "patchcore_ref"
        if name == "patchcore_ref":
            self.preprocess = "anomalib"


@pytest.fixture
def factory_harness(monkeypatch, harness):
    """The shared harness plus a patched factory, so a method with no backend can be asked for."""
    import vlmab.methods.gpu as gpu_mod

    mod, calls, root, tmp_path = harness
    built: list[tuple[str, int | None]] = []
    fetched: list[str] = []

    def _build(name, seed=None):
        if name not in ("patchcore_ref", "winclip"):
            raise KeyError(f"no GPU backend for {name!r}; methods with one: "
                           "['patchcore_ref', 'winclip']")
        built.append((name, seed))
        return _FakeMethod(name, seed)

    monkeypatch.setattr(gpu_mod, "build_runnable", _build)
    # These tests exercise the method-generic path, not the viability guard, which has its own
    # tests against the real factory. Without this stub they would all stop at the guard.
    monkeypatch.setattr(gpu_mod, "missing_backend", lambda name: None)
    monkeypatch.setattr(mod, "fetch", lambda category, root_, archives: fetched.append(category))
    return {"mod": mod, "calls": calls, "root": root, "tmp_path": tmp_path,
            "built": built, "fetched": fetched}


def test_two_methods_do_not_share_a_shard_directory(harness):
    mod, *_ = harness
    a = mod.default_results("patchcore_ref", "vial")
    b = mod.default_results("winclip", "vial")
    assert a != b
    assert a.parts[-2:] == ("patchcore_ref", "vial")
    assert b.parts[-2:] == ("winclip", "vial")


def test_drive_results_are_method_scoped_too(harness):
    mod, *_ = harness
    assert mod.default_drive_results("winclip", "vial").parts[-2:] == ("winclip", "vial")


def test_the_method_comes_from_the_factory_not_a_hardcoded_backend(factory_harness):
    h = factory_harness
    _run(h["mod"], h["root"], h["tmp_path"])
    assert [name for name, _ in h["built"]] == ["patchcore_ref"] * 3


def test_a_method_without_a_gpu_backend_exits_2_before_extracting(factory_harness, capsys):
    h = factory_harness
    rc = _run(h["mod"], h["root"], h["tmp_path"], "--method", "saa")
    assert rc == 2
    out = capsys.readouterr()
    assert "saa" in (out.out + out.err)
    assert h["fetched"] == [], "it must refuse before touching the archive"


def test_preprocess_is_stamped_only_when_the_method_has_one(factory_harness):
    """WinCLIP has no pre-processing identity. A null column records that; a fabricated
    "anomalib" would claim a property the method does not have."""
    h = factory_harness
    _run(h["mod"], h["root"], h["tmp_path"], "--method", "winclip")
    assert h["calls"][0]["meta"]["preprocess"] is None


def test_an_empty_category_directory_exits_2_rather_than_fitting_over_nothing(factory_harness):
    """An interrupted extraction leaves the directory present and useless."""
    h = factory_harness

    def _reject(category, root_):
        raise RuntimeError("prepare_data.py rejected the layout")

    import importlib
    monkey = pytest.MonkeyPatch()
    monkey.setattr(h["mod"], "verify_layout", _reject)
    try:
        rc = _run(h["mod"], h["root"], h["tmp_path"])
    finally:
        monkey.undo()
    assert rc == 2
    assert h["calls"] == []


# --- Task 3 (2026-10-01): a shard directory holding two methods is refused -------------------


def _write_shard(shards, method, seed=0, commit="a" * 40):
    import pandas as pd
    shards.mkdir(parents=True, exist_ok=True)
    frame = {
        "method": [method] * 2, "seed": [seed, seed], "meta_lighting": ["regular"] * 2,
        "label": [0, 1], "image_score": [0.1, 0.9], "map_path": [None, None],
    }
    if commit is not None:
        frame["commit"] = [commit] * 2
    pd.DataFrame(frame).to_parquet(
        shards / f"mvtec_ad2__{method}__vial__seed{seed}.parquet")


def test_summarise_refuses_a_directory_holding_two_methods(tmp_path):
    """Shard filenames carry the method, so two methods coexist on disk without colliding.
    Averaging them would produce a row describing no method at all."""
    mod = _module()
    for name in ("patchcore_ref", "winclip"):
        _write_shard(tmp_path / "shards", name)

    with pytest.raises(ValueError) as exc:
        mod.summarise(tmp_path)
    message = str(exc.value)
    assert "patchcore_ref" in message and "winclip" in message
    assert "pool" in message.lower()


# --- Task 4 (2026-10-01): the report's commit comes from the shards --------------------------


def _report_text(tmp_path):
    return (tmp_path.parent / f"{tmp_path.name}.md").read_text()


def test_the_report_records_the_shards_commit_not_the_machines(tmp_path):
    """2026-09-29: a Fruit Jelly report claimed 61704b4 for numbers computed at a7bc7bd."""
    mod = _module()
    _write_shard(tmp_path / "shards", "patchcore_ref",
                 commit="a7bc7bd7d13b28c7b548c3dd50997f34e0161147")
    mod.summarise(tmp_path)
    report = _report_text(tmp_path)
    assert "a7bc7bd" in report


def test_a_report_from_shards_at_two_commits_lists_both(tmp_path):
    """A resumed run mixes commits. One of them printed as THE commit is a false claim."""
    mod = _module()
    _write_shard(tmp_path / "shards", "patchcore_ref", seed=0, commit="a" * 40)
    _write_shard(tmp_path / "shards", "patchcore_ref", seed=1, commit="b" * 40)
    mod.summarise(tmp_path)
    report = _report_text(tmp_path)
    assert "aaaaaaa" in report and "bbbbbbb" in report
    assert "MIXED" in report


def test_shards_without_a_commit_column_report_unknown_rather_than_raising(tmp_path):
    """Shards predating provenance must not crash the report."""
    mod = _module()
    _write_shard(tmp_path / "shards", "patchcore_ref", commit=None)
    mod.summarise(tmp_path)
    assert "unknown" in _report_text(tmp_path)


# --- Task 5 (2026-10-01): the guard's recovery suggestion is runnable -----------------------


def test_the_recovery_suggestion_is_runnable(monkeypatch, tmp_path, capsys):
    """On Wall Plugs it suggested 9 GB for a peak it had just measured at 10.03 GB, and omitted
    --summarise-only, so following it verbatim failed twice over."""
    import vlmab.eval.aggregate as agg_mod
    mod = _module()
    needed = 10_027_008_000

    def _boom(*a, **kw):
        raise ValueError(
            f"group meta_lighting='overexposed': 125,337,600 pooled pixels need about "
            f"{needed:,} bytes (10.0 GB) at peak, which exceeds max_bytes=6,000,000,000"
        )

    monkeypatch.setattr(agg_mod, "aggregate", _boom)
    _write_shard(tmp_path / "vial" / "shards", "winclip")

    with pytest.raises(SystemExit):
        mod.summarise(tmp_path / "vial", 6_000_000_000)
    err = capsys.readouterr().err

    assert "--summarise-only" in err, "without the flag the retry reloads torch it does not need"
    assert "--method winclip" in err, "the retry must name the method it was run for"
    suggested = int(err.split("--max-bytes")[1].split()[0])
    assert suggested >= needed, (
        f"suggested {suggested:,} for a peak of {needed:,} — following it fails again"
    )


# --- Final review fixes (2026-10-02) --------------------------------------------------------


def test_the_report_names_the_method_that_produced_it(tmp_path):
    """Critical, found in review: the title was hardcoded, so a WinCLIP run wrote a committed
    report titled `patchcore_ref` — and these files are the paper's `% SOURCE:` references."""
    mod = _module()
    _write_shard(tmp_path / "shards", "winclip")
    mod.summarise(tmp_path)
    report = _report_text(tmp_path)
    assert report.splitlines()[0].startswith(f"# MVTec AD 2 — winclip on {tmp_path.name}")
    assert "patchcore_ref" not in report


def test_a_zero_shot_report_does_not_assert_patchcores_caveats(tmp_path):
    """The prose asserted what the `preprocess` column eleven lines earlier refuses to assert:
    that the method is the full-shot anchor and applies the anomalib 256x256 transform."""
    mod = _module()
    _write_shard(tmp_path / "shards", "winclip")
    mod.summarise(tmp_path)
    report = _report_text(tmp_path)
    assert "full-shot anchor" not in report
    assert "Resize([256, 256])" not in report
    assert "One category is not a dataset result" in report, "the method-neutral caveat stays"


def test_patchcores_report_keeps_both_of_its_caveats(tmp_path):
    mod = _module()
    _write_shard(tmp_path / "shards", "patchcore_ref")
    mod.summarise(tmp_path)
    report = _report_text(tmp_path)
    assert "full-shot anchor" in report and "Resize([256, 256])" in report


def test_a_method_whose_backend_does_not_exist_is_refused_before_any_download(tmp_path, capsys,
                                                                              monkeypatch):
    """Critical-adjacent, found in review: `winclip` is registered, so the name check passed,
    fetch extracted up to 10 GB, and the run died in the lazy import. No fakes here — this is
    the real factory."""
    # WinCLIP's backend exists since 2026-10-03, so a backendless method is simulated: the
    # real factory, with winclip's backend module pointed at one that does not exist.
    import vlmab.methods.gpu as gpu
    monkeypatch.setitem(gpu._BACKEND_MODULES, "winclip", "vlmab.methods.not_built_yet")
    mod = _module()
    rc = mod.main(["--category", "vial", "--method", "winclip",
                   "--root", str(tmp_path / "nope"),
                   "--results", str(tmp_path / "out"), "--no-save"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "not_built_yet" in err, (
        "the refusal must name the module that is missing, not look like a missing archive"
    )
