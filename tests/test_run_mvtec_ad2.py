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
        raise ValueError("... exceeds max_bytes=6,000,000,000 (6.0 GB). Aggregate a smaller group")

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
