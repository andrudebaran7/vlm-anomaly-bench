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
    return mod.main(["--category", "vial", "--root", str(root), "--no-mount", "--no-save",
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
    rc = mod.main(["--category", "fabric", "--root", str(root), "--no-mount", "--no-save",
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
    rc = mod.main(["--category", "vial", "--root", str(root), "--no-mount",
                   "--results", str(tmp / "results"), "--drive-results", str(tmp / "drive")])
    assert rc == 0
    assert saved == [1, 2, 3], "one copy per finished seed"
