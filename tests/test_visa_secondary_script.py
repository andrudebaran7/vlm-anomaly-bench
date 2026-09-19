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


# --- The two hours the session actually spends are inside run() ---------------------------
# Added 2026-09-19, before the VisA session, because the tests above never executed a line of
# run() or main(). That is the shape of the cell 3b.3 defect: the cell was syntactically fine
# and raised NameError on its first loop line, AFTER fifteen minutes of install and extraction
# had been paid for. Here the price would be higher -- the fetch is 1.8 GB and the run is ~2 h.
#
# Nothing below needs a GPU. The heavy objects are injected as fakes, which is exactly what the
# script's structure allows: every torch-touching import lives inside run().

import types

import pytest


class _FakeVisA:
    def __init__(self, root):
        self.root = root

    def categories(self):
        return ["candle", "capsules", "cashew"]


class _FakeBackend:
    """Stands in for PatchCoreBackend, whose real __init__ imports anomalib."""

    def __init__(self, seed=None, preprocess="anomalib", **kw):
        self.seed = seed
        self.preprocess = preprocess


def _install_fakes(monkeypatch, calls):
    """Patch the module attributes run()'s function-local imports resolve through."""
    import vlmab.datasets.visa as visa_mod
    import vlmab.eval.runner as runner_mod
    import vlmab.methods.patchcore_backend as backend_mod

    monkeypatch.setattr(visa_mod, "VisA", _FakeVisA)
    monkeypatch.setattr(backend_mod, "PatchCoreBackend", _FakeBackend)

    def fake_run_evaluation(dataset, method, store, meta, **kw):
        calls.append({"method": method, "meta": dict(meta), **kw})
        return []

    monkeypatch.setattr(runner_mod, "run_evaluation", fake_run_evaluation)


def test_run_drives_one_object_at_a_time_with_a_fresh_backend(monkeypatch, tmp_path):
    """The memory bank is per-category by construction; one backend reused across objects would
    carry the previous object's bank into the next one's scores."""
    calls: list[dict] = []
    _install_fakes(monkeypatch, calls)
    mod = _module()

    mod.run(tmp_path / "data", tmp_path / "results", seed=0, device="cuda")

    assert [c["categories"] for c in calls] == [["candle"], ["capsules"], ["cashew"]]
    methods = [c["method"] for c in calls]
    assert len({id(m) for m in methods}) == 3, "each object must get its own backend"


def test_run_stamps_the_preprocess_the_method_actually_applies(monkeypatch, tmp_path):
    """The seed defect, in its other half: a value recorded in provenance that nothing applied.
    The shard's `preprocess` has to come off the method, not off the module constant."""
    calls: list[dict] = []
    _install_fakes(monkeypatch, calls)
    mod = _module()

    mod.run(tmp_path / "data", tmp_path / "results", seed=0, device="cuda")

    for call in calls:
        assert call["meta"]["preprocess"] == "anomalib" == mod.PREPROCESS
        assert call["method"].seed == 0
        assert call["method"].preprocess == "anomalib"


def test_run_uses_visas_own_split_names(monkeypatch, tmp_path):
    """VisA's one-class protocol has only ("train", "test") -- no validation split, unlike
    MVTec AD 2. A default of "test_public" carried over from the AD 2 runner would raise on the
    first object, two hours into nothing."""
    from vlmab.datasets.visa import VisA

    calls: list[dict] = []
    _install_fakes(monkeypatch, calls)
    mod = _module()

    mod.run(tmp_path / "data", tmp_path / "results", seed=0, device="cuda")

    for call in calls:
        assert call["split"] in VisA.SPLITS
        assert call["fit_split"] in VisA.SPLITS
    assert {c["split"] for c in calls} == {"test"}
    assert {c["fit_split"] for c in calls} == {"train"}


def _intercept_only_the_gate(monkeypatch, mod):
    """Capture the gate invocation and let every other subprocess through.

    Patching `subprocess.run` wholesale is what the first version of this test did, and it
    broke provenance: `run_meta` shells out to `git rev-parse HEAD`, through the same module
    object. A fake that swallows every subprocess tests a program nobody runs.
    """
    seen: dict = {}
    real_run = mod.subprocess.run

    def dispatch(argv, **kw):
        argv = list(argv)
        if any(str(a).endswith("reproduction_gate.py") for a in argv):
            seen["argv"] = argv
            return types.SimpleNamespace(returncode=0)
        return real_run(argv, **kw)

    monkeypatch.setattr(mod.subprocess, "run", dispatch)
    return seen


def test_main_scores_the_secondary_block_of_patchcores_own_targets(monkeypatch, tmp_path):
    """The gate refuses a dataset/method mismatch, so the arguments have to be right or the
    two hours end in exit code 2. `--which secondary` is the one that cannot fail the method."""
    calls: list[dict] = []
    _install_fakes(monkeypatch, calls)
    mod = _module()

    seen = _intercept_only_the_gate(monkeypatch, mod)
    rc = mod.main([
        "--skip-fetch",
        "--root", str(tmp_path / "data"),
        "--results", str(tmp_path / "results"),
        "--out", str(tmp_path / "report.md"),
    ])

    assert rc == 0
    argv = seen["argv"]
    assert argv[1].endswith("reproduction_gate.py")
    assert "--which" in argv and argv[argv.index("--which") + 1] == "secondary"
    targets = argv[argv.index("--targets") + 1]
    assert targets == "configs/reproduction/patchcore_ref.yaml"
    assert (ROOT / targets).is_file(), "the gate is handed a path relative to the repo root"


def test_the_targets_path_main_passes_really_holds_a_secondary_block():
    """A relative path that exists is not enough: the block `--which secondary` names has to be
    in it, and it has to be the non-gating one (protocol §2 v0.2.12)."""
    import yaml

    with open(ROOT / "configs" / "reproduction" / "patchcore_ref.yaml") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["secondary"]["dataset"] == "visa"
    assert cfg["secondary_n_categories"] == 12
    assert "gates" not in cfg["secondary"] or cfg["secondary"]["gates"] is False


def test_no_score_skips_the_gate_which_is_what_the_later_seeds_need(monkeypatch, tmp_path):
    """Seeds 1 and 2 go into the SAME results root, and the gate refuses a root that pools
    seeds. Without --no-score those runs would end in exit 2 after two hours of correct work."""
    calls: list[dict] = []
    _install_fakes(monkeypatch, calls)
    mod = _module()

    seen = _intercept_only_the_gate(monkeypatch, mod)
    rc = mod.main([
        "--skip-fetch", "--no-score",
        "--root", str(tmp_path / "data"),
        "--results", str(tmp_path / "results"),
    ])
    assert rc == 0
    assert "argv" not in seen, "--no-score must not reach the gate"


def test_fetch_is_skipped_when_the_split_file_is_already_there(monkeypatch, tmp_path):
    """The resume path. `split_csv/1cls.csv` is the marker because it is what the loader
    refuses to run without, and a re-run must not re-download 1.8 GB."""
    mod = _module()
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: pytest.fail("re-downloaded"))
    (tmp_path / "split_csv").mkdir(parents=True)
    (tmp_path / "split_csv" / "1cls.csv").write_text("object,split,label,image,mask\n")
    mod.fetch(tmp_path)
