"""The MVTec AD classic gate runner. PatchCore's ran from notebook cells; WinCLIP needs a script.

Fifteen categories at one or three seeds, writing the shards `reproduction_gate.py` scores. Same
fake-injection style as the other runner tests: none of this runs on CPU, and the defects that
hurt are the ones that appear after the expensive part has already been paid for.
"""
import importlib.util
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_mvtec_ad.py"

FIFTEEN = ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
           "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"]


def _module():
    spec = importlib.util.spec_from_file_location("run_mvtec_ad", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def harness(monkeypatch, tmp_path):
    import vlmab.datasets.mvtec_ad as ds_mod
    import vlmab.eval.runner as runner_mod
    import vlmab.methods.gpu as gpu_mod

    calls: list[dict] = []
    stores: list[object] = []
    built: list[tuple[str, int | None]] = []

    class _FakeMethod:
        def __init__(self, name, seed):
            self.name, self.seed = name, seed
            self.zero_shot = name != "patchcore_ref"
            if name == "patchcore_ref":
                self.preprocess = "anomalib"
                self.resize_antialias = True

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
    monkeypatch.setattr(ds_mod, "MVTecAD",
                        lambda root: types.SimpleNamespace(root=root,
                                                           categories=lambda: list(FIFTEEN)))

    def _fake_run_evaluation(dataset, method, store, meta, **kw):
        stores.append(store)
        calls.append({"method": method, "meta": dict(meta), **kw})
        return []

    monkeypatch.setattr(runner_mod, "run_evaluation", _fake_run_evaluation)

    mod = _module()

    def _main(argv):
        return mod.main([*argv, "--root", str(tmp_path / "data"),
                         "--results", str(tmp_path / "out")])

    return {"mod": mod, "main": _main, "calls": calls, "stores": stores, "built": built}


def test_all_fifteen_categories_are_the_default(harness):
    harness["main"](["--method", "winclip", "--seeds", "0"])
    scored = [c for call in harness["calls"] for c in call["categories"]]
    assert scored == FIFTEEN, "the published 91.8 is a mean over fifteen; a subset is not it"


def test_every_category_gets_its_own_freshly_built_method(harness):
    """MEASURED 2026-10-21 (docs/next-steps.md): one PatchCore backend reused across categories
    fits only the first. anomalib 2.6.0 reuses the Engine's finished Trainer and skips `fit()`
    once `_is_fitted` is set, so zipper scored 26.71 after bottle against 97.82 alone, and the
    v0.2.21 gate came back at ~60. The old notebook cells built one backend per category; the
    script must too, for every method, at every seed."""
    harness["main"](["--seeds", "0", "1"])
    calls = harness["calls"]
    assert [call["categories"] for call in calls] == [[c] for c in FIFTEEN] * 2
    assert len(harness["built"]) == 30
    assert len({id(call["method"]) for call in calls}) == 30
    assert [call["method"].seed for call in calls] == [0] * 15 + [1] * 15


def test_it_scores_the_test_split_and_fits_on_train(harness):
    harness["main"](["--method", "winclip", "--seeds", "0"])
    kw = harness["calls"][0]
    assert kw["split"] == "test" and kw["fit_split"] == "train"


def test_a_subset_warns_that_the_gate_needs_all_fifteen(harness, capsys):
    harness["main"](["--method", "winclip", "--seeds", "0",
                     "--categories", "bottle", "cable"])
    out = capsys.readouterr().out
    assert "15" in out and "probe" in out.lower()


def test_the_method_comes_from_the_factory(harness):
    harness["main"](["--method", "winclip", "--seeds", "0"])
    assert harness["built"][0][0] == "winclip"


def test_patchcore_is_the_default_method(harness):
    harness["main"](["--seeds", "0"])
    assert harness["built"][0][0] == "patchcore_ref"


def test_a_method_without_a_backend_exits_2(harness, capsys):
    assert harness["main"](["--method", "saa", "--seeds", "0"]) == 2
    assert "saa" in capsys.readouterr().err
    assert harness["built"] == []


def test_three_seeds_are_the_default_because_section_6_requires_them(harness):
    harness["main"](["--method", "winclip"])
    # One build per category per seed (see the fresh-method test), so 15 of each, in order.
    assert [seed for _, seed in harness["built"]] == [0] * 15 + [1] * 15 + [2] * 15


def test_a_repeated_seed_is_refused(harness):
    assert harness["main"](["--method", "winclip", "--seeds", "0", "0"]) == 2


def test_shards_are_written_under_a_method_scoped_reproduction_path(harness):
    """The DEFAULT path is method-scoped, for the reason run_mvtec_ad2.default_results gives.
    That the store is rooted at <results>/shards is a separate claim, asserted below."""
    mod = harness["mod"]
    assert mod.default_results("winclip").parts[-2:] == ("winclip", "mvtec_ad")


def test_preprocess_is_stamped_only_when_the_method_has_one(harness):
    harness["main"](["--method", "winclip", "--seeds", "0"])
    assert harness["calls"][0]["meta"]["preprocess"] is None


def test_the_shard_records_antialiasing_and_runtime_versions(harness, monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: {"anomalib": "2.6.0",
                                                     "torch": "2.11.0+cu128"}[name])
    harness["main"](["--seeds", "0"])
    meta = harness["calls"][0]["meta"]
    assert meta["preprocess"] == "anomalib"
    assert meta["resize_antialias"] is True
    assert meta["anomalib_version"] == "2.6.0" and meta["torch_version"] == "2.11.0+cu128"


def test_runtime_versions_are_none_for_a_missing_package(harness, monkeypatch):
    import importlib.metadata as md

    def _missing(name):
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(md, "version", _missing)
    harness["main"](["--method", "winclip", "--seeds", "0"])
    meta = harness["calls"][0]["meta"]
    assert meta["resize_antialias"] is None
    assert meta["anomalib_version"] is None and meta["torch_version"] is None


def test_a_method_whose_backend_does_not_exist_is_refused_up_front(tmp_path, capsys, monkeypatch):
    """Found in review: registration is not viability."""
    # WinCLIP's backend exists since 2026-10-03, so a backendless method is simulated: the
    # real factory, with winclip's backend module pointed at one that does not exist.
    import vlmab.methods.gpu as gpu
    monkeypatch.setitem(gpu._BACKEND_MODULES, "winclip", "vlmab.methods.not_built_yet")
    mod = _module()
    rc = mod.main(["--method", "winclip", "--root", str(tmp_path / "nope"),
                   "--results", str(tmp_path / "out")])
    assert rc == 2
    assert "not_built_yet" in capsys.readouterr().err


def test_the_store_is_rooted_at_the_shards_subdirectory(harness, tmp_path):
    """Found in review: the earlier version of this test asserted only the shape of
    default_results() while its docstring claimed to assert the store's own root, so nothing
    verified that run_one_seed points ResultStore at <results>/shards."""
    harness["main"](["--method", "winclip", "--seeds", "0"])
    store = harness["stores"][0]
    assert Path(store.root) == tmp_path / "out" / "shards"


def test_default_results_uses_the_v0221_label_for_patchcore():
    mod = _module()
    assert mod.default_results("patchcore_ref") == Path(
        "results/reproduction/patchcore_ref_v0.2.21/mvtec_ad")
    assert mod.default_results("winclip") == Path("results/reproduction/winclip/mvtec_ad")
