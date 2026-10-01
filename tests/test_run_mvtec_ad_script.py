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

    def _build(name, seed=None):
        if name not in ("patchcore_ref", "winclip"):
            raise KeyError(f"no GPU backend for {name!r}; methods with one: "
                           "['patchcore_ref', 'winclip']")
        built.append((name, seed))
        return _FakeMethod(name, seed)

    monkeypatch.setattr(gpu_mod, "build_runnable", _build)
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
    assert len(harness["calls"][0]["categories"]) == 15, (
        "the published 91.8 is a mean over fifteen; a subset is not it"
    )


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
    assert [seed for _, seed in harness["built"]] == [0, 1, 2]


def test_a_repeated_seed_is_refused(harness):
    assert harness["main"](["--method", "winclip", "--seeds", "0", "0"]) == 2


def test_shards_are_written_under_a_method_scoped_reproduction_path(harness):
    """run_evaluation takes a store, not a path, so assert on the store's own root."""
    mod = harness["mod"]
    assert mod.default_results("winclip").parts[-2:] == ("winclip", "mvtec_ad")


def test_preprocess_is_stamped_only_when_the_method_has_one(harness):
    harness["main"](["--method", "winclip", "--seeds", "0"])
    assert harness["calls"][0]["meta"]["preprocess"] is None
