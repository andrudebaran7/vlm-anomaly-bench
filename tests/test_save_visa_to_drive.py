"""The rescue path for a long VisA run, tested where it is cheap to test.

Written the day a live run was 7/12 after two hours with nothing yet on Drive. Its whole job is
to be runnable mid-run, so the cases that matter are the partial ones: some shards and no report.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "save_visa_to_drive.py"


def _module():
    spec = importlib.util.spec_from_file_location("save_visa_to_drive", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _shard(dirpath: Path, name: str) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / name).write_bytes(b"not really parquet, but the script only ever copies it")


def test_it_copies_a_partial_run_without_a_report(tmp_path, capsys):
    """The case it exists for. Cell 3.5 asserts the report is there; mid-run it is not, and
    refusing then would defeat the purpose."""
    mod = _module()
    shards = tmp_path / "shards"
    for name in ("visa__patchcore_ref__candle__seed0.parquet",
                 "visa__patchcore_ref__cashew__seed0.parquet"):
        _shard(shards, name)

    rc = mod.main(["--no-mount", "--shards", str(shards),
                   "--report", str(tmp_path / "absent.md"), "--dest", str(tmp_path / "drive")])

    assert rc == 0
    copied = sorted(p.name for p in (tmp_path / "drive" / "shards").glob("*.parquet"))
    assert len(copied) == 2
    out = capsys.readouterr().out
    assert "10 of 12 objects still to go" in out
    assert "expected mid-run" in out


def test_it_takes_the_report_too_once_the_run_has_scored_itself(tmp_path, capsys):
    mod = _module()
    shards = tmp_path / "shards"
    _shard(shards, "visa__patchcore_ref__candle__seed0.parquet")
    report = tmp_path / "patchcore_visa.md"
    report.write_text("# Reproduction secondary check\n")

    rc = mod.main(["--no-mount", "--shards", str(shards), "--report", str(report),
                   "--dest", str(tmp_path / "drive")])

    assert rc == 0
    assert (tmp_path / "drive" / "patchcore_visa.md").read_text().startswith("# Reproduction")
    assert "had already scored itself" in capsys.readouterr().out


def test_it_is_idempotent(tmp_path):
    """It will be run more than once during a long session, and a second run must not fail on
    a destination that already holds the first run's copies."""
    mod = _module()
    shards = tmp_path / "shards"
    _shard(shards, "visa__patchcore_ref__candle__seed0.parquet")
    dest = tmp_path / "drive"

    assert mod.main(["--no-mount", "--shards", str(shards),
                     "--report", str(tmp_path / "absent.md"), "--dest", str(dest)]) == 0
    _shard(shards, "visa__patchcore_ref__cashew__seed0.parquet")
    assert mod.main(["--no-mount", "--shards", str(shards),
                     "--report", str(tmp_path / "absent.md"), "--dest", str(dest)]) == 0

    assert len(sorted((dest / "shards").glob("*.parquet"))) == 2


def test_it_refuses_rather_than_reporting_success_on_an_empty_run(tmp_path, capsys):
    """Exit 1, not 0. A rescue tool that prints a cheerful nothing is worse than one that fails:
    the whole reason to run it is to be told the work is safe."""
    mod = _module()
    (tmp_path / "shards").mkdir()

    rc = mod.main(["--no-mount", "--shards", str(tmp_path / "shards"),
                   "--report", str(tmp_path / "absent.md"), "--dest", str(tmp_path / "drive")])

    assert rc == 1
    assert "no shard has been written yet" in capsys.readouterr().err


def test_it_says_so_when_the_run_has_not_started(tmp_path, capsys):
    mod = _module()
    rc = mod.main(["--no-mount", "--shards", str(tmp_path / "nope"),
                   "--report", str(tmp_path / "absent.md"), "--dest", str(tmp_path / "drive")])
    assert rc == 1
    assert "does not exist" in capsys.readouterr().err


def test_the_destination_is_not_the_mvtec_folders():
    """`reproduction/` and `reproduction_3seed/` hold MVTec AD shards under filenames that carry
    no dataset name. Merging datasets into one root is caught by the gate, but only after the
    copy has already happened."""
    mod = _module()
    assert mod.DEST.name == "reproduction_visa"


def test_its_defaults_are_the_paths_the_visa_runner_actually_writes():
    """Found in review 2026-10-02: the runner's defaults became method-scoped and this script's
    did not, so a default PatchCore re-run wrote to new paths while this reported "nothing to
    copy" on the shards and copied the SEPTEMBER report to Drive as if it were the new one.

    The invariant is that the two agree for the same method — not that either path is a
    particular string.
    """
    import importlib.util

    runner_spec = importlib.util.spec_from_file_location(
        "run_visa_secondary", ROOT / "scripts" / "run_visa_secondary.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)

    mod = _module()
    assert mod.SHARDS == runner.default_results("patchcore_ref") / "shards"
    assert mod.REPORT == runner.default_out("patchcore_ref")


def test_visa_defaults_follow_the_v0221_label_for_patchcore_only():
    """A PatchCore VisA run after v0.2.21 scores through the antialiased path, so it must not
    write into (or skip on, or overwrite the report of) the old-configuration tree. WinCLIP's
    protocol did not change, so its paths stay put. Both scripts must agree, as above."""
    import importlib.util

    runner_spec = importlib.util.spec_from_file_location(
        "run_visa_secondary", ROOT / "scripts" / "run_visa_secondary.py")
    runner = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner)
    mod = _module()

    assert runner.default_results("patchcore_ref") != Path("results/reproduction/patchcore_ref/visa")
    assert runner.default_out("patchcore_ref") != Path("results/reproduction/patchcore_ref_visa.md")
    assert runner.default_results("patchcore_ref") == Path(
        "results/reproduction/patchcore_ref_v0.2.21/visa")
    assert runner.default_out("patchcore_ref") == Path(
        "results/reproduction/patchcore_ref_v0.2.21_visa.md")
    assert runner.default_results("winclip") == Path("results/reproduction/winclip/visa")
    assert runner.default_out("winclip") == Path("results/reproduction/winclip_visa.md")

    # The save script resolves --method through main(); its module defaults must agree too.
    assert mod.SHARDS == runner.default_results("patchcore_ref") / "shards"
    assert mod.REPORT == runner.default_out("patchcore_ref")
    assert mod.default_shards("winclip") == runner.default_results("winclip") / "shards"
    assert mod.default_report("winclip") == runner.default_out("winclip")
