import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from vlmab.eval import provenance
from vlmab.eval.provenance import config_hash, git_commit, gpu_name, run_meta


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})


def test_config_hash_changes_with_values():
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_config_hash_handles_nested_and_nonjson_values():
    from pathlib import Path

    h = config_hash({"root": Path("/data"), "nested": {"x": [1, 2]}})
    assert re.fullmatch(r"[0-9a-f]{12}", h)


def test_git_commit_is_sha_or_unknown():
    c = git_commit()
    assert c == "unknown" or re.fullmatch(r"[0-9a-f]{40}", c)


def test_git_commit_unknown_when_git_binary_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    assert git_commit() == "unknown"


def test_git_commit_unknown_when_not_a_repository(monkeypatch):
    class FakeCompletedProcess:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: FakeCompletedProcess()
    )
    assert git_commit() == "unknown"


def test_git_commit_unknown_when_git_times_out(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["git", "rev-parse", "HEAD"], timeout=5)

    monkeypatch.setattr(provenance.subprocess, "run", fake_run)
    assert git_commit() == "unknown"


def test_git_commit_returns_stripped_sha_on_success(monkeypatch):
    class FakeCompletedProcess:
        returncode = 0
        stdout = "abc123def456\n"

    monkeypatch.setattr(
        provenance.subprocess, "run", lambda *a, **k: FakeCompletedProcess()
    )
    assert git_commit() == "abc123def456"


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")
def test_git_commit_matches_actual_git_state_for_a_directory(tmp_path):
    """Real-environment check: establish ground truth via git itself first, so this
    can never pass for the wrong reason regardless of where the OS temp root sits.
    """
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    expected = result.stdout.strip() if result.returncode == 0 else "unknown"
    assert git_commit(tmp_path) == expected


def test_gpu_name_always_returns_a_string():
    assert isinstance(gpu_name(), str)


def test_run_meta_carries_all_provenance_fields():
    meta = run_meta({"method": "winclip"}, seed=0)
    assert set(meta) == {"config_hash", "commit", "seed", "gpu"}
    assert meta["seed"] == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git binary not available")
def test_run_meta_records_this_repo_head_regardless_of_cwd(monkeypatch, tmp_path):
    """The Colab case: the notebook's cwd is /content, not the clone.

    Resolving the commit from the process cwd records "unknown" there, or -- worse --
    a different repository's SHA if the cwd happens to be one. It must come from the
    package's own location instead.
    """
    repo_root = Path(provenance.__file__).resolve().parents[3]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        pytest.skip("package is not inside a git checkout")
    expected = result.stdout.strip()

    monkeypatch.chdir(tmp_path)
    assert run_meta({"method": "winclip"}, seed=0)["commit"] == expected


def test_package_commit_is_unknown_when_root_has_no_git(monkeypatch, tmp_path):
    """Installed non-editably, PACKAGE_REPO_ROOT lands inside the venv -- and git walks
    UP from there, so it would return some unrelated ancestor repository's SHA.
    Stamping a wrong commit is worse than stamping none: it makes a run look
    reproducible when it is not.

    The fixture reproduces exactly that: a directory with no `.git` of its own, nested
    inside a repository that does have one. Reading it naively yields the outer repo's
    real SHA, so this test has no power unless the outer repo exists.
    """
    import subprocess

    from vlmab.eval import provenance

    outer = tmp_path / "outer"
    nested = outer / "lib" / "python3.13" / "site-packages"
    nested.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=outer, check=True)
    subprocess.run(["git", "commit", "-q", "--allow-empty", "-m", "x"], cwd=outer,
                   check=True, env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                                    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
                                    "PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    assert re.fullmatch(r"[0-9a-f]{40}", git_commit(nested)), "fixture has no power"

    monkeypatch.setattr(provenance, "PACKAGE_REPO_ROOT", nested)
    assert provenance.package_commit() == "unknown"
    assert provenance.run_meta({"a": 1}, seed=0)["commit"] == "unknown"


def test_run_meta_reads_this_repo_even_when_cwd_is_elsewhere(monkeypatch, tmp_path):
    """On Colab the notebook cwd is /content, not the clone."""
    monkeypatch.chdir(tmp_path)
    commit = run_meta({"a": 1}, seed=0)["commit"]
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)
    assert commit == git_commit(Path(__file__).resolve().parents[1])
