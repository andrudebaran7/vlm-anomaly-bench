import re

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


def test_git_commit_unknown_outside_a_repo(tmp_path):
    assert git_commit(tmp_path) == "unknown"


def test_gpu_name_always_returns_a_string():
    assert isinstance(gpu_name(), str)


def test_run_meta_carries_all_provenance_fields():
    meta = run_meta({"method": "winclip"}, seed=0)
    assert set(meta) == {"config_hash", "commit", "seed", "gpu"}
    assert meta["seed"] == 0
