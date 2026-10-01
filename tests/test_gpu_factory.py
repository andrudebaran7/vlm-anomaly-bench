"""The GPU-backend factory: one lazy builder per method, refusing by name before any cost.

The registry builds CPU adapters with no backend on purpose, so CI never imports torch. This
module is where a backend gets attached, which means every import here is function-local.
"""
from pathlib import Path

import pytest

from vlmab.methods import gpu

ROOT = Path(__file__).resolve().parents[1]


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
    source = Path(gpu.__file__).read_text()
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            assert "torch" not in stripped and "anomalib" not in stripped, (
                f"module-scope GPU import: {stripped!r}"
            )


def test_every_builder_name_is_a_registered_method():
    from vlmab.methods.registry import available
    for name in gpu.gpu_builders():
        assert name in available(), f"{name} has a GPU builder but is not in the registry"


# --- Final review fix (2026-10-02): registration is not viability ---------------------------


def test_a_registered_builder_whose_backend_module_is_absent_is_reported():
    """Found in review: `winclip` is in gpu_builders(), so the runners' up-front check passed
    and the run died inside the lazy import — after fetching and extracting up to 10 GB."""
    assert gpu.missing_backend("winclip") == "vlmab.methods.winclip_backend"


def test_a_method_whose_backend_exists_reports_nothing_missing():
    assert gpu.missing_backend("patchcore_ref") is None


def test_importing_the_factory_and_checking_viability_imports_no_torch():
    """In a SUBPROCESS, because `sys.modules` is shared across a pytest session: another test's
    fixture imports patchcore_backend, so an in-process assertion passes alone and fails in the
    suite. The subprocess also makes this stronger than the source scan above — it would catch a
    module-scope import that pulls anomalib in transitively.

    `find_spec` locates a module without executing it, so the check is safe to call before a
    download; one that imported the backend to test it would defeat its own purpose.
    """
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import vlmab.methods.gpu as g;"
        "g.missing_backend('patchcore_ref');"
        "g.missing_backend('winclip');"
        "bad=[m for m in sys.modules if m.split('.')[0] in ('torch','anomalib')"
        " or m=='vlmab.methods.patchcore_backend'];"
        "print('LEAKED:'+','.join(sorted(bad)) if bad else 'CLEAN')"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "CLEAN", out.stdout.strip()


def test_an_unknown_name_has_no_backend_to_miss():
    with pytest.raises(KeyError):
        gpu.missing_backend("not_a_method")
