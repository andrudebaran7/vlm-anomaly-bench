"""The GPU-backend factory: one lazy builder per method, refusing by name before any cost.

The registry builds CPU adapters with no backend on purpose, so CI never imports torch. This
module is where a backend gets attached, which means every import here is function-local.
"""
from pathlib import Path

import pytest

from vlmab.methods import gpu


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
