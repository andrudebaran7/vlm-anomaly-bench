"""The antialiased-forward helper shared by the anomalib backends (WinCLIP v0.2.20, PatchCore v0.2.21).

CPU-only: the helper only inspects attributes, so stand-ins for torchvision's transforms suffice.
Its refusal messages must name the CALLING method's protocol entry, so a PatchCore failure never
points a reader at WinCLIP's v0.2.20.
"""
from types import SimpleNamespace

import pytest

from vlmab.methods.anomalib_preprocess import use_antialiased_forward


class Resize:   # named like torchvision's, which is what the helper inspects
    def __init__(self, antialias):
        self.antialias = antialias


class Normalize:
    pass


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms


def _fake_pre(antialias_in_transform=True):
    return SimpleNamespace(
        transform=Compose([Resize(antialias_in_transform), Normalize()]),
        export_transform=Compose([Resize(False), Normalize()]),
    )


def test_forward_is_routed_to_the_antialiased_transform():
    pre = _fake_pre()
    use_antialiased_forward(pre, protocol_ref="protocol §3 v0.2.21")
    assert pre.export_transform is pre.transform


def test_refusal_names_the_callers_protocol_entry():
    with pytest.raises(RuntimeError, match="v0.2.21"):
        use_antialiased_forward(
            _fake_pre(antialias_in_transform=False), protocol_ref="protocol §3 v0.2.21"
        )


def test_routing_refuses_a_transform_with_no_resize():
    pre = SimpleNamespace(transform=Compose([Normalize()]), export_transform=None)
    with pytest.raises(RuntimeError, match="Resize") as exc:
        use_antialiased_forward(pre, protocol_ref="protocol §3 v0.2.21")
    assert "v0.2.21" in str(exc.value)
