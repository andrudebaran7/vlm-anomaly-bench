"""The pre-processing choice, which is the only part of the anomalib backend CPU can reach.

The backend itself needs anomalib, torch and a GPU, so it is never constructed here. What IS
testable — and is the part that decides a reproduction verdict — is which transform each named
choice stands for, and the patch count that follows from it. The 2026-09-16 gate failure turned
on exactly this: anomalib 2.6.0 pre-processes to 256x256 with no crop, where classic PatchCore
does Resize(256) -> CenterCrop(224), and the two differ by 31% more patches per image.
"""
import pytest

from vlmab.methods.patchcore_backend import PREPROCESS_CHOICES, preprocess_spec


def test_the_classic_spec_is_the_papers_resize_then_centre_crop():
    spec = preprocess_spec("classic")
    assert spec["resize"] == 256
    assert spec["center_crop"] == 224


def test_the_anomalib_spec_resizes_the_full_frame_and_never_crops():
    """anomalib 2.6.0's own PreProcessor, recorded in the 2026-08-21 Colab session:
    Resize([256, 256]) + Normalize, no CenterCrop."""
    spec = preprocess_spec("anomalib")
    assert spec["resize"] == 256
    assert spec["center_crop"] is None


def test_each_spec_carries_the_patch_count_the_colab_cell_checks_against():
    """The measurable consequence, and the only thing a Colab cell can assert cheaply before
    spending 40 minutes on a fit. Measured 2026-09-16: every category yielded 102.4 coreset
    points per training image at ratio 0.1, i.e. 1024 patches, i.e. a 32x32 grid from 256x256.
    A 224x224 input gives 28x28 = 784."""
    assert preprocess_spec("anomalib")["patches_per_image"] == 1024
    assert preprocess_spec("classic")["patches_per_image"] == 784


def test_an_unknown_preprocessing_name_is_refused_and_names_the_choices():
    """A typo in a Colab cell must fail in a millisecond, not after a 40-minute fit produced
    numbers from whichever branch the typo fell through to."""
    with pytest.raises(ValueError, match="clasic"):
        preprocess_spec("clasic")
    assert set(PREPROCESS_CHOICES) == {"anomalib", "classic"}


def test_an_invalid_preprocess_is_refused_before_anomalib_is_imported():
    """The one thing about the backend CLASS that CPU can check. Constructing it here gets as
    far as the name check and no further — anomalib is not installed in CI, so if the check
    ran after the lazy import this would raise ImportError instead, and a typo in Colab would
    surface only after the install, the weights and the fit."""
    from vlmab.methods.patchcore_backend import PatchCoreBackend

    with pytest.raises(ValueError, match="clasic"):
        PatchCoreBackend(preprocess="clasic")


def test_classic_is_refused_before_anomalib_is_imported():
    """Protocol v0.2.21: the cropped transform makes the map cover only the central 224 px, and
    it is then stretched over the frame. An old notebook cell (3b.x, 3c) that still passes
    preprocess='classic' must fail at once, naming the version, not after the install."""
    from vlmab.methods.patchcore_backend import PatchCoreBackend

    with pytest.raises(ValueError, match="v0.2.21"):
        PatchCoreBackend(preprocess="classic")


def test_the_backend_routes_forward_to_the_antialiased_transform(monkeypatch):
    """anomalib and torch are absent here, so fake the three modules the constructor imports.
    What is checked is the wiring: forward() (which score() calls) must use the antialiased
    training transform, not the antialias-off export transform."""
    import sys
    import types
    from types import SimpleNamespace

    class Resize:
        def __init__(self, antialias):
            self.antialias = antialias

    class Normalize:
        pass

    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms

    class Patchcore:
        def __init__(self, **kwargs):
            self.pre_processor = SimpleNamespace(
                transform=Compose([Resize(True), Normalize()]),
                export_transform=Compose([Resize(False), Normalize()]),
            )

    class Engine:
        def __init__(self, **kwargs):
            pass

    anomalib = types.ModuleType("anomalib")
    models = types.ModuleType("anomalib.models")
    models.Patchcore = Patchcore
    engine = types.ModuleType("anomalib.engine")
    engine.Engine = Engine
    anomalib.models, anomalib.engine = models, engine
    monkeypatch.setitem(sys.modules, "anomalib", anomalib)
    monkeypatch.setitem(sys.modules, "anomalib.models", models)
    monkeypatch.setitem(sys.modules, "anomalib.engine", engine)

    from vlmab.methods.patchcore_backend import PatchCoreBackend

    b = PatchCoreBackend(device="cpu")
    pre = b._model.pre_processor
    assert pre.export_transform is pre.transform
    assert b.resize_antialias is True


def test_the_config_records_the_preprocessing_the_backend_defaults_to():
    """Two places state the pre-processing — the YAML a reader consults and the default a Colab
    cell gets — and nothing else would notice them drifting apart. The 2026-09-16 numbers came
    from the anomalib transform, so that is what the config must say they came from."""
    import yaml
    from pathlib import Path
    from vlmab.methods.patchcore_backend import PatchCoreBackend

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "methods" / "patchcore_ref.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    import inspect
    default = inspect.signature(PatchCoreBackend.__init__).parameters["preprocess"].default
    assert cfg["preprocess"] == default == "anomalib"


def test_the_anomalib_version_is_a_resolved_pin_not_a_range():
    """Phase 4 of the Colab playbook: freeze provenance. `>=1.1` describes a wish; 2.6.0 is what
    every measured number in this repo actually ran on, and the five API contradictions recorded
    in the session notes are specific to it."""
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "methods" / "patchcore_ref.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["anomalib_version"] == "2.6.0"


def test_the_config_records_the_antialiased_resize():
    """Protocol §3 v0.2.21: the corrected configuration is hard-wired, and the config says so."""
    import yaml
    from pathlib import Path

    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "methods" / "patchcore_ref.yaml"
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["resize_antialias"] is True
