from pathlib import Path

import numpy as np
import pytest

from mvtec_tree import CONDITIONS, build_category
from vlmab.datasets.mvtec_ad2 import LABEL_UNKNOWN, MVTecAD2, lighting_condition


@pytest.fixture
def dataset(tmp_path):
    build_category(tmp_path, "vial")
    build_category(tmp_path, "can")
    return MVTecAD2(tmp_path)


def test_lighting_condition_parses_the_filename_suffix():
    assert lighting_condition(Path("/x/000_regular.png")) == "regular"
    assert lighting_condition(Path("/x/013_shift_1.png")) == "shift_1"
    assert lighting_condition(Path("/x/007_mixed.png")) == "mixed"


def test_lighting_condition_rejects_a_name_with_no_condition():
    with pytest.raises(ValueError):
        lighting_condition(Path("/x/000.png"))


def test_categories_are_the_directories_sorted(dataset):
    assert dataset.categories() == ["can", "vial"]


def test_missing_root_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        MVTecAD2(tmp_path / "nope")


def test_unknown_split_is_rejected(dataset):
    with pytest.raises(ValueError):
        list(dataset.samples("test", "vial"))


def test_train_and_validation_are_defect_free(dataset):
    for split in ("train", "validation"):
        samples = list(dataset.samples(split, "vial"))
        assert samples, split
        assert {s.label for s in samples} == {0}
        assert {s.mask_path for s in samples} == {None}
        assert {s.meta["lighting"] for s in samples} == {"regular"}


def test_test_public_labels_good_and_bad(dataset):
    samples = list(dataset.samples("test_public", "vial"))
    good = [s for s in samples if s.label == 0]
    bad = [s for s in samples if s.label == 1]
    assert len(good) == 2 * len(CONDITIONS)
    assert len(bad) == 2 * len(CONDITIONS)
    assert all(s.mask_path is None for s in good)
    assert all(s.mask_path is not None for s in bad)


def test_test_public_covers_every_lighting_condition(dataset):
    samples = list(dataset.samples("test_public", "vial"))
    assert {s.meta["lighting"] for s in samples} == set(CONDITIONS)


def test_bad_samples_pair_with_the_right_mask_file(dataset):
    for s in dataset.samples("test_public", "vial"):
        if s.label == 1:
            assert s.mask_path.name == f"{s.image_path.stem}_mask.png"
            assert s.mask_path.is_file()


def test_a_missing_mask_is_an_error_not_a_silent_none(tmp_path):
    cat = build_category(tmp_path, "vial")
    next((cat / "test_public" / "ground_truth" / "bad").glob("*.png")).unlink()
    with pytest.raises(FileNotFoundError):
        list(MVTecAD2(tmp_path).samples("test_public", "vial"))


def test_private_splits_are_unlabelled_and_flat(dataset):
    for split, cond in (("test_private", "regular"), ("test_private_mixed", "mixed")):
        samples = list(dataset.samples(split, "vial"))
        assert samples, split
        assert {s.label for s in samples} == {LABEL_UNKNOWN}
        assert {s.mask_path for s in samples} == {None}
        assert {s.meta["lighting"] for s in samples} == {cond}


def test_samples_without_a_category_span_every_category(dataset):
    cats = {s.category for s in dataset.samples("test_public")}
    assert cats == {"can", "vial"}


def test_load_image_returns_three_channel_uint8_from_grayscale(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 0)
    img = dataset.load_image(sample)
    assert img.dtype == np.uint8
    assert img.ndim == 3 and img.shape[2] == 3
    # The grayscale channel is replicated, not dropped or rescaled.
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()
    assert img[0, 0, 0] == 10


def test_load_image_passes_rgb_through_unchanged(tmp_path):
    """Vial is grayscale, but that is verified for Vial only — RGB must work too."""
    build_category(tmp_path, "rgbcat", mode="RGB")
    ds = MVTecAD2(tmp_path)
    sample = next(s for s in ds.samples("test_public", "rgbcat") if s.label == 0)
    img = ds.load_image(sample)
    assert img.shape[2] == 3 and img[0, 0, 0] == 10


def test_load_mask_is_binary_and_matches_the_image_shape(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 1)
    mask = dataset.load_mask(sample)
    img = dataset.load_image(sample)
    assert mask.shape == img.shape[:2]
    assert set(np.unique(mask)) <= {0, 1}
    assert mask[0, 0] == 1 and mask[-1, -1] == 0


def test_load_mask_is_none_when_there_is_no_ground_truth(dataset):
    sample = next(s for s in dataset.samples("test_public", "vial") if s.label == 0)
    assert dataset.load_mask(sample) is None
