import numpy as np
import pytest

from vlmab.methods.postprocess import normalise_to_unit, upsample_to


def test_upsample_hits_the_exact_target_size():
    out = upsample_to(np.zeros((7, 7), dtype=np.float32), (1400, 1900))
    assert out.shape == (1400, 1900)
    assert out.dtype == np.float32


def test_upsample_is_nearest_block_not_interpolated():
    coarse = np.array([[0.0, 1.0]], dtype=np.float32)  # 1x2
    out = upsample_to(coarse, (1, 4))
    # Left half maps to the 0 cell, right half to the 1 cell; no in-between values.
    assert out.tolist() == [[0.0, 0.0, 1.0, 1.0]]


def test_upsample_preserves_a_single_hot_cell_as_a_block():
    coarse = np.zeros((7, 7), dtype=np.float32)
    coarse[1, 2] = 1.0  # cell "B3"
    out = upsample_to(coarse, (70, 70))
    assert set(np.unique(out)) == {0.0, 1.0}
    assert out[15, 25] == 1.0                 # inside the B3 block (rows 10-19, cols 20-29)
    assert out[0, 0] == 0.0


def test_upsample_handles_a_target_not_divisible_by_the_grid():
    coarse = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = upsample_to(coarse, (10, 10))       # 10 is not a multiple of 3
    assert out.shape == (10, 10)
    assert out.min() == 0.0 and out.max() == 8.0  # no new values invented


def test_normalise_maps_min_to_zero_and_max_to_one():
    out = normalise_to_unit(np.array([2.0, 4.0, 6.0], dtype=np.float32))
    assert out.tolist() == [0.0, 0.5, 1.0]
    assert out.dtype == np.float32


def test_normalise_a_constant_array_is_all_zero_not_nan():
    out = normalise_to_unit(np.full((4, 4), 3.0, dtype=np.float32))
    assert (out == 0.0).all()


def test_normalise_rejects_non_finite_values():
    with pytest.raises(ValueError):
        normalise_to_unit(np.array([0.0, np.nan, 1.0], dtype=np.float32))
