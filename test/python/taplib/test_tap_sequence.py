import pytest
import numpy as np
from aie.helpers.taplib import TensorAccessPattern


def test_tile_sequence_simple():
    shape = (16, 8)
    tile_dims = (4, 4)

    # Using TensorAccessPattern.tile_sequence
    tap = TensorAccessPattern(shape)
    tas = tap.tile_sequence(tile_dims)

    # Using TensorTiler2D.simple_tiler
    tas_ref = TensorAccessPattern(shape).tile_sequence(tile_dims)

    # Compare
    assert len(tas) == len(tas_ref)
    for i in range(len(tas)):
        assert tas[i].offset == tas_ref[i].offset
        assert list(tas[i].sizes) == list(tas_ref[i].sizes)
        assert list(tas[i].strides) == list(tas_ref[i].strides)


def test_tile_sequence_step():
    shape = (16, 8)
    tile_dims = (4, 4)
    step_dims = (2, 2)  # Step in units of tiles.
    # If step=2, we skip every other tile.

    # Using TensorAccessPattern.tile_sequence
    tap = TensorAccessPattern(shape)
    tas = tap.tile_sequence(tile_dims, step_dims=step_dims)

    # Expected behavior:
    # Dim 0: 16//4 = 4 tiles. Step 2. Indices 0, 2. (2 steps)
    # Dim 1: 8//4 = 2 tiles. Step 2. Indices 0. (1 step)
    # Total steps: 2 * 1 = 2.

    assert len(tas) == 2

    # Step 0: (0, 0)
    # Offset 0.
    assert tas[0].offset == 0
    # Sizes: (1, 1, 4, 4) (repeat=1)
    assert list(tas[0].sizes) == [1, 1, 4, 4]
    # Strides: (2*4*8, 2*4, 8, 1) = (64, 8, 8, 1)
    # Wait, stride for outer dims is step * tile_size * stride_orig
    # Dim 0: 2 * 4 * 8 = 64.
    # Dim 1: 2 * 4 * 1 = 8.
    # But size is 1, so stride is cleaned to 0.
    assert list(tas[0].strides) == [0, 0, 8, 1]

    # Step 1: (2, 0) (Dim 0 index 2, Dim 1 index 0) -> No, dim order [0, 1].
    # Inner loop is dim 1?
    # dim_order=[0, 1].
    # get_step_indices iterates reversed(dim_order) -> 1, 0.
    # rem = step.
    # dim 1: count = 1. val = step % 1 = 0. rem = step // 1 = step.
    # dim 0: count = 2. val = step % 2.
    # So step 0 -> (0, 0).
    # Step 1 -> (1, 0).
    # Wait, indices are (idx_0, idx_1).
    # idx_0 = 1. idx_1 = 0.
    # Tile index 0: 1 * 2 = 2.
    # Tile index 1: 0 * 2 = 0.
    # Offset: 2 * 4 * 8 + 0 = 64.

    assert tas[1].offset == 64
    assert list(tas[1].sizes) == [1, 1, 4, 4]
    assert list(tas[1].strides) == [0, 0, 8, 1]


def test_tile_sequence_repeat():
    shape = (16, 8)
    tile_dims = (4, 4)
    repeat_dims = (2, 2)

    # Using TensorAccessPattern.tile_sequence
    tap = TensorAccessPattern(shape)
    tas = tap.tile_sequence(tile_dims, repeat_dims=repeat_dims)

    # Using TensorTiler2D.group_tiler
    tas_ref = TensorAccessPattern(shape).tile_sequence(
        tile_dims, repeat_dims=repeat_dims
    )

    # Compare
    assert len(tas) == len(tas_ref)
    for i in range(len(tas)):
        assert tas[i].offset == tas_ref[i].offset
        assert list(tas[i].sizes) == list(tas_ref[i].sizes)
        assert list(tas[i].strides) == list(tas_ref[i].strides)


def test_tile_sequence_pattern_repeat():
    shape = (16, 8)
    tile_dims = (4, 4)
    pattern_repeat = 2

    # Using TensorAccessPattern.tile_sequence
    tap = TensorAccessPattern(shape)
    tas = tap.tile_sequence(tile_dims, pattern_repeat=pattern_repeat)

    # Using TensorTiler2D.simple_tiler
    tas_ref = TensorAccessPattern(shape).tile_sequence(
        tile_dims, pattern_repeat=pattern_repeat
    )

    # Compare
    assert len(tas) == len(tas_ref)
    for i in range(len(tas)):
        assert tas[i].offset == tas_ref[i].offset
        assert list(tas[i].sizes) == list(tas_ref[i].sizes)
        assert list(tas[i].strides) == list(tas_ref[i].strides)
