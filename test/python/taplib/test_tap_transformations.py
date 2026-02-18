import pytest
import numpy as np
from aie.helpers.taplib import TensorAccessPattern


def test_identity():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Check dimensions
    assert list(tap.sizes) == [4, 4]
    assert list(tap.strides) == [4, 1]

    # Check access order
    access_order = tap.access_order()
    expected = np.arange(16).reshape(4, 4)
    assert np.array_equal(access_order, expected)


def test_permute():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Transpose
    tap_T = tap.permute((1, 0))
    assert list(tap_T.sizes) == [4, 4]
    assert list(tap_T.strides) == [1, 4]

    access_order = tap_T.access_order()
    # Access order should be transposed in terms of values?
    # No, access_order[i, j] is the sequence number of element (i, j).
    # If we iterate (j, i), then (0, 0) is 0, (0, 1) is 1...
    # Wait, access_order returns an array of shape `tensor_dims`.
    # `tensor_dims` is `(4, 4)`.
    # `tap_T` iterates `(j, i)`.
    # So `(0, 0)` is accessed 0th.
    # `(0, 1)` (row 0, col 1) is accessed... when?
    # `tap_T` iterates `j` (outer) then `i` (inner).
    # `j=0, i=0` -> `(0, 0)`. Seq 0.
    # `j=0, i=1` -> `(1, 0)`. Seq 1.
    # ...
    # `j=1, i=0` -> `(0, 1)`. Seq 4.
    # So `access_order[0, 1]` should be 4.
    # `access_order[1, 0]` should be 1.

    expected = np.array([[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]])
    assert np.array_equal(access_order, expected)


def test_split_dim():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Split dim 0 (size 4) into (2, 2)
    tap_split = tap.split_dim(0, 2)
    assert list(tap_split.sizes) == [2, 2, 4]
    assert list(tap_split.strides) == [8, 4, 1]  # 2*4, 4, 1

    # Access order should be same as identity, just different view dimensions?
    # `access_order` returns array of shape `tensor_dims` (4, 4).
    # The sequence of access is the same (row-major).
    # So access_order should be same as identity.
    access_order = tap_split.access_order()
    expected = np.arange(16).reshape(4, 4)
    assert np.array_equal(access_order, expected)


def test_tile():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Tile (2, 2)
    tap_tiled = tap.tile((2, 2))
    # Expected sizes: (2, 2, 2, 2) (M//2, N//2, 2, 2)
    # Expected strides: (8, 2, 4, 1) (M//2 * 2*4, N//2 * 2? No)
    # Split M -> M//2, 2. Strides 8, 4.
    # Split N -> N//2, 2. Strides 8, 4, 2, 1.
    # Permute -> (M//2, N//2, 2, 2). Strides (8, 2, 4, 1).

    assert list(tap_tiled.sizes) == [2, 2, 2, 2]
    assert list(tap_tiled.strides) == [8, 2, 4, 1]

    # Access order:
    # Iterates blocks (2x2), then inside blocks.
    # Block (0, 0): (0,0), (0,1), (1,0), (1,1). Seq 0, 1, 2, 3.
    # Block (0, 1): (0,2), (0,3), (1,2), (1,3). Seq 4, 5, 6, 7.
    # ...

    access_order = tap_tiled.access_order()
    expected = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]])
    assert np.array_equal(access_order, expected)


def test_untile():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)
    tap_tiled = tap.tile((2, 2))

    # Untile back to original
    tap_untiled = tap_tiled.untile((2, 2))
    assert list(tap_untiled.sizes) == [4, 4]
    assert list(tap_untiled.strides) == [4, 1]

    access_order = tap_untiled.access_order()
    expected = np.arange(16).reshape(4, 4)
    assert np.array_equal(access_order, expected)

    # Untile without merge
    tap_permuted = tap_tiled.untile((2, 2), merge=False)
    # Should be (M//2, 2, N//2, 2) -> (2, 2, 2, 2)
    # Strides (8, 4, 2, 1)
    assert list(tap_permuted.sizes) == [2, 2, 2, 2]
    assert list(tap_permuted.strides) == [8, 4, 2, 1]


def test_slice():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Slice rows 1:3
    tap_slice = tap.slice(0, 1, 2)
    assert list(tap_slice.sizes) == [2, 4]
    assert tap_slice.offset == 4  # 1 * 4

    # Access order:
    # Only elements in rows 1 and 2 are accessed.
    # Row 1: 0, 1, 2, 3
    # Row 2: 4, 5, 6, 7
    # Others: -1

    access_order = tap_slice.access_order()
    expected = np.full((4, 4), -1)
    expected[1:3, :] = np.arange(8).reshape(2, 4)
    assert np.array_equal(access_order, expected)


def test_subview():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Subview (1, 1) size (2, 2)
    tap_sub = tap.subview((1, 1), (2, 2))
    assert list(tap_sub.sizes) == [2, 2]
    assert tap_sub.offset == 5  # 1*4 + 1

    access_order = tap_sub.access_order()
    expected = np.full((4, 4), -1)
    expected[1:3, 1:3] = np.arange(4).reshape(2, 2)
    assert np.array_equal(access_order, expected)


def test_resize():
    shape = (4, 4)
    tap = TensorAccessPattern(shape)

    # Resize to (2, 2) (top-left corner)
    tap_resize = tap.resize((2, 2))
    assert list(tap_resize.sizes) == [2, 2]
    assert tap_resize.offset == 0

    access_order = tap_resize.access_order()
    expected = np.full((4, 4), -1)
    expected[0:2, 0:2] = np.arange(4).reshape(2, 2)
    assert np.array_equal(access_order, expected)


def test_compose():
    # Inner: 2x2 tile in 4x4
    inner = TensorAccessPattern((4, 4)).tile((2, 2))
    # Outer: Select first block (0, 0)
    # Inner sizes: (2, 2, 2, 2)
    # Outer should match inner sizes.
    # Let's select block (0, 0).
    # Outer pattern on (2, 2, 2, 2).
    # We want to access only indices where dim0=0, dim1=0.
    # This is slice(0, 0, 1).slice(1, 0, 1).
    outer = TensorAccessPattern((2, 2, 2, 2)).slice(0, 0, 1).slice(1, 0, 1)

    composed = inner.compose(outer)
    # Should access only the first block (0, 0)
    # Elements (0,0), (0,1), (1,0), (1,1)

    access_order = composed.access_order()
    expected = np.full((4, 4), -1)
    expected[0:2, 0:2] = np.arange(4).reshape(2, 2)
    assert np.array_equal(access_order, expected)
