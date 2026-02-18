import sys
import os
import pytest
import numpy as np

# Add python directory to path to import helpers directly
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../python"))

from helpers.taplib import TensorAccessPattern


def test_compose_transpose():
    # Original tensor 2x3
    # TAP1: Identity
    tap1 = TensorAccessPattern((2, 3), 0, (2, 3), (3, 1))

    # TAP2: Transpose (3x2)
    # Maps (j, i) in 3x2 to (i, j) in 2x3
    # Strides in 2x3 flattened view (which is same as original)
    # Stride for dim 0 (size 3): corresponds to col step in 2x3 -> 1
    # Stride for dim 1 (size 2): corresponds to row step in 2x3 -> 3
    tap2 = TensorAccessPattern((2, 3), 0, (3, 2), (1, 3))

    # Compose: tap2 on tap1
    # Should result in a TAP that behaves like tap2
    comp = tap1.compose(tap2)

    assert tuple(comp.sizes) == (3, 2)
    assert tuple(comp.strides) == (1, 3)
    assert comp.offset == 0

    # Verify accesses
    # (0, 0) -> 0
    # (0, 1) -> 3
    # (1, 0) -> 1
    # (1, 1) -> 4
    # (2, 0) -> 2
    # (2, 1) -> 5
    accesses = list(comp.access_generator())
    expected = [0, 3, 1, 4, 2, 5]
    assert accesses == expected


def test_split_dim():
    # 4x4 tensor
    tap = TensorAccessPattern((4, 4), 0, (4, 4), (4, 1))

    # Split dim 0 (size 4) into 2x2
    split = tap.split_dim(0, 2)

    # New sizes: (2, 2, 4)
    assert tuple(split.sizes) == (2, 2, 4)
    # New strides: (2*4, 4, 1) = (8, 4, 1)
    assert tuple(split.strides) == (8, 4, 1)

    # Split dim 2 (size 4) into 2x2
    split2 = split.split_dim(2, 2)

    # New sizes: (2, 2, 2, 2)
    assert tuple(split2.sizes) == (2, 2, 2, 2)
    # New strides: (8, 4, 2*1, 1) = (8, 4, 2, 1)
    assert tuple(split2.strides) == (8, 4, 2, 1)


def test_permute():
    # 2x2x2x2
    tap = TensorAccessPattern((4, 4), 0, (2, 2, 2, 2), (8, 4, 2, 1))

    # Permute to (0, 2, 1, 3) -> (2, 2, 2, 2)
    # Strides: (8, 2, 4, 1)
    perm = tap.permute((0, 2, 1, 3))

    assert tuple(perm.sizes) == (2, 2, 2, 2)
    assert tuple(perm.strides) == (8, 2, 4, 1)


def test_micro_tile_construction():
    # Construct the pattern (m//r, k//s, r, s) from (m, k)
    m, k = 64, 64
    r, s = 4, 8

    # 1. Identity
    tap = TensorAccessPattern((m, k), 0, (m, k), (k, 1))

    # 2. Split m -> m//r, r
    tap = tap.split_dim(0, m // r)
    # Sizes: (m//r, r, k)
    # Strides: (r*k, k, 1)

    # 3. Split k -> k//s, s
    tap = tap.split_dim(2, k // s)
    # Sizes: (m//r, r, k//s, s)
    # Strides: (r*k, k, s*1, 1) = (r*k, k, s, 1)

    # 4. Permute to (m//r, k//s, r, s)
    # Current dims: 0, 1, 2, 3
    # Target: 0, 2, 1, 3
    tap = tap.permute((0, 2, 1, 3))

    assert tuple(tap.sizes) == (m // r, k // s, r, s)
    assert tuple(tap.strides) == (r * k, s, k, 1)

    # Verify transformation dims match expected
    expected_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    assert tap.transformation_dims == expected_dims


def test_tile_method():
    # Test tile method which combines split and permute
    m, k = 64, 64
    r, s = 4, 8

    # Identity
    tap = TensorAccessPattern((m, k), 0, (m, k), (k, 1))

    # Tile by (r, s)
    # Should produce (m//r, k//s, r, s)
    tiled = tap.tile((r, s))

    assert tuple(tiled.sizes) == (m // r, k // s, r, s)
    # Strides:
    # m//r -> r*k
    # k//s -> s*1
    # r -> k
    # s -> 1
    # Order: (r*k, s, k, 1)
    assert tuple(tiled.strides) == (r * k, s, k, 1)


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
