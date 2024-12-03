import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensor_tile
@construct_test
def tensor_tile():
    # Valid
    tile = TensorAccessPattern((2, 3), 4, sizes=[1, 2], strides=[0, 1])

    # Check accessors
    assert (
        tile.tensor_dims[0] == 2
        and tile.tensor_dims[1] == 3
        and len(tile.tensor_dims) == 2
    )
    assert tile.offset == 4
    assert tile.sizes == [1, 2]
    assert tile.strides == [0, 1]
    assert tile.transformation_dims == [(1, 0), (2, 1)]
    access_order, access_count = tile.accesses()
    assert (
        access_order == np.array([[-1, -1, -1], [-1, 0, 1]], dtype=access_order.dtype)
    ).all()
    assert (
        access_count == np.array([[0, 0, 0], [0, 1, 1]], dtype=access_count.dtype)
    ).all()

    tile2 = TensorAccessPattern((2, 3), 4, sizes=[1, 2], strides=[0, 1])
    assert tile2 == tile
    assert tile.compare_access_orders(tile2)

    tile3 = TensorAccessPattern((2, 3), 2, sizes=[1, 2], strides=[0, 1])
    assert tile3 != tile
    assert not tile.compare_access_orders(tile3)

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensor_tile_invalid
@construct_test
def tensor_tile_invalid():

    # Bad tensor dims
    try:
        tile = TensorAccessPattern((), 4, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad dims (no dims)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((0, 1), 4, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad dims (first dim 0)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((1, 0), 4, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad dims (second dim 0)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((-1, 1), 4, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad dims (first dim negative)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((1, -1), 4, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad dims (second dim negative)")
    except ValueError:
        # Good
        pass

    # Bad offset
    try:
        tile = TensorAccessPattern((2, 3), -1, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad offset (negative)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[1, 2], strides=[0, 1])
        raise Exception("Should fail, bad offset (too large)")
    except ValueError:
        # Good
        pass

    # Bad sizes
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[-1], strides=[1])
        raise Exception("Should fail, size (negative)")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[0], strides=[1])
        raise Exception("Should fail, size (zero)")
    except ValueError:
        # Good
        pass

    # Bad sizes + strides
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[], strides=[])
        raise Exception("Should fail, size and stride empty")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[1], strides=[0, 1])
        raise Exception("Should fail, sizes and strides uneven dimensions")
    except ValueError:
        # Good
        pass
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[1, 1], strides=[0])
        raise Exception("Should fail, sizes and strides uneven dimensions 2")
    except ValueError:
        # Good
        pass

    # Bad strides
    try:
        tile = TensorAccessPattern((2, 3), 2 * 3, sizes=[1], strides=[-1])
        raise Exception("Should fail, bad stride (negative)")
    except ValueError:
        # Good
        pass

    # CHECK: Pass!
    print("Pass!")
