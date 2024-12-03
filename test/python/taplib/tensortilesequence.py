import numpy as np

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensor_tile_sequence
@construct_test
def tensor_tile_sequence():

    empty_tiles = TensorAccessSequence((2, 2), 0)
    assert len(empty_tiles) == 0
    ref_access_order = np.array([[-1, -1], [-1, -1]])
    ref_access_count = np.array([[0, 0], [0, 0]])
    access_order, access_count = empty_tiles.accesses()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    def offset_fn(step, _prev_offset):
        return step

    tiles = TensorAccessSequence(
        (2, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert len(tiles) == 4
    ref_access_order = np.array([[0, 1], [2, 3]])
    ref_access_count = np.array([[1, 1], [1, 1]])
    access_order, access_count = tiles.accesses()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    tile = TensorAccessPattern((2, 2), offset=2, sizes=[1, 1], strides=[1, 1])
    assert tile in tiles
    assert tiles[2] == tile
    tiles2 = list(iter(tiles))
    assert tiles2[2] == tile

    del tiles[2]
    assert not (tile in tiles)
    tiles.insert(2, tile)
    assert tile in tiles

    tile2 = TensorAccessPattern((3, 3), offset=2, sizes=[1, 1], strides=[1, 1])
    assert not (tile2 in tiles)

    tiles3 = TensorAccessSequence(
        (2, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert tiles == tiles3
    tiles4 = TensorAccessSequence(
        (2, 2), 3, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert tiles != tiles4
    ref_access_order = np.array([[0, 1], [2, -1]])
    ref_access_count = np.array([[1, 1], [1, 0]])
    access_order, access_count = tiles4.accesses()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    tiles4_copy = TensorAccessSequence.from_taps(tiles4)
    assert tiles4_copy == tiles4
    assert tiles4_copy.compare_access_orders(tiles4)

    access_order, access_count = tiles4_copy.accesses()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensor_tile_sequence_invalid
@construct_test
def tensor_tile_sequence_invalid():
    def offset_fn(step, _prev_offset):
        return step

    try:
        tiles = TensorAccessSequence(
            (0, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad dims")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence(
            (2, 2), -1, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad num steps")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence(
            (2, 2), 1, sizes=[1, 0], strides=[0, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad sizes")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence(
            (2, 2), 1, sizes=[1, 1], strides=[-1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad strides")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence((2, 2), 1, strides=[1, 1], offset_fn=offset_fn)
        raise Exception("Should fail, missing sizes")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence((2, 2), 1, sizes=[1, 1], offset_fn=offset_fn)
        raise Exception("Should fail, missing strides")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorAccessSequence((2, 2), 1, strides=[1, 1], sizes=[1, 1])
        raise Exception("Should fail, missing offset")
    except ValueError:
        # Good
        pass

    tiles = TensorAccessSequence((2, 3), 1, offset=0, strides=[0, 1], sizes=[1, 1])
    try:
        tiles.append(
            TensorAccessPattern((3, 2), offset=0, strides=[0, 1], sizes=[1, 1])
        )
        raise Exception("Should not be able to add tile with inconsistent tensor dim")
    except ValueError:
        # Good
        pass

    try:
        TensorAccessSequence.from_taps([])
        raise Exception("Should not be able to create sequence from no tiles")
    except ValueError:
        # Good
        pass

    # CHECK: Pass!
    print("Pass!")
