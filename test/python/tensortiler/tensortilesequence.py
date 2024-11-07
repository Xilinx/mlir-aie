import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensor_tile_sequence
@construct_test
def tensor_tile_sequence():

    empty_tiles = TensorTileSequence((2, 2), 0)
    assert len(empty_tiles) == 0
    ref_access_order = np.array([[-1, -1], [-1, -1]])
    ref_access_count = np.array([[0, 0], [0, 0]])
    access_order, access_count = empty_tiles.access_tensors()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    def offset_fn(step, _prev_offset):
        return step

    tiles = TensorTileSequence(
        (2, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert len(tiles) == 4
    ref_access_order = np.array([[0, 1], [2, 3]])
    ref_access_count = np.array([[1, 1], [1, 1]])
    access_order, access_count = tiles.access_tensors()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    tile = TensorTile((2, 2), offset=2, sizes=[1, 1], strides=[1, 1])
    assert tile in tiles
    assert tiles[2] == tile
    tiles2 = list(iter(tiles))
    assert tiles2[2] == tile

    del tiles[2]
    assert not (tile in tiles)
    tiles.insert(2, tile)
    assert tile in tiles

    tile2 = TensorTile((3, 3), offset=2, sizes=[1, 1], strides=[1, 1])
    assert not (tile2 in tiles)

    tiles3 = TensorTileSequence(
        (2, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert tiles == tiles3
    tiles4 = TensorTileSequence(
        (2, 2), 3, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert tiles != tiles4
    ref_access_order = np.array([[0, 1], [2, -1]])
    ref_access_count = np.array([[1, 1], [1, 0]])
    access_order, access_count = tiles4.access_tensors()
    assert (access_order == ref_access_order).all()
    assert (access_count == ref_access_count).all()

    tiles4_copy = TensorTileSequence.from_tiles(tiles4)
    assert tiles4_copy == tiles4
    access_order, access_count = tiles4_copy.access_tensors()
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
        tiles = TensorTileSequence(
            (0, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad dims")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence(
            (2, 2), -1, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad num steps")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence(
            (2, 2), 1, sizes=[1, 0], strides=[0, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad sizes")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence(
            (2, 2), 1, sizes=[1, 1], strides=[-1, 1], offset_fn=offset_fn
        )
        raise Exception("Should fail, bad strides")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence((2, 2), 1, strides=[1, 1], offset_fn=offset_fn)
        raise Exception("Should fail, missing sizes")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence((2, 2), 1, sizes=[1, 1], offset_fn=offset_fn)
        raise Exception("Should fail, missing strides")
    except ValueError:
        # Good
        pass
    try:
        tiles = TensorTileSequence((2, 2), 1, strides=[1, 1], sizes=[1, 1])
        raise Exception("Should fail, missing offset")
    except ValueError:
        # Good
        pass

    tiles = TensorTileSequence((2, 3), 1, offset=0, strides=[0, 1], sizes=[1, 1])
    try:
        tiles.append(TensorTile((3, 2), offset=0, strides=[0, 1], sizes=[1, 1]))
        raise Exception("Should not be able to add tile with inconsistent tensor dim")
    except ValueError:
        # Good
        pass

    try:
        TensorTileSequence.from_tiles([])
        raise Exception("Should not be able to create sequence from no tiles")
    except ValueError:
        # Good
        pass

    # CHECK: Pass!
    print("Pass!")
