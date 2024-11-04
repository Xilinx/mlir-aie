import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensor_tile_sequence
@construct_test
def tensor_tile_sequence():
    # Valid
    def offset_fn(step, _prev_offset):
        return step

    tiles = TensorTileSequence(
        (2, 2), 4, sizes=[1, 1], strides=[1, 1], offset_fn=offset_fn
    )
    assert len(tiles) == 4

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

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensor_tile_sequence_invalid
@construct_test
def tensor_tile_sequence_invalid():
    # Valid
    def offset_fn(step, _prev_offset):
        return step

    # Bad tensor dims
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
        tiles = TensorTileSequence((2, 2), 0, strides=[1, 1], sizes=[1, 1])
        raise Exception("Should fail, missing offset")
    except ValueError:
        # Good
        pass

    # CHECK: Pass!
    print("Pass!")
