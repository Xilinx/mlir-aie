import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: simple_tiler
@construct_test
def simple_tiler():

    # TODO:
    tiles = TensorTiler2D.simple_tiler((9, 4), (3, 2))
    assert len(tiles) == 6

    def offset_fn(step, _prev_offset):
        offsets = [0, 2, 12, 14, 24, 26]
        return offsets[step]

    tiles2 = TensorTileSequence(
        (9, 4),
        num_steps=6,
        sizes=[1, 1, 3, 2],
        strides=[0, 0, 4, 1],
        offset_fn=offset_fn,
    )
    assert tiles == tiles2

    tile0_0 = TensorTile((9, 4), offset=0, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
    tile0_1 = TensorTile((9, 4), offset=2, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
    tile1_0 = TensorTile((9, 4), offset=12, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])
    tile1_1 = TensorTile((9, 4), offset=14, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1])

    assert tiles[0] == tile0_0
    assert tiles[1] == tile0_1
    assert tiles[2] == tile1_0
    assert tiles[3] == tile1_1

    # Check with column major iter order
    tiles_iter_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), iter_col_major=True
    )
    assert tiles_iter_col_major[0] == tile0_0
    assert tiles_iter_col_major[1] == tile1_0
    assert tiles_iter_col_major[3] == tile0_1
    assert tiles_iter_col_major[4] == tile1_1

    tiles_tile_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True
    )
    tile0_0 = TensorTile((9, 4), offset=0, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4])
    tile0_1 = TensorTile((9, 4), offset=2, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4])
    tile1_0 = TensorTile((9, 4), offset=12, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4])
    tile1_1 = TensorTile((9, 4), offset=14, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4])
    assert tiles_tile_col_major[0] == tile0_0
    assert tiles_tile_col_major[1] == tile0_1
    assert tiles_tile_col_major[2] == tile1_0
    assert tiles_tile_col_major[3] == tile1_1

    tiles_tile_col_major_iter_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True, iter_col_major=True
    )
    assert tiles_tile_col_major_iter_col_major[0] == tile0_0
    assert tiles_tile_col_major_iter_col_major[1] == tile1_0
    assert tiles_tile_col_major_iter_col_major[3] == tile0_1
    assert tiles_tile_col_major_iter_col_major[4] == tile1_1

    tiles_repeat = TensorTiler2D.simple_tiler((9, 4), (3, 2), pattern_repeat=5)
    tile_repeat0_0 = TensorTile(
        (9, 4), offset=0, sizes=[1, 5, 3, 2], strides=[0, 0, 4, 1]
    )
    assert tiles_repeat[0] == tile_repeat0_0

    tiles_repeat = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True, pattern_repeat=5
    )
    tile_repeat0_0 = TensorTile(
        (9, 4), offset=0, sizes=[1, 5, 2, 3], strides=[0, 0, 1, 4]
    )
    assert tiles_repeat[0] == tile_repeat0_0

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: simple_tiler_invalid
@construct_test
def simple_tiler_invalid():
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (), (3, 2), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (10, 9, 4), (3, 2), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (3, -1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (3,), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (1, 1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (3, 2), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Invalid repeat.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (4, 2), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Indivisible tile (height)")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.simple_tiler(
            (9, 4), (3, 3), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Indivisible tile (width)")
    except ValueError:
        # good
        pass

    # CHECK: Pass!
    print("Pass!")
