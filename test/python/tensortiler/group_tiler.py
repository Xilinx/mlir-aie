import numpy as np

from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: group_tiler
@construct_test
def group_tiler():
    tiles = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2), tile_dims=(3, 2), tile_group_dims=(5, 7)
    )
    assert len(tiles) == 3 * 2
    tile0_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=0, sizes=[5, 7, 3, 2], strides=[84, 2, 28, 1]
    )
    assert tiles[0] == tile0_0
    tile0_1 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=14, sizes=[5, 7, 3, 2], strides=[84, 2, 28, 1]
    )
    assert tiles[1] == tile0_1
    tile1_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=420, sizes=[5, 7, 3, 2], strides=[84, 2, 28, 1]
    )
    assert tiles[2] == tile1_0

    # iter_col_major
    tiles_col_iter = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        iter_col_major=True,
    )
    assert tiles_col_iter[0] == tile0_0
    assert tiles_col_iter[1] == tile1_0
    assert tiles_col_iter[3] == tile0_1

    # tile_col_major
    tiles_tile_col_major = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
    )
    tile0_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=0, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]
    )
    assert tiles_tile_col_major[0] == tile0_0
    tile0_1 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=14, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]
    )
    assert tiles_tile_col_major[1] == tile0_1
    tile1_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=420, sizes=[1, 5, 14, 3], strides=[0, 84, 1, 28]
    )
    assert tiles_tile_col_major[2] == tile1_0

    # iter_col_major and tile_col_major
    tiles_tile_col_major_col_iter = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        iter_col_major=True,
        tile_col_major=True,
    )
    assert tiles_tile_col_major_col_iter[0] == tile0_0
    assert tiles_tile_col_major_col_iter[1] == tile1_0
    assert tiles_tile_col_major_col_iter[3] == tile0_1

    # tile_col_major and pattern_repeat
    tiles_tile_col_major_pattern_repeat = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        pattern_repeat=2,
    )
    assert tiles_tile_col_major_pattern_repeat[0] == TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=0, sizes=[2, 5, 14, 3], strides=[0, 84, 1, 28]
    )

    # tile_group_col_major
    tiles_group_col_major = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_group_col_major=True,
    )
    tile0_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=0, sizes=[1, 7, 15, 2], strides=[0, 2, 28, 1]
    )
    assert tiles_group_col_major[0] == tile0_0
    tile0_1 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=14, sizes=[1, 7, 15, 2], strides=[0, 2, 28, 1]
    )
    assert tiles_group_col_major[1] == tile0_1
    tile1_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=420, sizes=[1, 7, 15, 2], strides=[0, 2, 28, 1]
    )
    assert tiles_group_col_major[2] == tile1_0

    # tile_group_col_major and tile_col_major
    tiles_group_col_major = TensorTiler2D.group_tiler(
        (3 * 5 * 3, 2 * 7 * 2),
        tile_dims=(3, 2),
        tile_group_dims=(5, 7),
        tile_col_major=True,
        tile_group_col_major=True,
    )
    tile0_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=0, sizes=[7, 5, 2, 3], strides=[2, 84, 1, 28]
    )
    assert tiles_group_col_major[0] == tile0_0
    tile0_1 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=14, sizes=[7, 5, 2, 3], strides=[2, 84, 1, 28]
    )
    assert tiles_group_col_major[1] == tile0_1
    tile1_0 = TensorTile(
        (3 * 5 * 3, 2 * 7 * 2), offset=420, sizes=[7, 5, 2, 3], strides=[2, 84, 1, 28]
    )
    assert tiles_group_col_major[2] == tile1_0

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: group_tiler_invalid
@construct_test
def group_tiler_invalid():
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (), (3, 2), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (10, 9, 4), (3, 2), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, -1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3,), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (1, 1, 1), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, 2), (1, 1), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Invalid repeat.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (4, 2), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (height)")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, 3), (1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (width)")
    except ValueError:
        # good
        pass

    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, 2), (1,), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, 2), (1, -1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (9, 4), (3, 2), (1, 1, 1), tile_col_major=True
        )
        raise ValueError("Too many tile group dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (18, 8), (3, 2), (2, 3), tile_col_major=True
        )
        raise ValueError(
            "Indivisible by tile repeat width (but without allow_partial)."
        )
    except ValueError:
        # good
        pass
    try:
        tiles_repeat = TensorTiler2D.group_tiler(
            (18, 8), (3, 2), (4, 2), tile_col_major=True
        )
        raise ValueError(
            "Indivisible by tile repeat height (but without allow_partial)."
        )
    except ValueError:
        # good
        pass

    # CHECK: Pass!
    print("Pass!")
