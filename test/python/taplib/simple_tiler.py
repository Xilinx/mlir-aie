import numpy as np

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: simple_tiler
@construct_test
def simple_tiler():
    single_tile = TensorTiler2D.simple_tiler((3, 5))
    assert len(single_tile) == 1
    ref_tile = TensorAccessPattern(
        (3, 5), offset=0, sizes=[1, 1, 3, 5], strides=[0, 0, 5, 1]
    )
    assert single_tile[0] == ref_tile

    tiles = TensorTiler2D.simple_tiler((9, 4), (3, 2))
    assert len(tiles) == 6
    # fmt: off
    ref_access_order_tensor = np.array([
            [ 0,  1,  6,  7],
            [ 2,  3,  8,  9],
            [ 4,  5, 10, 11],
            [12, 13, 18, 19],
            [14, 15, 20, 21],
            [16, 17, 22, 23],
            [24, 25, 30, 31],
            [26, 27, 32, 33],
            [28, 29, 34, 35]])
    # fmt: on
    access_order, access_count = tiles.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 1).all()

    def offset_fn(step, _prev_offset):
        offsets = [0, 2, 12, 14, 24, 26]
        return offsets[step]

    tiles2 = TensorAccessSequence(
        (9, 4),
        num_steps=6,
        sizes=[1, 1, 3, 2],
        strides=[0, 0, 4, 1],
        offset_fn=offset_fn,
    )
    assert tiles == tiles2

    tile0_0 = TensorAccessPattern(
        (9, 4), offset=0, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1]
    )
    tile0_1 = TensorAccessPattern(
        (9, 4), offset=2, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1]
    )
    tile1_0 = TensorAccessPattern(
        (9, 4), offset=12, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1]
    )
    tile1_1 = TensorAccessPattern(
        (9, 4), offset=14, sizes=[1, 1, 3, 2], strides=[0, 0, 4, 1]
    )

    assert tiles2[0] == tile0_0
    assert tiles2[1] == tile0_1
    assert tiles2[2] == tile1_0
    assert tiles2[3] == tile1_1

    access_order, access_count = tiles2.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 1).all()

    # Check with column major iter order
    tiles_iter_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), iter_col_major=True
    )
    assert tiles_iter_col_major[0] == tile0_0
    assert tiles_iter_col_major[1] == tile1_0
    assert tiles_iter_col_major[3] == tile0_1
    assert tiles_iter_col_major[4] == tile1_1

    # fmt: off
    ref_access_order_tensor = np.array([
            [ 0,  1, 18, 19],
            [ 2,  3, 20, 21],
            [ 4,  5, 22, 23],
            [ 6,  7, 24, 25],
            [ 8,  9, 26, 27],
            [10, 11, 28, 29],
            [12, 13, 30, 31],
            [14, 15, 32, 33],
            [16, 17, 34, 35]])
    # fmt: on
    access_order, access_count = tiles_iter_col_major.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 1).all()

    tiles_tile_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True
    )
    tile0_0 = TensorAccessPattern(
        (9, 4), offset=0, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4]
    )
    tile0_1 = TensorAccessPattern(
        (9, 4), offset=2, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4]
    )
    tile1_0 = TensorAccessPattern(
        (9, 4), offset=12, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4]
    )
    tile1_1 = TensorAccessPattern(
        (9, 4), offset=14, sizes=[1, 1, 2, 3], strides=[0, 0, 1, 4]
    )
    assert tiles_tile_col_major[0] == tile0_0
    assert tiles_tile_col_major[1] == tile0_1
    assert tiles_tile_col_major[2] == tile1_0
    assert tiles_tile_col_major[3] == tile1_1

    # fmt: off
    ref_access_order_tensor = np.array([
            [ 0,  3,  6,  9],
            [ 1,  4,  7, 10],
            [ 2,  5,  8, 11],
            [12, 15, 18, 21],
            [13, 16, 19, 22],
            [14, 17, 20, 23],
            [24, 27, 30, 33],
            [25, 28, 31, 34],
            [26, 29, 32, 35]])
    # fmt: on
    access_order, access_count = tiles_tile_col_major.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 1).all()

    tiles_tile_col_major_iter_col_major = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True, iter_col_major=True
    )
    assert tiles_tile_col_major_iter_col_major[0] == tile0_0
    assert tiles_tile_col_major_iter_col_major[1] == tile1_0
    assert tiles_tile_col_major_iter_col_major[3] == tile0_1
    assert tiles_tile_col_major_iter_col_major[4] == tile1_1

    # fmt: off
    ref_access_order_tensor = np.array([
            [ 0,  3, 18, 21],
            [ 1,  4, 19, 22],
            [ 2,  5, 20, 23],
            [ 6,  9, 24, 27],
            [ 7, 10, 25, 28],
            [ 8, 11, 26, 29],
            [12, 15, 30, 33],
            [13, 16, 31, 34],
            [14, 17, 32, 35]])
    # fmt: on
    access_order, access_count = tiles_tile_col_major_iter_col_major.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 1).all()

    tiles_repeat = TensorTiler2D.simple_tiler((9, 4), (3, 2), pattern_repeat=5)
    tile_repeat0_0 = TensorAccessPattern(
        (9, 4), offset=0, sizes=[1, 5, 3, 2], strides=[0, 0, 4, 1]
    )
    assert tiles_repeat[0] == tile_repeat0_0

    # fmt: off
    ref_access_order_tensor = np.array([
            [ 24,  25,  54,  55],
            [ 26,  27,  56,  57],
            [ 28,  29,  58,  59],
            [ 84,  85, 114, 115],
            [ 86,  87, 116, 117],
            [ 88,  89, 118, 119],
            [144, 145, 174, 175],
            [146, 147, 176, 177],
            [148, 149, 178, 179]])
    # fmt: on
    access_order, access_count = tiles_repeat.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 5).all()

    tiles_repeat = TensorTiler2D.simple_tiler(
        (9, 4), (3, 2), tile_col_major=True, pattern_repeat=5
    )
    tile_repeat0_0 = TensorAccessPattern(
        (9, 4), offset=0, sizes=[1, 5, 2, 3], strides=[0, 0, 1, 4]
    )
    assert tiles_repeat[0] == tile_repeat0_0

    # fmt: off
    ref_access_order_tensor = np.array([
            [ 24,  27,  54,  57],
            [ 25,  28,  55,  58],
            [ 26,  29,  56,  59],
            [ 84,  87, 114, 117],
            [ 85,  88, 115, 118],
            [ 86,  89, 116, 119],
            [144, 147, 174, 177],
            [145, 148, 175, 178],
            [146, 149, 176, 179]])
    # fmt: on
    access_order, access_count = tiles_repeat.accesses()
    assert (access_order == ref_access_order_tensor).all()
    assert (access_count == 5).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: simple_tiler_invalid
@construct_test
def simple_tiler_invalid():
    try:
        tiles = TensorTiler2D.simple_tiler(
            (), (3, 2), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (10, 9, 4), (3, 2), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tensor dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (3, -1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Bad tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (3,), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too few tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (1, 1, 1), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Too many tile dims, should fail.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (3, 2), tile_col_major=True, pattern_repeat=0
        )
        raise ValueError("Invalid repeat.")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (4, 2), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (height)")
    except ValueError:
        # good
        pass
    try:
        tiles = TensorTiler2D.simple_tiler(
            (9, 4), (3, 3), tile_col_major=True, pattern_repeat=5
        )
        raise ValueError("Indivisible tile (width)")
    except ValueError:
        # good
        pass

    # CHECK: Pass!
    print("Pass!")
