import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tile_repeat1
@construct_test
def tile_repeat1():
    # Tile repeat whole tensor, tile row wise
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    REPEAT_COUNT = 3
    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH)
    tile = next(tiler.tile_iter(tile_repeat=REPEAT_COUNT))
    access_order, access_count = tile.access_tensors()
    assert (access_count == REPEAT_COUNT).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15],
            [ 16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31],
            [ 32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47],
            [ 48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63],
            [ 64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79],
            [ 80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95],
            [ 96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            [112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127],
            [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143],
            [144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159],
            [160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175],
            [176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191],
            [192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207],
            [208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223],
            [224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239],
            [240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (
        TENSOR_HEIGHT * TENSOR_WIDTH * (REPEAT_COUNT - 1)
    )
    assert (access_order == reference_order).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat2
@construct_test
def tile_repeat2():
    # Tile repeat whole tensor, tile col wise
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    REPEAT_COUNT = 6
    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, tensor_col_major=True)
    tile = next(tiler.tile_iter(tile_repeat=REPEAT_COUNT))
    access_order, access_count = tile.access_tensors()
    assert (access_count == REPEAT_COUNT).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,  16,  32,  48,  64,  80,  96, 112, 128, 144, 160, 176, 192, 208, 224, 240],
            [  1,  17,  33,  49,  65,  81,  97, 113, 129, 145, 161, 177, 193, 209, 225, 241],
            [  2,  18,  34,  50,  66,  82,  98, 114, 130, 146, 162, 178, 194, 210, 226, 242],
            [  3,  19,  35,  51,  67,  83,  99, 115, 131, 147, 163, 179, 195, 211, 227, 243],
            [  4,  20,  36,  52,  68,  84, 100, 116, 132, 148, 164, 180, 196, 212, 228, 244],
            [  5,  21,  37,  53,  69,  85, 101, 117, 133, 149, 165, 181, 197, 213, 229, 245],
            [  6,  22,  38,  54,  70,  86, 102, 118, 134, 150, 166, 182, 198, 214, 230, 246],
            [  7,  23,  39,  55,  71,  87, 103, 119, 135, 151, 167, 183, 199, 215, 231, 247],
            [  8,  24,  40,  56,  72,  88, 104, 120, 136, 152, 168, 184, 200, 216, 232, 248],
            [  9,  25,  41,  57,  73,  89, 105, 121, 137, 153, 169, 185, 201, 217, 233, 249],
            [ 10,  26,  42,  58,  74,  90, 106, 122, 138, 154, 170, 186, 202, 218, 234, 250],
            [ 11,  27,  43,  59,  75,  91, 107, 123, 139, 155, 171, 187, 203, 219, 235, 251],
            [ 12,  28,  44,  60,  76,  92, 108, 124, 140, 156, 172, 188, 204, 220, 236, 252],
            [ 13,  29,  45,  61,  77,  93, 109, 125, 141, 157, 173, 189, 205, 221, 237, 253],
            [ 14,  30,  46,  62,  78,  94, 110, 126, 142, 158, 174, 190, 206, 222, 238, 254],
            [ 15,  31,  47,  63,  79,  95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (
        TENSOR_HEIGHT * TENSOR_WIDTH * (REPEAT_COUNT - 1)
    )
    assert (access_order == reference_order).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat3
@construct_test
def tile_repeat3():
    # Tile repeat, iter row wise
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH)
    tiles = list(tiler.tile_iter(tile_repeat=REPEAT_COUNT))
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (access_count[0:TILE_HEIGHT, 0:TILE_WIDTH] == REPEAT_COUNT).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   1,   2,   3], 
            [  4,   5,   6,   7], 
            [  8,   9,  10,  11],
            [ 12,  13,  14,  15],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (TILE_HEIGHT * TILE_WIDTH * (REPEAT_COUNT - 1))
    assert (
        access_order[0:TILE_HEIGHT, 0:TILE_WIDTH] == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[2]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[0:TILE_HEIGHT, TILE_WIDTH * 2 : TILE_WIDTH * 3] == REPEAT_COUNT
    ).all()
    assert (
        access_order[0:TILE_HEIGHT, TILE_WIDTH * 2 : TILE_WIDTH * 3] == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat4
@construct_test
def tile_repeat4():
    # Tile repeat, iter col wise
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH)
    tiles = list(tiler.tile_iter(tile_repeat=REPEAT_COUNT, col_major=True))
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (access_count[0:TILE_HEIGHT, 0:TILE_WIDTH] == REPEAT_COUNT).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   1,   2,   3], 
            [  4,   5,   6,   7], 
            [  8,   9,  10,  11],
            [ 12,  13,  14,  15],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (TILE_HEIGHT * TILE_WIDTH * (REPEAT_COUNT - 1))
    assert (
        access_order[0:TILE_HEIGHT, 0:TILE_WIDTH] == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[2]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[TILE_HEIGHT * 2 : TILE_HEIGHT * 3, 0:TILE_WIDTH] == REPEAT_COUNT
    ).all()
    assert (
        access_order[TILE_HEIGHT * 2 : TILE_HEIGHT * 3, 0:TILE_WIDTH] == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat5
@construct_test
def tile_repeat5():
    # Tile repeat, iter col wise and tile col major
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    tiler = TensorTiler2D(
        TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH, tile_col_major=True
    )
    tiles = list(tiler.tile_iter(tile_repeat=REPEAT_COUNT, col_major=True))
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (access_count[0:TILE_HEIGHT, 0:TILE_WIDTH] == REPEAT_COUNT).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   4,   8,  12], 
            [  1,   5,   9,  13], 
            [  2,   6,  10,  14],
            [  3,   7,  11,  15],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (TILE_HEIGHT * TILE_WIDTH * (REPEAT_COUNT - 1))
    assert (
        access_order[0:TILE_HEIGHT, 0:TILE_WIDTH] == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[2]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[TILE_HEIGHT * 2 : TILE_HEIGHT * 3, 0:TILE_WIDTH] == REPEAT_COUNT
    ).all()
    assert (
        access_order[TILE_HEIGHT * 2 : TILE_HEIGHT * 3, 0:TILE_WIDTH] == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat6
@construct_test
def tile_repeat6():
    # Tile repeat, tile group width
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    TILE_GROUP_WIDTH = 2
    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH)
    tiles = list(
        tiler.tile_iter(tile_repeat=REPEAT_COUNT, tile_group_width=TILE_GROUP_WIDTH)
    )
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[0:TILE_HEIGHT, 0 : TILE_WIDTH * TILE_GROUP_WIDTH] == REPEAT_COUNT
    ).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   1,   2,   3,  16,  17,  18,  19], 
            [  4,   5,   6,   7,  20,  21,  22,  23], 
            [  8,   9,  10,  11,  24,  25,  26,  27],
            [ 12,  13,  14,  15,  28,  29,  30,  31],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (
        TILE_HEIGHT * TILE_WIDTH * TILE_GROUP_WIDTH * (REPEAT_COUNT - 1)
    )
    assert (
        access_order[0:TILE_HEIGHT, 0 : TILE_WIDTH * TILE_GROUP_WIDTH]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[1]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[
            0:TILE_HEIGHT,
            TILE_WIDTH * TILE_GROUP_WIDTH : TILE_WIDTH * TILE_GROUP_WIDTH * 2,
        ]
        == REPEAT_COUNT
    ).all()
    assert (
        access_order[
            0:TILE_HEIGHT,
            TILE_WIDTH * TILE_GROUP_WIDTH : TILE_WIDTH * TILE_GROUP_WIDTH * 2,
        ]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat7
@construct_test
def tile_repeat7():
    # Tile repeat and tile group width with tile col major
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    TILE_GROUP_WIDTH = 2
    tiler = TensorTiler2D(
        TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH, tile_col_major=True
    )
    tiles = list(
        tiler.tile_iter(tile_group_width=TILE_GROUP_WIDTH, tile_repeat=REPEAT_COUNT)
    )
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[0:TILE_HEIGHT, 0 : TILE_WIDTH * TILE_GROUP_WIDTH] == REPEAT_COUNT
    ).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   4,   8,  12,  16,  20,  24,  28], 
            [  1,   5,   9,  13,  17,  21,  25,  29], 
            [  2,   6,  10,  14,  18,  22,  26,  30],
            [  3,   7,  11,  15,  19,  23,  27,  31],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (
        TILE_HEIGHT * TILE_WIDTH * TILE_GROUP_WIDTH * (REPEAT_COUNT - 1)
    )
    assert (
        access_order[0:TILE_HEIGHT, 0 : TILE_WIDTH * TILE_GROUP_WIDTH]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[1]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[
            0:TILE_HEIGHT,
            TILE_WIDTH * TILE_GROUP_WIDTH : TILE_WIDTH * TILE_GROUP_WIDTH * 2,
        ]
        == REPEAT_COUNT
    ).all()
    assert (
        access_order[
            0:TILE_HEIGHT,
            TILE_WIDTH * TILE_GROUP_WIDTH : TILE_WIDTH * TILE_GROUP_WIDTH * 2,
        ]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tile_repeat9
@construct_test
def tile_repeat9():
    # Tile repeat and tile group height and tile col major
    TENSOR_HEIGHT = 16
    TENSOR_WIDTH = 16
    TILE_HEIGHT = 4
    TILE_WIDTH = 4
    REPEAT_COUNT = 3
    TILE_GROUP_HEIGHT = 2
    tiler = TensorTiler2D(
        TENSOR_HEIGHT, TENSOR_WIDTH, TILE_HEIGHT, TILE_WIDTH, tile_col_major=True
    )
    tiles = list(
        tiler.tile_iter(tile_group_height=TILE_GROUP_HEIGHT, tile_repeat=REPEAT_COUNT)
    )
    tile = tiles[0]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[0 : TILE_HEIGHT * TILE_GROUP_HEIGHT, 0:TILE_WIDTH] == REPEAT_COUNT
    ).all()

    reference_order = np.array(
        [
            # fmt: off
            [  0,   4,   8,  12], 
            [  1,   5,   9,  13], 
            [  2,   6,  10,  14],
            [  3,   7,  11,  15],
            [ 16,  20,  24,  28],
            [ 17,  21,  25,  29],
            [ 18,  22,  26,  30],
            [ 19,  23,  27,  31],
            # fmt: on
        ],
        dtype=np.int32,
    )
    reference_order = reference_order + (
        TILE_HEIGHT * TILE_GROUP_HEIGHT * TILE_WIDTH * (REPEAT_COUNT - 1)
    )
    assert (
        access_order[0 : TILE_HEIGHT * TILE_GROUP_HEIGHT, 0:TILE_WIDTH]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    tile = tiles[1]
    access_order, access_count = tile.access_tensors()
    assert (
        access_count[0 : TILE_HEIGHT * TILE_GROUP_HEIGHT, TILE_WIDTH : TILE_WIDTH * 2]
        == REPEAT_COUNT
    ).all()
    assert (
        access_order[0 : TILE_HEIGHT * TILE_GROUP_HEIGHT, TILE_WIDTH : TILE_WIDTH * 2]
        == reference_order
    ).all(), f"{reference_order} {access_order}"

    # CHECK: Pass!
    print("Pass!")
