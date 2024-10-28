import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensortiler_simple
@construct_test
def tensortiler_simple():
    TENSOR_HEIGHT = 2
    TENSOR_WIDTH = 3

    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH)
    access_map = tiler.access_order()
    count_map = tiler.access_count()

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert expected.shape == access_map.shape
    assert (expected == access_map).all()
    assert (count_map == 1).all()

    iter = tiler.tile_iter()
    t = next(iter)

    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [1, 1, TENSOR_HEIGHT, TENSOR_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [0, 0, 3, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected == t.access_order()
    ).all(), f"Expected {expected} but got {t.access_order()}"
    assert (t.access_count() == 1).all()

    try:
        next(iter)
        assert False, "Iterator should only have one step"
    except StopIteration:
        pass

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_row_major_tile_col_major
@construct_test
def tensortiler_tensor_row_major_tile_col_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 12
    TILE_HEIGHT = 3
    TILE_WIDTH = 4

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
        tensor_col_major=False,
        tile_col_major=True,
    )
    access_map = tiler.access_order()
    count_map = tiler.access_count()

    expected_tile = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()
    assert (count_map == 1).all()

    expected_tile = expected_tile
    expected_tile2 = np.array([[12, 15, 18, 21], [13, 16, 19, 22], [14, 17, 20, 23]])
    assert (
        expected_tile2 == access_map[0:TILE_HEIGHT, TILE_WIDTH : 2 * TILE_WIDTH]
    ).all()

    iter = tiler.tile_iter()
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // TILE_HEIGHT) * (TENSOR_WIDTH // TILE_WIDTH)
    assert (
        len(tiles) == expected_num_tiles
    ), f"Expected {expected_num_tiles} tiles but got {len(tiles)}"

    t = tiles[0]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [1, 1, TILE_WIDTH, TILE_HEIGHT]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [0, 0, 1, 12]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"
    assert (t.access_count()[0:TILE_HEIGHT, 0:TILE_WIDTH] == 1).all()
    assert (t.access_count()[0:TENSOR_HEIGHT, TILE_WIDTH:TENSOR_WIDTH] == 0).all()
    assert (t.access_count()[TILE_HEIGHT:TENSOR_HEIGHT, 0:TILE_WIDTH] == 0).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_col_major_tile_col_major
@construct_test
def tensortiler_tensor_col_major_tile_col_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 12
    TILE_HEIGHT = 3
    TILE_WIDTH = 4

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
        tensor_col_major=True,
        tile_col_major=True,
    )
    access_map = tiler.access_order()
    access_count = tiler.access_count()

    expected_tile = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()
    assert (access_count == 1).all()

    expected_tile = expected_tile
    expected_tile2 = np.array([[12, 15, 18, 21], [13, 16, 19, 22], [14, 17, 20, 23]])
    assert (
        expected_tile2 == access_map[TILE_HEIGHT : 2 * TILE_HEIGHT, 0:TILE_WIDTH]
    ).all()

    iter = tiler.tile_iter()
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // TILE_HEIGHT) * (TENSOR_WIDTH // TILE_WIDTH)
    assert (
        len(tiles) == expected_num_tiles
    ), f"Expected {expected_num_tiles} tiles but got {len(tiles)}"

    t = tiles[0]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [1, 1, TILE_WIDTH, TILE_HEIGHT]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [0, 0, 1, 12]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"
    assert (t.access_count()[0:TILE_HEIGHT, 0:TILE_WIDTH] == 1).all()
    assert (t.access_count()[0:TENSOR_HEIGHT, TILE_WIDTH:TENSOR_WIDTH] == 0).all()
    assert (t.access_count()[TILE_HEIGHT:TENSOR_HEIGHT, 0:TILE_WIDTH] == 0).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_col_major_tile_row_major
@construct_test
def tensortiler_tensor_col_major_tile_row_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 12
    TILE_HEIGHT = 3
    TILE_WIDTH = 4

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
        tensor_col_major=True,
        tile_col_major=False,
    )
    access_map = tiler.access_order()
    access_count = tiler.access_count()

    expected_tile = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()
    assert (access_count == 1).all()

    expected_tile = expected_tile
    expected_tile2 = np.array([[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]])
    assert (
        expected_tile2 == access_map[TILE_HEIGHT : 2 * TILE_HEIGHT, 0:TILE_WIDTH]
    ).all()

    iter = tiler.tile_iter()
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // TILE_HEIGHT) * (TENSOR_WIDTH // TILE_WIDTH)
    assert (
        len(tiles) == expected_num_tiles
    ), f"Expected {expected_num_tiles} tiles but got {len(tiles)}"

    t = tiles[0]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [1, 1, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [0, 0, 12, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"
    assert (t.access_count()[0:TILE_HEIGHT, 0:TILE_WIDTH] == 1).all()
    assert (t.access_count()[0:TENSOR_HEIGHT, TILE_WIDTH:TENSOR_WIDTH] == 0).all()
    assert (t.access_count()[TILE_HEIGHT:TENSOR_HEIGHT, 0:TILE_WIDTH] == 0).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_iter_group_row_major
@construct_test
def tensortiler_tensor_iter_group_row_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 12
    TILE_HEIGHT = 3
    TILE_WIDTH = 2

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
    )

    expected_tile = np.array([[0, 1], [2, 3], [4, 5]])
    expected_group = np.array(
        [
            [0, 1, 6, 7],
            [2, 3, 8, 9],
            [4, 5, 10, 11],
            [12, 13, 18, 19],
            [14, 15, 20, 21],
            [16, 17, 22, 23],
        ]
    )

    GROUP_HEIGHT = 2
    GROUP_WIDTH = 2
    iter = tiler.tile_iter(tile_group_height=GROUP_HEIGHT, tile_group_width=GROUP_WIDTH)
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // (TILE_HEIGHT * GROUP_HEIGHT)) * (
        TENSOR_WIDTH // (TILE_WIDTH * GROUP_WIDTH)
    )
    assert (
        len(tiles) == expected_num_tiles
    ), f"Expected {expected_num_tiles} tiles but got {len(tiles)}"

    t = tiles[0]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [GROUP_HEIGHT, GROUP_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [36, 2, 12, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"

    t = tiles[1]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert (
        t.offset == GROUP_WIDTH * TILE_WIDTH
    ), f"Expected offset {GROUP_WIDTH * TILE_WIDTH} but got {t.offset}"

    expected_sizes = [GROUP_HEIGHT, GROUP_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [36, 2, 12, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_group
        == t.access_order()[
            0 : GROUP_HEIGHT * TILE_HEIGHT,
            GROUP_WIDTH * TILE_WIDTH : 2 * GROUP_WIDTH * TILE_WIDTH,
        ]
    ).all(), f"Expected {expected_group} but got {t.access_order()[0:GROUP_HEIGHT * TILE_HEIGHT, GROUP_WIDTH * TILE_WIDTH : 2 * GROUP_WIDTH * TILE_WIDTH]}"
    assert (
        t.access_count()[
            0 : GROUP_HEIGHT * TILE_HEIGHT,
            GROUP_WIDTH * TILE_WIDTH : 2 * GROUP_WIDTH * TILE_WIDTH,
        ]
        == 1
    ).all()
    assert (t.access_count()[0:TENSOR_HEIGHT, 0 : GROUP_WIDTH * TILE_WIDTH] == 0).all()
    assert (
        t.access_count()[0:TENSOR_HEIGHT, GROUP_WIDTH * TILE_WIDTH * 2 : TENSOR_WIDTH]
        == 0
    ).all()
    assert (
        t.access_count()[
            GROUP_HEIGHT * TILE_HEIGHT : TENSOR_HEIGHT,
            GROUP_WIDTH * TILE_WIDTH : 2 * GROUP_WIDTH * TILE_WIDTH,
        ]
        == 0
    ).all()

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_iter_group_col_major
@construct_test
def tensortiler_tensor_iter_group_col_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 8
    TILE_HEIGHT = 3
    TILE_WIDTH = 2

    expected_tile = np.array([[0, 1], [2, 3], [4, 5]])
    expected_group = np.array(
        [
            [0, 1, 6, 7],
            [2, 3, 8, 9],
            [4, 5, 10, 11],
            [12, 13, 18, 19],
            [14, 15, 20, 21],
            [16, 17, 22, 23],
        ]
    )

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
    )
    access_order = tiler.access_order()
    assert (access_order[0:TILE_HEIGHT, 0:TILE_WIDTH] == expected_tile).all()
    access_count = tiler.access_count()
    assert (access_count == 1).all()

    GROUP_HEIGHT = 2
    GROUP_WIDTH = 2
    iter = tiler.tile_iter(
        tile_group_height=GROUP_HEIGHT, tile_group_width=GROUP_WIDTH, col_major=True
    )
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // (TILE_HEIGHT * GROUP_HEIGHT)) * (
        TENSOR_WIDTH // (TILE_WIDTH * GROUP_WIDTH)
    )
    assert (
        len(tiles) == expected_num_tiles
    ), f"Expected {expected_num_tiles} tiles but got {len(tiles)}"

    t = tiles[0]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 0, f"Expected offset 0 but got {t.offset}"

    expected_sizes = [GROUP_HEIGHT, GROUP_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [24, 2, 8, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_group
        == t.access_order()[
            0 : TILE_HEIGHT * GROUP_HEIGHT, 0 : TILE_WIDTH * GROUP_WIDTH
        ]
    ).all(), f"Expected {expected_group} but got {t.access_order()[0:TILE_HEIGHT*GROUP_HEIGHT, 0:TILE_WIDTH*GROUP_WIDTH]}"
    assert (
        t.access_count()[0 : TILE_HEIGHT * GROUP_HEIGHT, 0 : TILE_WIDTH * GROUP_WIDTH]
        == 1
    ).all()
    assert (
        t.access_count()[0:TENSOR_HEIGHT, TILE_WIDTH * GROUP_WIDTH : TENSOR_WIDTH] == 0
    ).all()
    assert (
        t.access_count()[TILE_HEIGHT * GROUP_HEIGHT : TENSOR_HEIGHT, 0:TENSOR_WIDTH]
        == 0
    ).all()

    t = tiles[1]
    assert (
        t.tensor_height == TENSOR_HEIGHT
    ), f"Expected tensor height {TENSOR_HEIGHT} but got {t.tensor_height}"
    assert (
        t.tensor_width == TENSOR_WIDTH
    ), f"Expected tensor width {TENSOR_WIDTH} but got {t.tensor_width}"
    assert t.offset == 48, f"Expected offset 48 but got {t.offset}"

    expected_sizes = [GROUP_HEIGHT, GROUP_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [24, 2, 8, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_group
        == t.access_order()[
            GROUP_HEIGHT * TILE_HEIGHT : 2 * GROUP_HEIGHT * TILE_HEIGHT,
            0 : TILE_WIDTH * GROUP_WIDTH,
        ]
    ).all(), f"Expected {expected_group} but got {t.access_order()[GROUP_HEIGHT * TILE_HEIGHT : 2 * GROUP_HEIGHT * TILE_HEIGHT, 0:TILE_WIDTH * GROUP_WIDTH]}"
    assert (
        t.access_count()[
            GROUP_HEIGHT * TILE_HEIGHT : 2 * GROUP_HEIGHT * TILE_HEIGHT,
            0 : TILE_WIDTH * GROUP_WIDTH,
        ]
        == 1
    ).all()
    # TODO: Could check other regions are 0

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: access_order_from_sizes_strides
@construct_test
def access_order_from_sizes_strides():
    access_order, _ = TensorTiler2D.get_access_tensors(
        8, 16, [1, 8, 8, 2], [0, 2, 16, 1]
    )
    # fmt: off
    reference_order = [
        [  0,   1,  16,  17,  32,  33,  48,  49,  64,  65,  80,  81,  96,  97, 112, 113],
        [  2,   3,  18,  19,  34,  35,  50,  51,  66,  67,  82,  83,  98,  99, 114, 115],
        [  4,   5,  20,  21,  36,  37,  52,  53,  68,  69,  84,  85, 100, 101, 116, 117],
        [  6,   7,  22,  23,  38,  39,  54,  55,  70,  71,  86,  87, 102, 103, 118, 119],
        [  8,   9,  24,  25,  40,  41,  56,  57,  72,  73,  88,  89, 104, 105, 120, 121],
        [ 10,  11,  26,  27,  42,  43,  58,  59,  74,  75,  90,  91, 106, 107, 122, 123],
        [ 12,  13,  28,  29,  44,  45,  60,  61,  76,  77,  92,  93, 108, 109, 124, 125],
        [ 14,  15,  30,  31,  46,  47,  62,  63,  78,  79,  94,  95, 110, 111, 126, 127]
    ]
    # fmt: on
    np.equal(access_order, reference_order)

    _, access_count = TensorTiler2D.get_access_tensors(
        8, 16, [1, 8, 8, 2], [0, 2, 16, 1]
    )
    assert (access_count == 1).all()
