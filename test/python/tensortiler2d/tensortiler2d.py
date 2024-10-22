import numpy as np

from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensortiler_simple
@construct_test
def tensortiler_simple():
    TENSOR_HEIGHT = 2
    TENSOR_WIDTH = 3

    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH)
    access_map = tiler.access_order()

    expected = np.array([[0, 1, 2], [3, 4, 5]])
    assert expected.shape == access_map.shape
    assert (expected == access_map).all()

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
    expected_strides = [1, 3, 3, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected == t.access_order()
    ).all(), f"Expected {expected} but got {t.access_order()}"

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

    expected_tile = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()

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
    expected_strides = [1, 36, 1, 12]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"

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

    expected_tile = np.array([[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()

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
    expected_strides = [1, 36, 1, 12]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"

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

    expected_tile = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    assert (TENSOR_HEIGHT, TENSOR_WIDTH) == access_map.shape
    assert (expected_tile == access_map[0:TILE_HEIGHT, 0:TILE_WIDTH]).all()

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
    expected_strides = [1, 4, 12, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile == t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, 0:TILE_WIDTH]}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_iter_chunk_row_major
@construct_test
def tensortiler_tensor_iter_chunk_row_major():
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

    CHUNK_HEIGHT = 2
    CHUNK_WIDTH = 2
    iter = tiler.tile_iter(chunk_height=2, chunk_width=2)
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // (TILE_HEIGHT * CHUNK_HEIGHT)) * (
        TENSOR_WIDTH // (TILE_WIDTH * CHUNK_WIDTH)
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

    expected_sizes = [CHUNK_HEIGHT, CHUNK_WIDTH, TILE_HEIGHT, TILE_WIDTH]
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
        t.offset == CHUNK_WIDTH * TILE_WIDTH
    ), f"Expected offset {CHUNK_WIDTH * TILE_WIDTH} but got {t.offset}"

    expected_sizes = [CHUNK_HEIGHT, CHUNK_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [36, 2, 12, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile
        == t.access_order()[
            0:TILE_HEIGHT, CHUNK_WIDTH * TILE_WIDTH : CHUNK_WIDTH * (TILE_WIDTH + 1)
        ]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[0:TILE_HEIGHT, CHUNK_WIDTH*TILE_WIDTH:CHUNK_WIDTH*(TILE_WIDTH+1)]}"

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: tensortiler_tensor_iter_chunk_col_major
@construct_test
def tensortiler_tensor_iter_chunk_col_major():
    TENSOR_HEIGHT = 12
    TENSOR_WIDTH = 8
    TILE_HEIGHT = 3
    TILE_WIDTH = 2

    expected_tile = np.array([[0, 1], [2, 3], [4, 5]])

    tiler = TensorTiler2D(
        TENSOR_HEIGHT,
        TENSOR_WIDTH,
        TILE_HEIGHT,
        TILE_WIDTH,
    )

    CHUNK_HEIGHT = 2
    CHUNK_WIDTH = 2
    iter = tiler.tile_iter(chunk_height=2, chunk_width=2, col_major=True)
    tiles = list(iter)
    expected_num_tiles = (TENSOR_HEIGHT // (TILE_HEIGHT * CHUNK_HEIGHT)) * (
        TENSOR_WIDTH // (TILE_WIDTH * CHUNK_WIDTH)
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

    expected_sizes = [CHUNK_HEIGHT, CHUNK_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [24, 2, 8, 1]
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
    assert t.offset == 48, f"Expected offset 48 but got {t.offset}"

    expected_sizes = [CHUNK_HEIGHT, CHUNK_WIDTH, TILE_HEIGHT, TILE_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [24, 2, 8, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    assert (
        expected_tile
        == t.access_order()[
            CHUNK_HEIGHT * TILE_HEIGHT : (CHUNK_HEIGHT + 1) * TILE_HEIGHT, 0:TILE_WIDTH
        ]
    ).all(), f"Expected {expected_tile} but got {t.access_order()[CHUNK_HEIGHT*TILE_HEIGHT:(CHUNK_HEIGHT+1)*TILE_HEIGHT, 0:TILE_WIDTH]}"

    # CHECK: Pass!
    print("Pass!")
