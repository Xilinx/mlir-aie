import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: rectangular_tiler_col_major_tensor_and_tile
@construct_test
def rectangular_tiler_col_major_tensor_and_tile():
    tiler = TensorTiler2D(16, 8, 4, 2, tensor_col_major=True, tile_col_major=True)
    access_order = tiler.access_order()
    reference_access = np.array(
        # fmt: off
        [
            [  0,   4,  32,  36,  64,  68,  96, 100],
            [  1,   5,  33,  37,  65,  69,  97, 101],
            [  2,   6,  34,  38,  66,  70,  98, 102],
            [  3,   7,  35,  39,  67,  71,  99, 103],
            [  8,  12,  40,  44,  72,  76, 104, 108],
            [  9,  13,  41,  45,  73,  77, 105, 109],
            [ 10,  14,  42,  46,  74,  78, 106, 110],
            [ 11,  15,  43,  47,  75,  79, 107, 111],
            [ 16,  20,  48,  52,  80,  84, 112, 116],
            [ 17,  21,  49,  53,  81,  85, 113, 117],
            [ 18,  22,  50,  54,  82,  86, 114, 118],
            [ 19,  23,  51,  55,  83,  87, 115, 119],
            [ 24,  28,  56,  60,  88,  92, 120, 124],
            [ 25,  29,  57,  61,  89,  93, 121, 125],
            [ 26,  30,  58,  62,  90,  94, 122, 126],
            [ 27,  31,  59,  63,  91,  95, 123, 127]
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )
    assert (reference_access == access_order).all()

    tile1_reference_order = np.array(
        # fmt: off
        [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [ 0,  4, -1, -1, -1, -1, -1, -1],
            [ 1,  5, -1, -1, -1, -1, -1, -1],
            [ 2,  6, -1, -1, -1, -1, -1, -1],
            [ 3,  7, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1]
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )

    tile_count = 0
    for t in tiler.tile_iter():
        if tile_count == 8:
            tile_access_order = t.access_order()
            assert (tile_access_order == tile1_reference_order).all()
        tile_count += 1
    assert tile_count == (16 // 4) * (8 // 2)

    # CHECK: Pass!
    print("Pass!")
