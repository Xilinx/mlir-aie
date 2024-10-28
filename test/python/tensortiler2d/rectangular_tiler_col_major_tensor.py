import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: rectangular_tiler_col_major_tensor
@construct_test
def rectangular_tiler_col_major_tensor():
    tiler = TensorTiler2D(8, 16, 4, 2, tensor_col_major=True)
    access_order = tiler.access_order()
    reference_access = np.array(
        # fmt: off
        [
            [  0,   1,  16,  17,  32,  33,  48,  49,  64,  65,  80,  81,  96,  97, 112, 113],
            [  2,   3,  18,  19,  34,  35,  50,  51,  66,  67,  82,  83,  98,  99, 114, 115],
            [  4,   5,  20,  21,  36,  37,  52,  53,  68,  69,  84,  85, 100, 101, 116, 117],
            [  6,   7,  22,  23,  38,  39,  54,  55,  70,  71,  86,  87, 102, 103, 118, 119],
            [  8,   9,  24,  25,  40,  41,  56,  57,  72,  73,  88,  89, 104, 105, 120, 121],
            [ 10,  11,  26,  27,  42,  43,  58,  59,  74,  75,  90,  91, 106, 107, 122, 123],
            [ 12,  13,  28,  29,  44,  45,  60,  61,  76,  77,  92,  93, 108, 109, 124, 125],
            [ 14,  15,  30,  31,  46,  47,  62,  63,  78,  79,  94,  95, 110, 111, 126, 127],
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )
    assert (reference_access == access_order).all()

    tile1_reference_order = np.array(
        # fmt: off
        [
            [-1, -1, -1, -1,  0,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )

    tile_count = 0
    for t in tiler.tile_iter():
        if tile_count == 2:
            tile_access_order = t.access_order()
            assert (tile_access_order == tile1_reference_order).all()
        tile_count += 1
    assert tile_count == (8 // 4) * (16 // 2)

    # CHECK: Pass!
    print("Pass!")
