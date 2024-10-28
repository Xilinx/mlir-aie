import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: tensor_tile_same_height
@construct_test
def tensor_tile_same_height():
    tiler = TensorTiler2D(12, 8, 12, 2)
    access_order = tiler.access_order()
    reference_access = np.array(
        # fmt: off
        [
            [ 0,  1, 24, 25, 48, 49, 72, 73],
            [ 2,  3, 26, 27, 50, 51, 74, 75],
            [ 4,  5, 28, 29, 52, 53, 76, 77],
            [ 6,  7, 30, 31, 54, 55, 78, 79],
            [ 8,  9, 32, 33, 56, 57, 80, 81],
            [10, 11, 34, 35, 58, 59, 82, 83],
            [12, 13, 36, 37, 60, 61, 84, 85],
            [14, 15, 38, 39, 62, 63, 86, 87],
            [16, 17, 40, 41, 64, 65, 88, 89],
            [18, 19, 42, 43, 66, 67, 90, 91],
            [20, 21, 44, 45, 68, 69, 92, 93],
            [22, 23, 46, 47, 70, 71, 94, 95]
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )
    assert (reference_access == access_order).all()

    tile1_reference_order = np.array(
        # fmt: off
        [
            [-1, -1, -1, -1,  0,  1, -1, -1],
            [-1, -1, -1, -1,  2,  3, -1, -1],
            [-1, -1, -1, -1,  4,  5, -1, -1],
            [-1, -1, -1, -1,  6,  7, -1, -1],
            [-1, -1, -1, -1,  8,  9, -1, -1],
            [-1, -1, -1, -1, 10, 11, -1, -1],
            [-1, -1, -1, -1, 12, 13, -1, -1],
            [-1, -1, -1, -1, 14, 15, -1, -1],
            [-1, -1, -1, -1, 16, 17, -1, -1],
            [-1, -1, -1, -1, 18, 19, -1, -1],
            [-1, -1, -1, -1, 20, 21, -1, -1],
            [-1, -1, -1, -1, 22, 23, -1, -1]
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
    assert tile_count == (12 // 12) * (8 // 2)

    # CHECK: Pass!
    print("Pass!")
