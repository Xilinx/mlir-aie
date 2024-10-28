import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: rectangular_tiler2
@construct_test
def rectangular_tiler2():
    tiler = TensorTiler2D(12, 8, 3, 2)
    access_order = tiler.access_order()
    reference_access = np.array(
        # fmt: off
        [
            [ 0,  1,  6,  7, 12, 13, 18, 19],
            [ 2,  3,  8,  9, 14, 15, 20, 21],
            [ 4,  5, 10, 11, 16, 17, 22, 23],
            [24, 25, 30, 31, 36, 37, 42, 43],
            [26, 27, 32, 33, 38, 39, 44, 45],
            [28, 29, 34, 35, 40, 41, 46, 47],
            [48, 49, 54, 55, 60, 61, 66, 67],
            [50, 51, 56, 57, 62, 63, 68, 69],
            [52, 53, 58, 59, 64, 65, 70, 71],
            [72, 73, 78, 79, 84, 85, 90, 91],
            [74, 75, 80, 81, 86, 87, 92, 93],
            [76, 77, 82, 83, 88, 89, 94, 95],
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
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [ 0,  1, -1, -1, -1, -1, -1, -1],
            [ 2,  3, -1, -1, -1, -1, -1, -1],
            [ 4,  5, -1, -1, -1, -1, -1, -1],
        ],
        # fmt: on
        dtype=TensorTiler2D.DTYPE,
    )

    tile_count = 0
    for t in tiler.tile_iter():
        if tile_count == 12:
            tile_access_order = t.access_order()
            assert (tile_access_order == tile1_reference_order).all()
        tile_count += 1
    assert tile_count == (12 // 3) * (8 // 2)

    # CHECK: Pass!
    print("Pass!")
