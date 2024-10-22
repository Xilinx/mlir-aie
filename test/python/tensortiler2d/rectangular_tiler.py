import numpy as np

from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: rectangular_tiler
@construct_test
def rectangular_tiler():
    tiler = TensorTiler2D(16, 8, 4, 4)
    access_order = tiler.access_order()
    reference_access = np.array(
        #fmt: off
        [
            [  0,   1,   2,   3,  16,  17,  18,  19],
            [  4,   5,   6,   7,  20,  21,  22,  23],
            [  8,   9,  10,  11,  24,  25,  26,  27],
            [ 12,  13,  14,  15,  28,  29,  30,  31],
            [ 32,  33,  34,  35,  48,  49,  50,  51],
            [ 36,  37,  38,  39,  52,  53,  54,  55],
            [ 40,  41,  42,  43,  56,  57,  58,  59],
            [ 44,  45,  46,  47,  60,  61,  62,  63],
            [ 64,  65,  66,  67,  80,  81,  82,  83],
            [ 68,  69,  70,  71,  84,  85,  86,  87],
            [ 72,  73,  74,  75,  88,  89,  90,  91],
            [ 76,  77,  78,  79,  92,  93,  94,  95],
            [ 96,  97,  98,  99, 112, 113, 114, 115],
            [100, 101, 102, 103, 116, 117, 118, 119],
            [104, 105, 106, 107, 120, 121, 122, 123],
            [108, 109, 110, 111, 124, 125, 126, 127],
        ],
        #fmt: on
        dtype=TensorTiler2D.DTYPE,
    )
    assert (reference_access == access_order).all()

    tile1_reference_order = np.array(
        #fmt: off
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
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [ 0,  1,  2,  3, -1, -1, -1, -1],
            [ 4,  5,  6,  7, -1, -1, -1, -1],
            [ 8,  9, 10, 11, -1, -1, -1, -1],
            [12, 13, 14, 15, -1, -1, -1, -1],
        ],
        #fmt: on
        dtype=TensorTiler2D.DTYPE,
    )

    tile_count = 0
    for t in tiler.tile_iter():
        if tile_count == 6:
            tile_access_order = t.access_order()
            assert (tile_access_order == tile1_reference_order).all()
        tile_count += 1
    assert tile_count == (16 // 4) * (8 // 4)

    # CHECK: Pass!
    print("Pass!")
