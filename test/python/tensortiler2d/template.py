import numpy as np

from aie.helpers.tensortiler.tensortiler2D import TensorTiler2D
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: square_tiler
@construct_test
def square_tiler():
    tiler = TensorTiler2D(32, 32, 4, 4)
    access_order = tiler.access_order()
    reference_access = np.array(
        [0],  # TODOD: fill this in
        dtype=TensorTiler2D.DTYPE,
    )
    assert (reference_access == access_order).all()

    tile1_reference_order = np.array(
        [0],  # TODO: fill this in
        dtype=TensorTiler2D.DTYPE,
    )

    tile_count = 0
    for t in tiler.tile_iter():
        if tile_count == 1:
            tile_access_order = t.access_order()
            assert (tile_access_order == tile1_reference_order).all()
        tile_count += 1
    assert tile_count == 64

    # CHECK: Pass!
    print("Pass!")
