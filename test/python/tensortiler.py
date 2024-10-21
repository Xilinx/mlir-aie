import numpy as np

from aie.helpers.tensortiler import TensorTiler2D
from util import construct_test


# CHECK-LABEL: tensor_tiler_row_major
@construct_test
def tensor_tiler_simple():
    TENSOR_HEIGHT = 2
    TENSOR_WIDTH = 3

    tiler = TensorTiler2D(TENSOR_HEIGHT, TENSOR_WIDTH, TENSOR_HEIGHT, TENSOR_WIDTH)
    access_map = tiler.access_order_map()

    expected = np.ndarray([[0, 1, 2], [3, 4, 5]])
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

    expected_sizes = [0, 0, TENSOR_HEIGHT, TENSOR_WIDTH]
    assert (
        t.sizes == expected_sizes
    ), f"Expected sizes {expected_sizes} but got {t.sizes}"
    expected_strides = [1, 1, 1, 1]
    assert (
        t.strides == expected_strides
    ), f"Expected strides {expected_strides} but got {t.strides}"

    t2 = next(iter)
    assert t2 is None, "Should only be one tile in iter"

    # CHECK: Pass!
    print("Pass!")
