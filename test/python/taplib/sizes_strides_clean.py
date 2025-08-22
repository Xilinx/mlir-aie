from aie.helpers.taplib.utils import validate_and_clean_sizes_strides
from util import construct_test

# RUN: %python %s | FileCheck %s


# CHECK-LABEL: sizes_strides_clean
@construct_test
def sizes_strides_clean():
    sizes = [1, 1, 1, 1]
    strides = [1, 1, 1, 1]

    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(sizes, strides)
    assert sizes_fixup == [1, 1, 1, 1] and sizes_fixup == sizes
    assert (
        strides_fixup == [0, 0, 0, 1] and strides_fixup != strides
    ), f"{strides_fixup}"

    sizes = [1, 3, 1, 1]
    strides = [0, 1, 1, 1]
    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(sizes, strides)
    assert sizes_fixup == [1, 3, 1, 1] and sizes_fixup == sizes
    assert strides_fixup == [0, 1, 1, 1] and strides_fixup == strides

    sizes = [1, 3, 1, 1]
    strides = [1, 1, 1, 1]
    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(sizes, strides)
    assert sizes_fixup == [1, 3, 1, 1] and sizes_fixup == sizes
    assert strides_fixup == [0, 1, 1, 1] and strides_fixup != strides

    sizes = [1, 1, 1, 2]
    strides = [1, 1, 1, 1]
    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(sizes, strides)
    assert sizes_fixup == [1, 1, 1, 2] and sizes_fixup == sizes
    assert strides_fixup == [0, 0, 0, 1] and strides_fixup != strides

    sizes = [2, 1, 1, 2]
    strides = [1, 1, 1, 1]
    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(sizes, strides)
    assert sizes_fixup == [2, 1, 1, 2] and sizes_fixup == sizes
    assert strides_fixup == [1, 1, 1, 1] and strides_fixup == strides
