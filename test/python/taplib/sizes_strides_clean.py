# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence
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


# CHECK-LABEL: sizes_strides_from_tuples
@construct_test
def sizes_strides_from_tuples():
    sizes_fixup, strides_fixup = validate_and_clean_sizes_strides(
        (1, 3, 1, 4), (1, 4, 4, 1)
    )
    assert sizes_fixup == [1, 3, 1, 4], f"{sizes_fixup}"
    assert strides_fixup == [0, 4, 4, 1], f"{strides_fixup}"

    from_tuples = TensorAccessPattern((3, 4), 0, (3, 4), (4, 1))
    from_lists = TensorAccessPattern([3, 4], 0, [3, 4], [4, 1])
    assert from_tuples == from_lists, f"{from_tuples} != {from_lists}"
    assert from_tuples.sizes.copy() == [3, 4]
    assert from_tuples.strides.copy() == [4, 1]

    tas = TensorAccessSequence.from_taps([from_tuples, from_lists])
    assert len(tas) == 2 and tas[0] == tas[1]
