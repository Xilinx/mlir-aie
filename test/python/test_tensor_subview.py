# test_tensor_subview.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

import numpy as np
import pytest

from aie.utils.hostruntime.tensor_class import CPUOnlyTensor


def test_subview_shares_storage():
    parent = CPUOnlyTensor((4, 8), dtype=np.float32)  # 32 elements
    view = parent.subview(8, (8,))  # elements [8:16] == row 1
    assert view.shape == (8,)
    view.data[:] = 3.0
    # The view aliases the parent's storage, so the write lands in the parent,
    # and sibling regions are untouched.
    assert np.allclose(parent.data[1], 3.0)
    assert np.allclose(parent.data[0], 0.0)


def test_subview_dtype_reinterpret():
    parent = CPUOnlyTensor((16,), dtype=np.uint8)  # 16 bytes
    view = parent.subview(0, (4,), dtype=np.uint32)  # same 16 bytes, as 4 u32
    assert view.shape == (4,)
    assert view.nbytes == parent.nbytes


def test_subview_bounds_checked():
    parent = CPUOnlyTensor((4,), dtype=np.float32)
    with pytest.raises(ValueError):
        parent.subview(2, (4,))  # 2 + 4 == 6 elements > 4
