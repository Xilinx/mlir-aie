# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

import numpy as np

from aie.helpers.taplib import TensorAccessPattern
from aie.iron.runtime.data import RuntimeData


def test_slice_returns_metadata_without_runtime_storage():
    for dtype in (np.int8, np.int32, np.float32):
        data = RuntimeData(np.ndarray[(4, 8), np.dtype[dtype]])
        tap = data[1::2, 2::3]
        assert isinstance(tap, TensorAccessPattern)
        assert tap.offset == 10
        assert tap.sizes == [2, 2]
        assert tap.strides == [16, 3]

        scalar = data[-1, -1]
        assert isinstance(scalar, TensorAccessPattern)
        assert scalar.offset == 31
        assert scalar.sizes == [1]


test_slice_returns_metadata_without_runtime_storage()
print("RuntimeData slicing returns metadata")
