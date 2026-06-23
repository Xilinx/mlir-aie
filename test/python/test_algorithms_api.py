# test_algorithms_api.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Unit tests for public IRON algorithm package exports."""

import numpy as np

from aie.iron.device import NPU1Col1
from aie.utils.hostruntime import set_current_device


def test_transform_export_is_callable_and_returns_module():
    from aie.iron.algorithms import transform

    assert callable(transform)

    set_current_device(NPU1Col1())
    try:
        tensor_ty = np.ndarray[(1024,), np.dtype[np.int32]]
        module = transform(lambda x: x + 1, tensor_ty, tile_size=16)
        assert module is not None
        assert hasattr(module, "operation")
    finally:
        set_current_device(None)
