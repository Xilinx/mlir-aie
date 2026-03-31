# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
from aie.iron.device import NPU1Col1, NPU1Col2, NPU1, NPU2


@pytest.fixture(params=[NPU1Col1, NPU1Col2, NPU1, NPU2])
def device(request):
    return request.param()


def test_cols(device):
    assert device.cols == device._tm.columns()


def test_rows(device):
    assert device.rows == device._tm.rows()
