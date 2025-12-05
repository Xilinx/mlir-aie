# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
from pathlib import Path
from aie.iron.hostruntime.xrtruntime.hostruntime import XRTKernelHandle


def test_kernel_handle_equality():
    h1 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"))
    h2 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"))

    assert h1 == h2
    assert hash(h1) == hash(h2)

    d = {}
    d[h1] = 1
    assert h2 in d
