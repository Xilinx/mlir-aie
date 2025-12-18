# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import aie.iron as iron
from aie.iron.device import NPU2, NPU1


def test_set_current_device():
    iron.set_current_device(NPU1)
    current_device = iron.get_current_device()
    assert isinstance(current_device, NPU1)

    old_device = iron.set_current_device(NPU2)
    current_device = iron.get_current_device()
    assert isinstance(current_device, NPU2)
    assert old_device == NPU1
