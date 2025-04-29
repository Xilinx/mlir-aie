# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import aie.iron as iron
from aie.iron.device import NPU2


def test_set_current_device():
    device = NPU2()
    iron.set_current_device(device)
    current_device = iron.get_current_device()
    assert current_device == device
