# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
from aie.iron.device import NPU1, NPU2
import aie.utils as utils


def test_device_override():
    # Save current device if any (likely from runtime)
    original_device = utils.get_current_device()

    # Create a dummy device
    if isinstance(original_device, NPU1):
        dummy_device = NPU2()
    else:
        dummy_device = NPU1()

    # Set override
    utils.set_current_device(dummy_device)

    # Check if override works
    assert utils.get_current_device() == dummy_device

    # Reset override
    utils.set_current_device(None)

    # Check if reset works (should return original device)
    # If runtime is present, it returns runtime device.
    # If not, it returns None.
    # In both cases, it should match original_device (assuming runtime didn't change/disappear)
    assert type(utils.get_current_device()) == type(original_device)
