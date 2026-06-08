# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s

import pytest
from aie.iron.device import NPU1, NPU1Col1, NPU2, NPU2Col1
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


def test_device_consistency():
    # Define a mock runtime class
    class MockRuntime(utils.HostRuntime):
        def load(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

        def device(self):
            return NPU1()

    runtime = MockRuntime()

    # Set compatible override
    utils.set_current_device(NPU1())
    # Should not raise
    runtime.check_device_consistency()

    # A 1-column variant of the same generation is loadable on the full device.
    utils.set_current_device(NPU1Col1())
    runtime.check_device_consistency()

    # Set incompatible override (wrong generation)
    utils.set_current_device(NPU2())
    with pytest.raises(RuntimeError, match="not loadable on runtime device"):
        runtime.check_device_consistency()

    # Wrong generation, narrower variant — still incompatible.
    utils.set_current_device(NPU2Col1())
    with pytest.raises(RuntimeError, match="not loadable on runtime device"):
        runtime.check_device_consistency()

    # Reset
    utils.set_current_device(None)
