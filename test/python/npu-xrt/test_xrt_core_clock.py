# test_xrt_core_clock.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings, system-linux

"""On-device test that XRTHostRuntime reads the AIE core clock from amdxdna."""

from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime


def test_the_core_clock_is_read_from_the_driver():
    runtime = XRTHostRuntime()
    mhz = runtime.core_mhz
    assert isinstance(mhz, int)
    # Every NPU part clocks its cores between a few hundred MHz and ~2 GHz.
    assert 200 <= mhz <= 3000
    assert runtime.core_mhz == mhz
