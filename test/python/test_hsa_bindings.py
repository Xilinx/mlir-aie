# test/python/test_hsa_bindings.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import ctypes
import pytest

from aie.utils.hostruntime.hsaruntime import discovery
from aie.utils.hostruntime.hsaruntime import _bindings
from aie.utils.hostruntime import hsaruntime


def test_packet_struct_size_is_64_bytes():
    # The AIE dispatch packet is a 64-byte AQL packet.
    assert ctypes.sizeof(_bindings.HsaAieKernelDispatchPacket) == 64


def test_context_requires_libhsa():
    if not discovery.hsa_available():
        pytest.skip("libhsa-runtime64.so not discoverable on this host")
    # If HSA is present but there is no AIE agent, construction raises HSAError.
    # Either a valid context or a clear HSAError is acceptable here.
    try:
        ctx = hsaruntime.HSAContext.get()
    except hsaruntime.HSAError:
        return
    assert ctx.aie_agent != 0
    assert ctx.device_gen in ("npu1", "npu2")
