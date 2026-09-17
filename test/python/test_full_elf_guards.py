# test_full_elf_guards.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit test for the full-ELF (reconfiguration) runtime guard -- no NPU.

HRX has no full-ELF / load_pdi path, so a full-ELF kernel (elf_path set, xclbin
None) must be rejected with a clear error instead of crashing on ``Path(None)``.
Reconfiguration full-ELFs run on the XRT runtime; dispatching an ordered sequence
of their entrypoints is done by the application via ``pyxrt.runlist``, not a
library API.
"""

from types import SimpleNamespace

import pytest

from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.hostruntime.hrxruntime.hostruntime import HRXHostRuntime


def _hrx_no_device():
    rt = object.__new__(HRXHostRuntime)  # bypass __init__ (no device needed)
    rt.check_device_consistency = lambda: None
    return rt


def test_hrx_rejects_full_elf_kernel_clearly():
    """A full-ELF kernel (elf_path set, xclbin None) must raise a clear error,
    not an opaque ``Path(None)`` TypeError."""
    rt = _hrx_no_device()
    npu_kernel = SimpleNamespace(
        elf_path="/tmp/overlay.elf",
        xclbin_path=None,
        insts_path=None,
        kernel_name="main:config_1",
    )
    with pytest.raises(HostRuntimeError, match="full-ELF"):
        rt._resolve_kernel(npu_kernel)
