# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Static-only Peano compilation of the generated fused wrapper and source.

Run alongside test_kernels_compile.py with ``-m extensive``; excluded from lit
because the shared CASES smoke test already compiles this source per PR.
"""

import os

import pytest
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels.fused import fused_mm
from aie.utils.compile import compile_cxx_core_function
from aie.utils.hostruntime import set_current_device

_TARGETS = {
    "npu1": ("aie2", NPU1Col1),
    "npu2": ("aie2p", NPU2Col1),
}
_SELECTED = os.environ.get("KERNEL_TEST_DEVICE")
_COMPILE_TARGETS = [_TARGETS[_SELECTED]] if _SELECTED else list(_TARGETS.values())


@pytest.mark.extensive
@pytest.mark.parametrize("arch,device", _COMPILE_TARGETS)
@pytest.mark.parametrize("epilogue", ["none", "gelu", "silu", "sigmoid"])
def test_fused_source_compiles(arch, device, epilogue, tmp_path):
    set_current_device(device())
    try:
        fn = fused_mm(dim_k=48, epilogue=epilogue, clamp=(-0.125, 0.75))
        source = tmp_path / "fused.cc"
        obj = tmp_path / "fused.o"
        source.write_text(fn.source_string)
        compile_cxx_core_function(
            source_path=str(source),
            target_arch=arch,
            output_path=str(obj),
            include_dirs=fn.include_dirs,
            compile_args=fn.compile_flags,
        )
        assert obj.stat().st_size > 0
    finally:
        set_current_device(None)
