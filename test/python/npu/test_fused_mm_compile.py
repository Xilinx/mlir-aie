# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Static-only Peano compilation of the fused tile source.

Run alongside test_kernels_compile.py with ``-m extensive``; excluded from lit
because the shared CASES smoke test already compiles this source per PR.
"""

import os
import subprocess

import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernels.fused import fused_mm
from aie.utils import config
from aie.utils.compile import compile_cxx_core_function
from aie.utils.hostruntime import set_current_device

_TARGETS = {
    "npu1": ("aie2", NPU1Col1),
    "npu2": ("aie2p", NPU2Col1),
}
_SELECTED = os.environ.get("KERNEL_TEST_DEVICE")
_COMPILE_TARGETS = [_TARGETS[_SELECTED]] if _SELECTED else list(_TARGETS.values())


def _check_lut_linkage(obj, arch):
    defined = subprocess.check_output(
        [config.nm_path(), "--defined-only", str(obj)], text=True
    )
    undefined = subprocess.check_output(
        [config.nm_path(), "--undefined-only", str(obj)], text=True
    )
    assert ("exp_ilut_ab" in defined) == (arch == "aie2")
    if arch == "aie2":
        assert "exp_ilut_ab" not in undefined
        assert "exp_ilut_cd" not in undefined


@pytest.mark.extensive
@pytest.mark.parametrize("arch,device", _COMPILE_TARGETS)
@pytest.mark.parametrize("epilogue", ["none", "gelu", "silu", "sigmoid"])
def test_fused_source_compiles(arch, device, epilogue, tmp_path):
    set_current_device(device())
    try:
        fn = fused_mm(dim_k=48, epilogue=epilogue, clamp=(-0.125, 0.75))
        obj = tmp_path / "fused.o"
        assert fn.source_string is None
        compile_cxx_core_function(
            source_path=fn.source_file,
            target_arch=arch,
            output_path=str(obj),
            include_dirs=fn.include_dirs,
            compile_args=fn.compile_flags,
        )
        assert obj.stat().st_size > 0
        _check_lut_linkage(obj, arch)
    finally:
        set_current_device(None)


@pytest.mark.extensive
@pytest.mark.parametrize("arch,device", _COMPILE_TARGETS)
@pytest.mark.parametrize(
    "factory",
    ["softmax", "gelu", "silu", "swiglu", "bf16_exp", "tanh", "sigmoid", "leaky_relu"],
)
def test_activation_source_compiles_with_target_lut_linkage(
    arch, device, factory, tmp_path
):
    set_current_device(device())
    try:
        fn = getattr(kernels, factory)()
        assert fn.source_string is None
        obj = tmp_path / "activation.o"
        compile_cxx_core_function(
            source_path=fn.source_file,
            target_arch=arch,
            output_path=str(obj),
            include_dirs=fn.include_dirs,
            compile_args=fn.compile_flags,
        )
        assert obj.stat().st_size > 0
        _check_lut_linkage(obj, arch)
    finally:
        set_current_device(None)
