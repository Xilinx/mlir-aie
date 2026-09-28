# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Python test for scratchpad parameter passing using ParameterScratchpad.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s
#
# This is the Python equivalent of the C++ test in ../scratchpad_params/.
# It exercises the full flow:
#   1. aiecc compiles aie.mlir → aie.elf + params.txt
#   2. This script loads the ELF through the host runtime, takes the handle's
#      ParameterScratchpad from params.txt, writes bf16 parameters, and
#      verifies the core computes foo * bar.
#   3. A second parametrized case with different values tests parameter re-use
#      across runs.

import pytest
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.npukernel import NPUKernel


@pytest.fixture(scope="module")
def kernel_setup():
    runtime = XRTHostRuntime()
    handle = runtime.load(NPUKernel(elf_path="aie.elf", kernel_name="test:sequence"))

    # Output buffer: 2 x bf16 (only the first element is written by the core)
    out_tensor = iron.tensor((2,), dtype=bfloat16, device="cpu")

    # Bound to the handle's one run, so what is written here reaches every
    # later runtime.run() of the handle.
    params = handle.parameter_scratchpad("params.txt")
    return runtime, handle, params, out_tensor


@pytest.mark.parametrize(
    "foo,bar",
    [
        (bfloat16(3.0), bfloat16(4.0)),
        (bfloat16(2.0), bfloat16(5.0)),
    ],
)
def test_scratchpad_param_multiply(kernel_setup, foo, bar):
    runtime, handle, params, out_tensor = kernel_setup

    out_tensor.data.fill(0)

    params.write("foo", foo)
    params.write("bar", bar)
    params.sync()

    runtime.run(handle, [out_tensor])

    out_tensor.to("cpu")
    result = float(out_tensor.numpy()[0])
    expected = float(foo) * float(bar)
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
