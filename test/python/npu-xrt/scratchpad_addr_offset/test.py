# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for DMA address offset patching via offset_parameter (IRON flow).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: aiecc.py -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir
# RUN: cp aie.mlir.prj/params.txt .
# RUN: %run_on_npu2% %pytest %s
#
# Setup:
#   - Input buffer: 32 i32 values [0, 1, 2, ..., 31]
#   - Core: passthrough of 8 elements
#   - offset_parameter @input_offset controls the DMA read start position
#
# We run three times with different offsets and verify the output each time.

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N_INPUT = 32
N_OUTPUT = 8


@pytest.fixture(scope="module")
def kernel_setup():
    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    # Input buffer: [0, 1, 2, ..., 31] as i32
    in_tensor = iron.arange(N_INPUT, dtype=np.int32, device="cpu")
    out_tensor = iron.tensor((N_OUTPUT,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_tensor.buffer_object())

    params = ParameterScratchpad(run, "params.txt")
    return run, params, in_tensor, out_tensor


@pytest.mark.parametrize(
    "offset,expected",
    [
        (0, list(range(0, 8))),
        (8, list(range(8, 16))),
        (16, list(range(16, 24))),
    ],
)
def test_input_offset(kernel_setup, offset, expected):
    run, params, in_tensor, out_tensor = kernel_setup

    # Clear output
    out_tensor.data.fill(0)
    out_tensor.to("npu")
    in_tensor.to("npu")

    # Write offset parameter (in elements)
    params.write("input_offset", np.int32(offset))
    params.sync()

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    result = out_tensor.numpy().tolist()
    assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
