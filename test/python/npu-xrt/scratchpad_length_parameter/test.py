# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for a transfer length set at runtime via length_parameter (IRON flow).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

# Each run checks the tiles moved (see aie_design.py) and that
# nothing past them in the -1-filled output was written.

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 256
TILE = 8


@pytest.fixture(scope="module")
def kernel_setup():
    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    in_tensor = iron.arange(N, dtype=np.int32, device="cpu")
    out_tensor = iron.tensor((N,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, in_tensor.buffer_object())
    run.set_arg(1, out_tensor.buffer_object())

    params = ParameterScratchpad(run, "params.txt")
    return run, params, in_tensor, out_tensor


@pytest.mark.parametrize("tiles", [0, 3, 1, 32])
def test_length_parameter(kernel_setup, tiles):
    run, params, in_tensor, out_tensor = kernel_setup

    out_tensor.data.fill(-1)
    out_tensor.to("npu")
    in_tensor.to("npu")

    params.write("tiles", np.int32(tiles))
    params.sync()

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    moved = tiles * TILE
    expected = np.full(N, -1, dtype=np.int32)
    expected[:moved] = np.arange(moved, dtype=np.int32) + 1
    np.testing.assert_array_equal(out_tensor.numpy(), expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
