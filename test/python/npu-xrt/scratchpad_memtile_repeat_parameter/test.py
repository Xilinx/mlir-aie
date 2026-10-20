# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for a mem tile channel's repeat count set at runtime via repeat_parameter
# (IRON flow).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

# Each run checks the tiles moved (see aie_design.py) and that nothing past
# them in the -1-filled output was written.

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 1024
TILE = 64


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


@pytest.mark.parametrize("reps", [2, 0, 6, N // TILE - 1, 2])
def test_repeat_parameter(kernel_setup, reps):
    run, params, in_tensor, out_tensor = kernel_setup

    out_tensor.data.fill(-1)
    out_tensor.to("npu")
    in_tensor.to("npu")

    params.write("reps", np.int32(reps))
    params.write("tiles", np.int32(reps + 1))
    params.sync()

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    moved = (reps + 1) * TILE
    expected = np.full(N, -1, dtype=np.int32)
    expected[:moved] = np.arange(moved, dtype=np.int32)
    np.testing.assert_array_equal(out_tensor.numpy(), expected)


@pytest.mark.parametrize("reps", [N // TILE, -1])
def test_repeat_parameter_out_of_range(kernel_setup, reps):
    _, params, _, _ = kernel_setup
    params.write("reps", np.int32(1))
    slot = params.read("reps")
    with pytest.raises(ValueError, match="outside \\[0, 15\\]"):
        params.write("reps", np.int32(reps))
    assert params.read("reps") == slot == 1 << 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
