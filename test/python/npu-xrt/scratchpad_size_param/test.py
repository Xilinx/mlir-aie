# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for the per-run extent of a shim transfer via size_parameters (IRON flow).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s
#
# Setup:
#   - Input buffer: 256 i32 values [0, 1, ..., 255]; tiles of 16, at most 16
#   - @n: tiles the input transfer moves (a DMA-only parameter, on D2)
#   - @m: pairs of tiles the core copies; the output transfer moves m tiles
#     into each half of the output (dimension 2 under an iteration; read by
#     the core, so stored shifted; the firmware multiplier accounts for it)
#   - @off: element offset of the input transfer, on the BD @n sizes
#
# Each run checks that exactly 16 * m values arrive in each half and the rest
# of the output keeps the host's fill.

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 256
TILE = 16
UNTOUCHED = -1


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


# Repeated counts check the patched length is rewritten each run, not
# accumulated; the full 8 per half is the length the BD was built with.
@pytest.mark.parametrize(
    "tiles,offset",
    [(8, 0), (3, 32), (1, 0), (5, 64), (3, 32), (8, 0)],
)
def test_transfer_extent(kernel_setup, tiles, offset):
    run, params, in_tensor, out_tensor = kernel_setup

    out_tensor.data.fill(UNTOUCHED)
    out_tensor.to("npu")
    in_tensor.to("npu")

    params.write("n", np.int32(2 * tiles))
    params.write("m", np.int32(tiles))
    params.write("off", np.int32(offset))
    params.sync()

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    result = out_tensor.numpy()
    moved = tiles * TILE
    expected = np.full(N, UNTOUCHED, dtype=np.int32)
    expected[:moved] = np.arange(offset, offset + moved, dtype=np.int32)
    half = N // 2
    expected[half : half + moved] = np.arange(
        offset + moved, offset + 2 * moved, dtype=np.int32
    )
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
