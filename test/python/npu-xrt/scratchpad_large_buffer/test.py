# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# A runtime offset reaches every word of a 2.125 GiB int32 host buffer: past
# 2^28 words (1 GiB, the most an int8-typed buffer's offset could address) and
# the last TILE words.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

WORDS = 2_281_701_376 // 4
TILE = 16
OFFSETS = (0, (1 << 28) + 5 * TILE, WORDS - TILE)


@pytest.fixture(scope="module")
def setup():
    device = XRTHostRuntime()._device
    context = pyxrt.hw_context(device, pyxrt.elf("aie.elf"))
    kernel = pyxrt.ext.kernel(context, "test:sequence")
    big = iron.tensor((WORDS,), dtype=np.int32, device="cpu")
    for off in OFFSETS:
        big.data[off : off + TILE] = np.arange(off, off + TILE, dtype=np.int64).astype(
            np.int32
        )
    big.to("npu")
    out = iron.tensor((TILE,), dtype=np.int32, device="cpu")
    run = pyxrt.run(kernel)
    run.set_arg(0, big.buffer_object())
    run.set_arg(1, out.buffer_object())
    return run, ParameterScratchpad(run, "params.txt"), out


@pytest.mark.parametrize("off", OFFSETS)
def test_offset_reaches_the_whole_buffer(setup, off):
    run, params, out = setup
    out.data.fill(-1)
    out.to("npu")
    params.write("off", np.int32(off))
    params.sync()
    run.start()
    run.wait2()
    out.to("cpu")
    np.testing.assert_array_equal(
        out.numpy(), np.arange(off, off + TILE, dtype=np.int64).astype(np.int32)
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
