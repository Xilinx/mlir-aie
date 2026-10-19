# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test that one run sets another run's DMA offset through a scratchpad alias.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s
#
# The host sets the consumer's `start` to 0; the producer run, bound to the
# consumer's scratchpad alias, writes another start over it with no host step,
# and the consumer then moves the values from that start on (see aie_design.py).

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

N = 256
MOVED = 32


@pytest.fixture(scope="module")
def kernels():
    context = pyxrt.hw_context(acquire_device(), pyxrt.elf("aie.elf"))
    return (
        pyxrt.ext.kernel(context, "test:consume"),
        pyxrt.ext.kernel(context, "test:produce"),
    )


@pytest.mark.parametrize("start", [5, 100, 224])
def test_a_producer_run_sets_the_consumers_dma_offset(kernels, start):
    consume_kernel, produce_kernel = kernels
    values = iron.arange(N, dtype=np.int32, device="cpu")
    out = iron.tensor((N,), dtype=np.int32, device="cpu")
    out.data.fill(-1)
    values.to("npu")
    out.to("npu")
    consume = pyxrt.run(consume_kernel)
    consume.set_arg(0, values.buffer_object())
    consume.set_arg(1, out.buffer_object())
    params = ParameterScratchpad(consume, "params.txt")
    params.write("start", np.int32(0))
    params.sync()

    word = iron.tensor(np.array([start], dtype=np.int32), device="cpu")
    word.to("npu")
    produce = pyxrt.run(produce_kernel)
    produce.set_arg(0, word.buffer_object())
    produce.set_arg(1, params.alias())
    produce.start()
    produce.wait2()
    params.sync_from_device()
    assert params.read("start") == start

    consume.start()
    consume.wait2()
    out.to("cpu")
    expected = np.full(N, -1, dtype=np.int32)
    expected[:MOVED] = np.arange(start, start + MOVED, dtype=np.int32) + 1
    np.testing.assert_array_equal(out.numpy(), expected)
