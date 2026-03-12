# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

# Tests that the placed style (function returns None; module is built
# implicitly via @device/@core/@runtime_sequence context managers) works
# correctly through the JIT auto-detection path.

import sys
import pytest
import numpy as np
import aie.iron as iron

from aie.dialects.aie import ObjectFifoPort, core, device, object_fifo, tile
from aie.dialects.aiex import (
    dma_await_task,
    dma_free_task,
    dma_start_task,
    runtime_sequence,
    shim_dma_single_bd_task,
)
from aie.iron.controlflow import range_


@iron.jit
def passthrough(input, output):
    num_elements = np.size(input)
    dtype = input.dtype

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]

    @device(iron.get_current_device().resolve())
    def device_body():
        ShimTile = tile(0, 0)
        ComputeTile = tile(0, 2)

        of_in = object_fifo("in", ShimTile, ComputeTile, 2, tensor_ty)
        of_out = object_fifo("out", ComputeTile, ShimTile, 2, tensor_ty)

        @core(ComputeTile)
        def core_body():
            for _ in range_(sys.maxsize):
                elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(num_elements):
                    elem_out[i] = elem_in[i]
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        @runtime_sequence(tensor_ty, tensor_ty)
        def sequence(A, B):
            in_task = shim_dma_single_bd_task(of_in, A, sizes=[1, 1, 1, num_elements])
            out_task = shim_dma_single_bd_task(
                of_out, B, sizes=[1, 1, 1, num_elements], issue_token=True
            )
            dma_start_task(in_task, out_task)
            dma_await_task(out_task)
            dma_free_task(in_task)


@pytest.mark.parametrize("num_elements", [16, 64])
@pytest.mark.parametrize("dtype", [np.int32])
def test_jit_placed_style_passthrough(num_elements, dtype):
    input = iron.randint(0, 100, (num_elements,), dtype=dtype, device="npu")
    output = iron.zeros_like(input)

    passthrough(input, output)

    assert np.array_equal(input.numpy(), output.numpy())
