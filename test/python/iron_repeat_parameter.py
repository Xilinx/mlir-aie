# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# A fill's repeat_parameter is its start's: the tap's outer dimension gives the
# task its largest repeat count and its iteration, and the parameter the count
# each run pushes. The drain moves the matching rows by a length parameter.

import numpy as np

from aie.dialects.aiex import dma_start_task
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter


# CHECK-LABEL: test_fill_repeat_parameter
# CHECK: %[[IN:.*]] = aiex.dma_configure_task_for @of_in {
# CHECK: aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 64 sizes = [16, 1, 1, 64] strides = [64, 0, 0, 1])
# CHECK: } {repeat_count = 15 : i32}
# CHECK: aiex.dma_start_task(%[[IN]]) {repeat_parameter = @reps}
# CHECK: aie.dma_bd(%arg1 : memref<1024xi32> offset = 0 len = 0 sizes = [1, 1, 1, 64] strides = [0, 0, 0, 1]) {length_parameter = @tiles, length_unit = 64 : i32}
# CHECK: aiex.scratchpad_parameter @reps : i32
# CHECK: aiex.scratchpad_parameter @tiles : i32
def test_fill_repeat_parameter():
    print("test_fill_repeat_parameter")
    buf_ty = np.ndarray[(1024,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(64,), np.dtype[np.int32]]

    reps = ScratchpadParameter("reps", np.int32)
    tiles = ScratchpadParameter("tiles", np.int32)

    of_in = ObjectFifo(tile_ty, name="of_in")
    of_out = of_in.cons().forward(name="of_out")

    def sequence(a, b, in_h, out_h):
        stepped = TensorAccessPattern((1024,), 0, [16, 1, 1, 64], [64, 0, 0, 1])
        row = TensorAccessPattern((1024,), 0, [1, 1, 1, 64], [0, 0, 0, 1])
        in_h.fill(a, tap=stepped, repeat_parameter=reps)
        out_h.drain(b, tap=row, wait=True, length_parameter=tiles)

    rt = Runtime(sequence, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])
    print(Program(NPU2Col1(), rt).resolve_program())


# CHECK-LABEL: test_one_count_per_start
# CHECK: takes a repeat_count or a repeat_parameter, not both
def test_one_count_per_start():
    print("test_one_count_per_start")
    try:
        dma_start_task(None, repeat_count=1, repeat_parameter="reps")
    except ValueError as e:
        print(e)


test_fill_repeat_parameter()
test_one_count_per_start()
