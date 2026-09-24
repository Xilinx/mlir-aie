# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

# A ScratchpadParameter used only by runtime transfers, never by a Worker, is
# still declared once by the Program, and each transfer carries it by name.

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.iron.scratchpad_parameter import ScratchpadParameter
from aie.helpers.taplib import TensorAccessPattern


# CHECK-NOT: aiex.scratchpad_parameter
# CHECK: aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 0 sizes = [1, 1, 8] strides = [0, 0, 1]) {length_parameter = @rows, length_unit = 8 : i32, offset_parameter = @start}
# CHECK: aie.dma_bd(%arg1 : memref<256xi32> offset = 0 len = 0 sizes = [1, 1, 8] strides = [0, 0, 1]) {length_parameter = @rows, length_unit = 8 : i32}
# CHECK: aiex.scratchpad_parameter @start : i32
# CHECK-NEXT: aiex.scratchpad_parameter @rows : i32
# CHECK-NOT: aiex.scratchpad_parameter
def test_transfer_only_parameters():
    buf_ty = np.ndarray[(256,), np.dtype[np.int32]]
    row_ty = np.ndarray[(8,), np.dtype[np.int32]]

    rows = ScratchpadParameter("rows", np.int32)
    start = ScratchpadParameter("start", np.int32)

    of_in = ObjectFifo(row_ty, name="of_in")
    of_out = ObjectFifo(row_ty, name="of_out")

    def core_fn(of_in, of_out):
        for _ in range_(4):
            in_elem = of_in.acquire(1)
            out_elem = of_out.acquire(1)
            for i in range_(8):
                out_elem[i] = in_elem[i]
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_fn, [of_in.cons(), of_out.prod()], while_true=False)

    def sequence(a, b, in_h, out_h):
        row = TensorAccessPattern((256,), 0, [1, 1, 1, 8], [0, 0, 0, 1])
        in_h.fill(a, tap=row, offset_parameter=start, length_parameter=rows)
        out_h.drain(b, tap=row, wait=True, length_parameter=rows)

    rt = Runtime(sequence, [buf_ty, buf_ty, of_in.prod(), of_out.cons()])
    print(Program(NPU2Col1(), rt, workers=[worker]).resolve_program())


test_transfer_only_parameters()
