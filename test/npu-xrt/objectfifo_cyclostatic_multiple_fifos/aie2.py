# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Two fifos in the same loop body, each cyclostatic with a different carry:
#   X: acq 3 / rel 1 (carry 2)
#   Y: acq 2 / rel 1 (carry 1)
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_LINES = 10
LINE_LEN = 8
X_ACQ = 3
Y_ACQ = 2
X_CARRY = X_ACQ - 1
Y_CARRY = Y_ACQ - 1
X_IN_LINES = N_LINES + X_CARRY
Y_IN_LINES = N_LINES + Y_CARRY
OUT_LEN = N_LINES * LINE_LEN


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_tile = tile(0, 2)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            x_in_ty = np.ndarray[(X_IN_LINES * LINE_LEN,), np.dtype[np.int8]]
            y_in_ty = np.ndarray[(Y_IN_LINES * LINE_LEN,), np.dtype[np.int8]]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            x_l3l2 = object_fifo("x_l3l2", shim, memtile, X_ACQ + 1, line_ty)
            x_l2l1 = object_fifo("x_l2l1", memtile, core_tile, X_ACQ + 1, line_ty)
            object_fifo_link(x_l3l2, x_l2l1)

            y_l3l2 = object_fifo("y_l3l2", shim, memtile, Y_ACQ + 1, line_ty)
            y_l2l1 = object_fifo("y_l2l1", memtile, core_tile, Y_ACQ + 1, line_ty)
            object_fifo_link(y_l3l2, y_l2l1)

            out_l1l2 = object_fifo("out_l1l2", core_tile, memtile, 3, line_ty)
            out_l2l3 = object_fifo("out_l2l3", memtile, shim, 3, line_ty)
            object_fifo_link(out_l1l2, out_l2l3)

            @core(core_tile)
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(N_LINES):
                        x = x_l2l1.acquire(ObjectFifoPort.Consume, X_ACQ)
                        y = y_l2l1.acquire(ObjectFifoPort.Consume, Y_ACQ)
                        out = out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            out[b] = x[0][b] + x[1][b] + x[2][b] + y[0][b] + y[1][b]
                        x_l2l1.release(ObjectFifoPort.Consume, 1)
                        y_l2l1.release(ObjectFifoPort.Consume, 1)
                        out_l1l2.release(ObjectFifoPort.Produce, 1)
                    x_l2l1.release(ObjectFifoPort.Consume, X_CARRY)
                    y_l2l1.release(ObjectFifoPort.Consume, Y_CARRY)

            @runtime_sequence(x_in_ty, y_in_ty, out_ty)
            def sequence(X, Y, Out):
                xt = shim_dma_single_bd_task(
                    x_l3l2, X, offset=0, sizes=[1, 1, 1, X_IN_LINES * LINE_LEN]
                )
                yt = shim_dma_single_bd_task(
                    y_l3l2, Y, offset=0, sizes=[1, 1, 1, Y_IN_LINES * LINE_LEN]
                )
                ot = shim_dma_single_bd_task(
                    out_l2l3, Out, offset=0, sizes=[1, 1, 1, OUT_LEN], issue_token=True
                )
                dma_start_task(xt, yt, ot)
                dma_await_task(ot)
                dma_free_task(xt, yt)

        print(ctx.module)


if __name__ == "__main__":
    dev_str = sys.argv[1] if len(sys.argv) > 1 else "npu2"
    dev = {"npu1": AIEDevice.npu1_1col, "npu2": AIEDevice.npu2}[dev_str]
    build(dev)
