# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# Cyclostatic acquire on BOTH outer (W) and inner (X) loops with different
# fifos. Each loop carries (acq - rel) items per iteration; the trailing
# release after each loop drains its carry.
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_OUTER = 4
N_INNER = 7
LINE_LEN = 8
W_ACQ = 2
X_ACQ = 3
W_CARRY = W_ACQ - 1
X_CARRY = X_ACQ - 1

# X has the same cyclostatic pattern as the first test (carry 2). W has
# carry 1. Each outer iter produces N_INNER output lines using W's window
# (per outer iter) and X's window (per inner iter).
W_LINES = N_OUTER + W_CARRY
X_LINES = (N_INNER + X_CARRY) * N_OUTER
OUT_LINES = N_OUTER * N_INNER
OUT_LEN = OUT_LINES * LINE_LEN


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_tile = tile(0, 2)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            w_in_ty = np.ndarray[(W_LINES * LINE_LEN,), np.dtype[np.int8]]
            x_in_ty = np.ndarray[(X_LINES * LINE_LEN,), np.dtype[np.int8]]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            w_l3l2 = object_fifo("w_l3l2", shim, memtile, W_ACQ + 1, line_ty)
            w_l2l1 = object_fifo("w_l2l1", memtile, core_tile, W_ACQ + 1, line_ty)
            object_fifo_link(w_l3l2, w_l2l1)

            x_l3l2 = object_fifo("x_l3l2", shim, memtile, X_ACQ + 1, line_ty)
            x_l2l1 = object_fifo("x_l2l1", memtile, core_tile, X_ACQ + 1, line_ty)
            object_fifo_link(x_l3l2, x_l2l1)

            out_l1l2 = object_fifo("out_l1l2", core_tile, memtile, 3, line_ty)
            out_l2l3 = object_fifo("out_l2l3", memtile, shim, 3, line_ty)
            object_fifo_link(out_l1l2, out_l2l3)

            @core(core_tile)
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(N_OUTER):
                        w = w_l2l1.acquire(ObjectFifoPort.Consume, W_ACQ)
                        for _ in range_(N_INNER):
                            x = x_l2l1.acquire(ObjectFifoPort.Consume, X_ACQ)
                            out = out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                            # out[b] = w[0][b] + w[1][b] + x[0][b] + x[1][b] + x[2][b]
                            for b in range_(LINE_LEN):
                                out[b] = w[0][b] + w[1][b] + x[0][b] + x[1][b] + x[2][b]
                            x_l2l1.release(ObjectFifoPort.Consume, 1)
                            out_l1l2.release(ObjectFifoPort.Produce, 1)
                        x_l2l1.release(ObjectFifoPort.Consume, X_CARRY)
                        w_l2l1.release(ObjectFifoPort.Consume, 1)
                    w_l2l1.release(ObjectFifoPort.Consume, W_CARRY)

            @runtime_sequence(w_in_ty, x_in_ty, out_ty)
            def sequence(W, X, Out):
                w_task = shim_dma_single_bd_task(
                    w_l3l2, W, offset=0, sizes=[1, 1, 1, W_LINES * LINE_LEN]
                )
                x_task = shim_dma_single_bd_task(
                    x_l3l2, X, offset=0, sizes=[1, 1, 1, X_LINES * LINE_LEN]
                )
                out_task = shim_dma_single_bd_task(
                    out_l2l3, Out, offset=0, sizes=[1, 1, 1, OUT_LEN], issue_token=True
                )
                dma_start_task(w_task, x_task, out_task)
                dma_await_task(out_task)
                dma_free_task(w_task, x_task)

        print(ctx.module)


if __name__ == "__main__":
    dev_str = sys.argv[1] if len(sys.argv) > 1 else "npu2"
    dev = {"npu1": AIEDevice.npu1_1col, "npu2": AIEDevice.npu2}[dev_str]
    build(dev)
