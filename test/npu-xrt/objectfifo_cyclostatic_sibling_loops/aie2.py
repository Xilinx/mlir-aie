# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# Two sibling loops in the same scope, each cyclostatic on the same fifo.
# Each loop's trailing release drains its hoisted carry before the next loop.
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_LINES = 6
LINE_LEN = 8
ACQ = 3
CARRY = ACQ - 1
# Each loop consumes N_LINES outputs + CARRY trailing items; two loops total.
IN_LINES = 2 * (N_LINES + CARRY)
OUT_LINES = 2 * N_LINES
OUT_LEN = OUT_LINES * LINE_LEN


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_tile = tile(0, 2)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            in_ty = np.ndarray[(IN_LINES * LINE_LEN,), np.dtype[np.int8]]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            in_l3l2 = object_fifo("in_l3l2", shim, memtile, ACQ + 1, line_ty)
            in_l2l1 = object_fifo("in_l2l1", memtile, core_tile, ACQ + 1, line_ty)
            object_fifo_link(in_l3l2, in_l2l1)

            out_l1l2 = object_fifo("out_l1l2", core_tile, memtile, 3, line_ty)
            out_l2l3 = object_fifo("out_l2l3", memtile, shim, 3, line_ty)
            object_fifo_link(out_l1l2, out_l2l3)

            @core(core_tile)
            def core_body():
                for _ in range_(sys.maxsize):
                    # First sibling loop.
                    for _ in range_(N_LINES):
                        x = in_l2l1.acquire(ObjectFifoPort.Consume, ACQ)
                        out = out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            out[b] = x[0][b] + x[1][b] + x[2][b]
                        in_l2l1.release(ObjectFifoPort.Consume, 1)
                        out_l1l2.release(ObjectFifoPort.Produce, 1)
                    in_l2l1.release(ObjectFifoPort.Consume, CARRY)
                    # Second sibling loop (same shape).
                    for _ in range_(N_LINES):
                        x = in_l2l1.acquire(ObjectFifoPort.Consume, ACQ)
                        out = out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            out[b] = x[0][b] + x[1][b] + x[2][b]
                        in_l2l1.release(ObjectFifoPort.Consume, 1)
                        out_l1l2.release(ObjectFifoPort.Produce, 1)
                    in_l2l1.release(ObjectFifoPort.Consume, CARRY)

            @runtime_sequence(in_ty, out_ty)
            def sequence(In, Out):
                it = shim_dma_single_bd_task(
                    in_l3l2, In, offset=0, sizes=[1, 1, 1, IN_LINES * LINE_LEN]
                )
                ot = shim_dma_single_bd_task(
                    out_l2l3, Out, offset=0, sizes=[1, 1, 1, OUT_LEN], issue_token=True
                )
                dma_start_task(it, ot)
                dma_await_task(ot)
                dma_free_task(it)

        print(ctx.module)


if __name__ == "__main__":
    dev_str = sys.argv[1] if len(sys.argv) > 1 else "npu2"
    dev = {"npu1": AIEDevice.npu1_1col, "npu2": AIEDevice.npu2}[dev_str]
    build(dev)
