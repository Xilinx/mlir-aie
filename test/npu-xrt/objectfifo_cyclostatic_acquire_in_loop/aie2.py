# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# Regression for https://github.com/Xilinx/mlir-aie/issues/2463.
# Cyclostatic acquire (hold 3, slide window by 1 per iter) inside an scf.for
# loop. The default dynamic-objFifos lowering emits AcquireGreaterEqual(3)
# every iter instead of (1) for iters >= 1, exhausting the producer pool and
# deadlocking on hardware.
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_LINES = 14  # inner loop trip count
LINE_LEN = 8  # bytes per FIFO element
OUT_LEN = N_LINES * LINE_LEN
FIFO_DEPTH = 3  # cyclostatic window size


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_tile = tile(0, 2)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            in_ty = np.ndarray[
                ((N_LINES + FIFO_DEPTH - 1) * LINE_LEN,), np.dtype[np.int8]
            ]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            in_l3l2 = object_fifo("in_l3l2", shim, memtile, FIFO_DEPTH, line_ty)
            in_l2l1 = object_fifo("in_l2l1", memtile, core_tile, FIFO_DEPTH, line_ty)
            object_fifo_link(in_l3l2, in_l2l1)

            out_l1l2 = object_fifo("out_l1l2", core_tile, memtile, FIFO_DEPTH, line_ty)
            out_l2l3 = object_fifo("out_l2l3", memtile, shim, FIFO_DEPTH, line_ty)
            object_fifo_link(out_l1l2, out_l2l3)

            @core(core_tile)
            def core_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(N_LINES):
                        win = in_l2l1.acquire(ObjectFifoPort.Consume, FIFO_DEPTH)
                        out = out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        # Sum the 3-line window into the output line.
                        for b in range_(LINE_LEN):
                            out[b] = win[0][b] + win[1][b] + win[2][b]
                        in_l2l1.release(ObjectFifoPort.Consume, 1)
                        out_l1l2.release(ObjectFifoPort.Produce, 1)
                    in_l2l1.release(ObjectFifoPort.Consume, FIFO_DEPTH - 1)

            @runtime_sequence(in_ty, out_ty)
            def sequence(In, Out):
                in_task = shim_dma_single_bd_task(
                    in_l3l2,
                    In,
                    offset=0,
                    sizes=[1, 1, 1, (N_LINES + FIFO_DEPTH - 1) * LINE_LEN],
                )
                out_task = shim_dma_single_bd_task(
                    out_l2l3, Out, offset=0, sizes=[1, 1, 1, OUT_LEN], issue_token=True
                )
                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

        print(ctx.module)


if __name__ == "__main__":
    dev_str = sys.argv[1] if len(sys.argv) > 1 else "npu2"
    dev = {"npu1": AIEDevice.npu1_1col, "npu2": AIEDevice.npu2}[dev_str]
    build(dev)
