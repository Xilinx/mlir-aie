# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Pre-implementation spec for the cyclostatic-acquire peeling rewrite.
#
# Two-core pipeline:
#   shim -> A -> B -> shim
# Core A is a PRODUCER: forwards shim input verbatim to a_to_b (no cyclostatic).
# Core B is a CYCLOSTATIC CONSUMER of a_to_b: 3-line sliding window sum.
#
# This is the simplest 2-core test that exercises cross-core cyclostatic
# sync on real hardware: B's `acquire(a_to_b, 3)` per iter requires A to
# have produced N+2 items by the time B reaches iter N. Peel preserves the
# order of B's per-iter acquire-and-release on a_to_b, so the producer-side
# lock count drains correctly across iterations.
#
# A purely mutual ping-pong (both cores cyclostatic on a fifo the other
# produces) has a structural circular dependency that requires extra fifo
# depth or a different acq/rel ordering to avoid deadlock — that pattern
# would be a separate (and harder) test.
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_LINES = 12
LINE_LEN = 8
FIFO_DEPTH = 3
A_OUT_LINES = N_LINES + (FIFO_DEPTH - 1)  # B needs N_LINES + 2 from A
A_IN_LINES = A_OUT_LINES  # A forwards 1:1 from shim
IN_LEN = A_IN_LINES * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_a = tile(0, 2)
            core_b = tile(0, 3)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            in_ty = np.ndarray[(IN_LEN,), np.dtype[np.int8]]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            # shim -> memtile -> core_a
            in_l3l2 = object_fifo("in_l3l2", shim, memtile, FIFO_DEPTH, line_ty)
            in_l2a = object_fifo("in_l2a", memtile, core_a, FIFO_DEPTH, line_ty)
            object_fifo_link(in_l3l2, in_l2a)

            # core_a -> core_b (direct, this is what B is cyclostatic on)
            a_to_b = object_fifo("a_to_b", core_a, core_b, FIFO_DEPTH, line_ty)

            # core_b -> memtile -> shim (output)
            out_bl2 = object_fifo("out_bl2", core_b, memtile, FIFO_DEPTH, line_ty)
            out_l2l3 = object_fifo("out_l2l3", memtile, shim, FIFO_DEPTH, line_ty)
            object_fifo_link(out_bl2, out_l2l3)

            # Core A: simple producer, forward 1:1 (no cyclostatic).
            @core(core_a)
            def core_a_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(A_OUT_LINES):
                        src = in_l2a.acquire(ObjectFifoPort.Consume, 1)
                        dst = a_to_b.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            dst[b] = src[b]
                        in_l2a.release(ObjectFifoPort.Consume, 1)
                        a_to_b.release(ObjectFifoPort.Produce, 1)

            # Core B: cyclostatic 3-line sliding-window sum on a_to_b.
            @core(core_b)
            def core_b_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(N_LINES):
                        win = a_to_b.acquire(ObjectFifoPort.Consume, FIFO_DEPTH)
                        out = out_bl2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            out[b] = win[0][b] + win[1][b] + win[2][b]
                        a_to_b.release(ObjectFifoPort.Consume, 1)
                        out_bl2.release(ObjectFifoPort.Produce, 1)
                    a_to_b.release(ObjectFifoPort.Consume, FIFO_DEPTH - 1)

            @runtime_sequence(in_ty, out_ty)
            def sequence(In, Out):
                in_task = shim_dma_single_bd_task(
                    in_l3l2, In, offset=0, sizes=[1, 1, 1, IN_LEN]
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
