# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Regression for https://github.com/Xilinx/mlir-aie/issues/2463.
# Cyclostatic acquire (hold 3, slide window by 1 per iter) inside an scf.for
# loop. Without the peel rewrite, the dynamic-objFifos lowering emits
# AcquireGreaterEqual(3) every iter instead of (1) for iters >= 1, exhausting
# the producer pool and deadlocking on hardware.
import sys
import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

N_LINES = 14
LINE_LEN = 8
FIFO_DEPTH = 3
IN_LEN = (N_LINES + FIFO_DEPTH - 1) * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


def core_body(of_in, of_out):
    for _ in range_(sys.maxsize):
        for _ in range_(N_LINES):
            win = of_in.acquire(FIFO_DEPTH)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = win[0][b] + win[1][b] + win[2][b]
            of_in.release(1)
            of_out.release(1)
        of_in.release(FIFO_DEPTH - 1)


@iron.jit
def cyclostatic_acquire_in_loop(in_tensor: In, out_tensor: Out):
    in_ty = np.ndarray[(IN_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    of_in_l3l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="in_l3l2")
    of_in_l2l1 = of_in_l3l2.cons().forward(name="in_l2l1", depth=FIFO_DEPTH)
    of_out_l1l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="out_l1l2")
    of_out_l2l3 = of_out_l1l2.cons().forward(name="out_l2l3", depth=FIFO_DEPTH)

    worker = Worker(core_body, fn_args=[of_in_l2l1.cons(), of_out_l1l2.prod()])

    rt = Runtime()

    def sequence(a_in, c_out):
        of_in_l3l2.prod().fill(a_in)
        of_out_l2l3.cons().drain(c_out, wait=True)

    rt.sequence(sequence, [in_ty, out_ty])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)
    src2d = src.reshape(N_LINES + FIFO_DEPTH - 1, LINE_LEN).astype(np.int32)
    ref = (
        (src2d[:N_LINES] + src2d[1 : N_LINES + 1] + src2d[2 : N_LINES + 2])
        .astype(np.int8)
        .reshape(-1)
    )

    inA = iron.tensor(src.copy(), dtype=np.int8, device="npu")
    out = iron.zeros((OUT_LEN,), dtype=np.int8, device="npu")

    cyclostatic_acquire_in_loop(inA, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
