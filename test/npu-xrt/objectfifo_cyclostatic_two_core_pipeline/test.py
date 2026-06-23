# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Two-core pipeline (#2463):
#   shim -> A -> B -> shim
# Core A is a PRODUCER: forwards shim input verbatim to a_to_b (no cyclostatic).
# Core B is a CYCLOSTATIC CONSUMER of a_to_b: 3-line sliding window sum.
#
# This is the simplest 2-core test that exercises cross-core cyclostatic
# sync on real hardware: B's `acquire(a_to_b, 3)` per iter requires A to
# have produced N+2 items by the time B reaches iter N. Peel preserves the
# order of B's per-iter acquire-and-release on a_to_b, so the producer-side
# lock count drains correctly across iterations.
import sys
import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

N_LINES = 12
LINE_LEN = 8
FIFO_DEPTH = 3
A_OUT_LINES = N_LINES + (FIFO_DEPTH - 1)  # B needs N_LINES + 2 from A
A_IN_LINES = A_OUT_LINES  # A forwards 1:1 from shim
IN_LEN = A_IN_LINES * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


# Core A: simple producer, forward 1:1 (no cyclostatic).
def core_a_body(of_in, of_a_to_b):
    for _ in range_(sys.maxsize):
        for _ in range_(A_OUT_LINES):
            src = of_in.acquire(1)
            dst = of_a_to_b.acquire(1)
            for b in range_(LINE_LEN):
                dst[b] = src[b]
            of_in.release(1)
            of_a_to_b.release(1)


# Core B: cyclostatic 3-line sliding-window sum on a_to_b.
def core_b_body(of_a_to_b, of_out):
    for _ in range_(sys.maxsize):
        for _ in range_(N_LINES):
            win = of_a_to_b.acquire(FIFO_DEPTH)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = win[0][b] + win[1][b] + win[2][b]
            of_a_to_b.release(1)
            of_out.release(1)
        of_a_to_b.release(FIFO_DEPTH - 1)


@iron.jit
def cyclostatic_two_core_pipeline(in_tensor: In, out_tensor: Out):
    in_ty = np.ndarray[(IN_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    # shim -> memtile -> core_a
    of_in_l3l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="in_l3l2")
    of_in_l2a = of_in_l3l2.cons().forward(name="in_l2a", depth=FIFO_DEPTH)

    # core_a -> core_b (direct, this is what B is cyclostatic on)
    of_a_to_b = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="a_to_b")

    # core_b -> memtile -> shim (output)
    of_out_bl2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="out_bl2")
    of_out_l2l3 = of_out_bl2.cons().forward(name="out_l2l3", depth=FIFO_DEPTH)

    worker_a = Worker(core_a_body, fn_args=[of_in_l2a.cons(), of_a_to_b.prod()])
    worker_b = Worker(core_b_body, fn_args=[of_a_to_b.cons(), of_out_bl2.prod()])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(worker_a, worker_b)
        rt.fill(of_in_l3l2.prod(), a_in)
        rt.drain(of_out_l2l3.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)

    # Core A forwards verbatim, so B sees `src` lines on a_to_b.
    # Core B's per-output-line i: out[i] = src[i] + src[i+1] + src[i+2].
    src2d = src.reshape(A_OUT_LINES, LINE_LEN).astype(np.int32)
    ref = (
        (src2d[:N_LINES] + src2d[1 : N_LINES + 1] + src2d[2 : N_LINES + 2])
        .astype(np.int8)
        .reshape(-1)
    )

    inA = iron.tensor(src, dtype=np.int8, device="npu")
    out = iron.zeros(OUT_LEN, dtype=np.int8, device="npu")

    cyclostatic_two_core_pipeline(inA, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
