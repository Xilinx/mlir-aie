# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Two sibling loops in the same scope, each cyclostatic on the same fifo.
# Each loop's trailing release drains its hoisted carry before the next loop.
import sys
import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

N_LINES = 6
LINE_LEN = 8
ACQ = 3
CARRY = ACQ - 1
# Each loop consumes N_LINES outputs + CARRY trailing items; two loops total.
IN_LINES = 2 * (N_LINES + CARRY)
OUT_LINES = 2 * N_LINES
IN_LEN = IN_LINES * LINE_LEN
OUT_LEN = OUT_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


def core_body(of_in, of_out):
    for _ in range_(sys.maxsize):
        # First sibling loop.
        for _ in range_(N_LINES):
            x = of_in.acquire(ACQ)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = x[0][b] + x[1][b] + x[2][b]
            of_in.release(1)
            of_out.release(1)
        of_in.release(CARRY)
        # Second sibling loop (same shape).
        for _ in range_(N_LINES):
            x = of_in.acquire(ACQ)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = x[0][b] + x[1][b] + x[2][b]
            of_in.release(1)
            of_out.release(1)
        of_in.release(CARRY)


@iron.jit
def cyclostatic_sibling_loops(in_tensor: In, out_tensor: Out):
    in_ty = np.ndarray[(IN_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    of_in_l3l2 = ObjectFifo(LINE_TY, depth=ACQ + 1, name="in_l3l2")
    of_in_l2l1 = of_in_l3l2.cons().forward(name="in_l2l1", depth=ACQ + 1)
    of_out_l1l2 = ObjectFifo(LINE_TY, depth=3, name="out_l1l2")
    of_out_l2l3 = of_out_l1l2.cons().forward(name="out_l2l3", depth=3)

    worker = Worker(core_body, fn_args=[of_in_l2l1.cons(), of_out_l1l2.prod()])

    rt = Runtime()
    with rt.sequence(in_ty, out_ty) as (a_in, c_out):
        rt.start(worker)
        rt.fill(of_in_l3l2.prod(), a_in)
        rt.drain(of_out_l2l3.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-8, 8, size=(IN_LEN,), dtype=np.int8)

    src2d = src.reshape(IN_LINES, LINE_LEN).astype(np.int32)
    ref = np.empty((OUT_LINES, LINE_LEN), dtype=np.int32)
    for half in range(2):
        base_in = half * (N_LINES + CARRY)
        base_out = half * N_LINES
        for i in range(N_LINES):
            ref[base_out + i] = (
                src2d[base_in + i] + src2d[base_in + i + 1] + src2d[base_in + i + 2]
            )
    ref = ref.astype(np.int8).reshape(-1)

    inA = iron.tensor(src, dtype=np.int8, device="npu")
    out = iron.zeros(OUT_LEN, dtype=np.int8, device="npu")

    cyclostatic_sibling_loops(inA, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
