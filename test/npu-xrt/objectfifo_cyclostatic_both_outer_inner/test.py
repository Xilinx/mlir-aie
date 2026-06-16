# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Cyclostatic acquire on BOTH outer (W) and inner (X) loops with different
# fifos. Each loop carries (acq - rel) items per iteration; the trailing
# release after each loop drains its carry.
import sys
import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

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
W_LEN = W_LINES * LINE_LEN
X_LEN = X_LINES * LINE_LEN
OUT_LEN = OUT_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


def core_body(of_w, of_x, of_out):
    for _ in range_(sys.maxsize):
        for _ in range_(N_OUTER):
            w = of_w.acquire(W_ACQ)
            for _ in range_(N_INNER):
                x = of_x.acquire(X_ACQ)
                out = of_out.acquire(1)
                for b in range_(LINE_LEN):
                    out[b] = w[0][b] + w[1][b] + x[0][b] + x[1][b] + x[2][b]
                of_x.release(1)
                of_out.release(1)
            of_x.release(X_CARRY)
            of_w.release(1)
        of_w.release(W_CARRY)


@iron.jit
def cyclostatic_both_outer_inner(w_tensor: In, x_tensor: In, out_tensor: Out):
    w_in_ty = np.ndarray[(W_LEN,), np.dtype[np.int8]]
    x_in_ty = np.ndarray[(X_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    of_w_l3l2 = ObjectFifo(LINE_TY, depth=W_ACQ + 1, name="w_l3l2")
    of_w_l2l1 = of_w_l3l2.cons().forward(name="w_l2l1", depth=W_ACQ + 1)
    of_x_l3l2 = ObjectFifo(LINE_TY, depth=X_ACQ + 1, name="x_l3l2")
    of_x_l2l1 = of_x_l3l2.cons().forward(name="x_l2l1", depth=X_ACQ + 1)
    of_out_l1l2 = ObjectFifo(LINE_TY, depth=3, name="out_l1l2")
    of_out_l2l3 = of_out_l1l2.cons().forward(name="out_l2l3", depth=3)

    worker = Worker(
        core_body,
        fn_args=[of_w_l2l1.cons(), of_x_l2l1.cons(), of_out_l1l2.prod()],
    )

    rt = Runtime()

    def sequence(w_in, x_in, c_out):
        of_w_l3l2.prod().fill(w_in)
        of_x_l3l2.prod().fill(x_in)
        of_out_l2l3.cons().drain(c_out, wait=True)

    rt.sequence(sequence, [w_in_ty, x_in_ty, out_ty])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(0)
    w = rng.integers(-8, 8, size=(W_LEN,), dtype=np.int8)
    x = rng.integers(-8, 8, size=(X_LEN,), dtype=np.int8)

    w2d = w.reshape(W_LINES, LINE_LEN).astype(np.int32)
    x2d = x.reshape(N_OUTER, N_INNER + (X_ACQ - 1), LINE_LEN).astype(np.int32)
    ref = np.empty((N_OUTER, N_INNER, LINE_LEN), dtype=np.int32)
    for o in range(N_OUTER):
        for i in range(N_INNER):
            ref[o, i] = w2d[o] + w2d[o + 1] + x2d[o, i] + x2d[o, i + 1] + x2d[o, i + 2]
    ref = ref.astype(np.int8).reshape(-1)

    inW = iron.tensor(w, dtype=np.int8, device="npu")
    inX = iron.tensor(x, dtype=np.int8, device="npu")
    out = iron.zeros(OUT_LEN, dtype=np.int8, device="npu")

    cyclostatic_both_outer_inner(inW, inX, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
