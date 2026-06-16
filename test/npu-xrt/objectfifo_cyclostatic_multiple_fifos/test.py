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

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

N_LINES = 10
LINE_LEN = 8
X_ACQ = 3
Y_ACQ = 2
X_CARRY = X_ACQ - 1
Y_CARRY = Y_ACQ - 1
X_IN_LINES = N_LINES + X_CARRY
Y_IN_LINES = N_LINES + Y_CARRY
X_LEN = X_IN_LINES * LINE_LEN
Y_LEN = Y_IN_LINES * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


def core_body(of_x, of_y, of_out):
    for _ in range_(sys.maxsize):
        for _ in range_(N_LINES):
            x = of_x.acquire(X_ACQ)
            y = of_y.acquire(Y_ACQ)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = x[0][b] + x[1][b] + x[2][b] + y[0][b] + y[1][b]
            of_x.release(1)
            of_y.release(1)
            of_out.release(1)
        of_x.release(X_CARRY)
        of_y.release(Y_CARRY)


@iron.jit
def cyclostatic_multiple_fifos(x_tensor: In, y_tensor: In, out_tensor: Out):
    x_in_ty = np.ndarray[(X_LEN,), np.dtype[np.int8]]
    y_in_ty = np.ndarray[(Y_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    of_x_l3l2 = ObjectFifo(LINE_TY, depth=X_ACQ + 1, name="x_l3l2")
    of_x_l2l1 = of_x_l3l2.cons().forward(name="x_l2l1", depth=X_ACQ + 1)
    of_y_l3l2 = ObjectFifo(LINE_TY, depth=Y_ACQ + 1, name="y_l3l2")
    of_y_l2l1 = of_y_l3l2.cons().forward(name="y_l2l1", depth=Y_ACQ + 1)
    of_out_l1l2 = ObjectFifo(LINE_TY, depth=3, name="out_l1l2")
    of_out_l2l3 = of_out_l1l2.cons().forward(name="out_l2l3", depth=3)

    worker = Worker(
        core_body,
        fn_args=[of_x_l2l1.cons(), of_y_l2l1.cons(), of_out_l1l2.prod()],
    )

    rt = Runtime()

    def sequence(x_in, y_in, c_out):
        of_x_l3l2.prod().fill(x_in)
        of_y_l3l2.prod().fill(y_in)
        of_out_l2l3.cons().drain(c_out, wait=True)

    rt.sequence(sequence, [x_in_ty, y_in_ty, out_ty])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def main():
    rng = np.random.default_rng(0)
    x = rng.integers(-8, 8, size=(X_LEN,), dtype=np.int8)
    y = rng.integers(-8, 8, size=(Y_LEN,), dtype=np.int8)

    x2d = x.reshape(X_IN_LINES, LINE_LEN).astype(np.int32)
    y2d = y.reshape(Y_IN_LINES, LINE_LEN).astype(np.int32)
    ref = np.empty((N_LINES, LINE_LEN), dtype=np.int32)
    for i in range(N_LINES):
        ref[i] = x2d[i] + x2d[i + 1] + x2d[i + 2] + y2d[i] + y2d[i + 1]
    ref = ref.astype(np.int8).reshape(-1)

    inX = iron.tensor(x, dtype=np.int8, device="npu")
    inY = iron.tensor(y, dtype=np.int8, device="npu")
    out = iron.zeros(OUT_LEN, dtype=np.int8, device="npu")

    cyclostatic_multiple_fifos(inX, inY, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
