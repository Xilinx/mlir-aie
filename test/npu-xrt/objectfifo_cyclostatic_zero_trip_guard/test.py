# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# E2E for the runtime-trip-count guard added by the cyclostatic acquire-in-
# loop peel rewrite. Two @iron.jit designs share the same cyclostatic core
# body; they differ only in their Runtime sequence:
#   - normal: shim sends trip=N_LINES and the input window; the peel guard
#             goes cond=true and the cyclostatic body runs; output matches
#             the 3-line sliding-window sum reference.
#   - zero:   shim sends trip=0 and no input window; the peel guard goes
#             cond=false and the cyclostatic body must be skipped without
#             deadlocking on the producer-side lock.
import sys
import numpy as np

import aie.iron as iron
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_, if_
from aie.dialects.arith import cmpi, CmpIPredicate, constant
from aie.extras import types as T

N_LINES = 14
LINE_LEN = 8
FIFO_DEPTH = 3
IN_LEN = (N_LINES + FIFO_DEPTH - 1) * LINE_LEN
OUT_LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
TRIP_TY = np.ndarray[(1,), np.dtype[np.int32]]
DONE_TY = np.ndarray[(1,), np.dtype[np.int32]]


def core_body(of_in, of_out, of_trip, of_done):
    for _ in range_(sys.maxsize):
        trip_sv = of_trip.acquire(1)
        trip = trip_sv[0]
        # Inner loop ub is a runtime value from of_trip → the peel cannot
        # prove >= 1 statically and emits an scf.if guard around iter-0.
        for _ in range_(trip):
            win = of_in.acquire(FIFO_DEPTH)
            out = of_out.acquire(1)
            for b in range_(LINE_LEN):
                out[b] = win[0][b] + win[1][b] + win[2][b]
            of_in.release(1)
            of_out.release(1)
        # User trailing drain: only valid when the loop ran (the peel held
        # tracker only kicks in then). Mirror that with an scf.if so the
        # zero-trip path doesn't release a never-acquired lock.
        zero_i32 = constant(T.i32(), 0)
        nonzero = cmpi(CmpIPredicate.sgt, trip, zero_i32)
        with if_(nonzero, hasElse=False):
            of_in.release(FIFO_DEPTH - 1)
        done = of_done.acquire(1)
        done[0] = trip
        of_done.release(1)
        of_trip.release(1)


@iron.jit
def cyclostatic_normal(in_tensor, trip_tensor, out_tensor, done_tensor):
    in_ty = np.ndarray[(IN_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]
    trip_ty = TRIP_TY
    done_ty = DONE_TY

    of_in_l3l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="in_l3l2")
    of_in_l2l1 = of_in_l3l2.cons().forward(name="in_l2l1", depth=FIFO_DEPTH)
    of_out_l1l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="out_l1l2")
    of_out_l2l3 = of_out_l1l2.cons().forward(name="out_l2l3", depth=FIFO_DEPTH)
    of_trip = ObjectFifo(TRIP_TY, depth=1, name="trip_fifo")
    of_done_l1l2 = ObjectFifo(DONE_TY, depth=1, name="done_l1l2")
    of_done_l2l3 = of_done_l1l2.cons().forward(name="done_l2l3", depth=1)

    worker = Worker(
        core_body,
        fn_args=[
            of_in_l2l1.cons(),
            of_out_l1l2.prod(),
            of_trip.cons(),
            of_done_l1l2.prod(),
        ],
    )

    rt = Runtime()
    with rt.sequence(in_ty, trip_ty, out_ty, done_ty) as (a_in, trip, c_out, done):
        rt.start(worker)
        rt.fill(of_in_l3l2.prod(), a_in)
        rt.fill(of_trip.prod(), trip)
        rt.drain(of_out_l2l3.cons(), c_out)
        rt.drain(of_done_l2l3.cons(), done, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


# The zero-trip path reuses the same @iron.jit'd cyclostatic_normal: it sends
# trip=0 alongside a dummy input window and a throwaway output buffer. The
# core body's `for _ in range_(trip)` runs zero iterations and the scf.if
# guard around the trailing release fires. The point of the test is that the
# producer-side lock doesn't deadlock and `done` still reaches the shim.


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-32, 32, size=(IN_LEN,), dtype=np.int8)
    src2d = src.reshape(N_LINES + FIFO_DEPTH - 1, LINE_LEN).astype(np.int32)
    ref_out = (
        (src2d[:N_LINES] + src2d[1 : N_LINES + 1] + src2d[2 : N_LINES + 2])
        .astype(np.int8)
        .reshape(-1)
    )

    # --- cond=true path ---
    inA = iron.tensor(src.copy(), dtype=np.int8, device="npu")
    trip = iron.tensor(np.array([N_LINES], dtype=np.int32), dtype=np.int32, device="npu")
    out = iron.zeros((OUT_LEN,), dtype=np.int8, device="npu")
    done = iron.zeros((1,), dtype=np.int32, device="npu")

    cyclostatic_normal(inA, trip, out, done)
    if not np.array_equal(out.numpy(), ref_out):
        print("FAIL: normal output mismatch")
        sys.exit(1)
    if int(done.numpy()[0]) != N_LINES:
        print(f"FAIL: normal done {int(done.numpy()[0])} != {N_LINES}")
        sys.exit(1)

    # --- cond=false path (the actual point of this test) ---
    inA0 = iron.zeros((IN_LEN,), dtype=np.int8, device="npu")
    trip0 = iron.tensor(np.array([0], dtype=np.int32), dtype=np.int32, device="npu")
    out0 = iron.zeros((OUT_LEN,), dtype=np.int8, device="npu")
    done0 = iron.zeros((1,), dtype=np.int32, device="npu")
    cyclostatic_normal(inA0, trip0, out0, done0)
    if int(done0.numpy()[0]) != 0:
        print(f"FAIL: zero done {int(done0.numpy()[0])} != 0")
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
