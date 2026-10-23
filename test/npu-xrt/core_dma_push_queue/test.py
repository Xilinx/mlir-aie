# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# An out-of-order channel takes the objects its queue was pushed for and
# stops. This design moves more objects than one push covers, so the receiving
# core has to push again as it goes for the run to finish at all.
import os
import sys
import time

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

LINE_LEN = 8
# One push covers 256 objects, so this takes two.
N_LINES = 400
LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int32]]


def pass_through(of_in, of_out):
    for _ in range_(sys.maxsize):
        src = of_in.acquire(1)
        dst = of_out.acquire(1)
        for b in range_(LINE_LEN):
            dst[b] = src[b]
        of_in.release(1)
        of_out.release(1)


def add_one(of_in, of_out):
    for _ in range_(sys.maxsize):
        src = of_in.acquire(1)
        dst = of_out.acquire(1)
        for b in range_(LINE_LEN):
            dst[b] = src[b] + 1
        of_in.release(1)
        of_out.release(1)


# The wait for room on the task queue is what a push costs beyond the writes
# that make it, so the run is timed with it and without it.
CHECK_QUEUE_SPACE = os.environ.get("CHECK_QUEUE_SPACE", "1") != "0"


@iron.jit(
    aiecc_flags=[f"--check-queue-space={'true' if CHECK_QUEUE_SPACE else 'false'}"]
)
def core_dma_push_queue(in_tensor: In, out_tensor: Out):
    tensor_ty = np.ndarray[(LEN,), np.dtype[np.int32]]

    of_in = ObjectFifo(LINE_TY, depth=2, name="in")
    # The core on the receiving tile pushes onto this channel's queue: it
    # names no iter_count, so it receives objects without end.
    # Adjacent tiles would share memory, and a shared-memory fifo has no
    # channel to push onto.
    of_mid = ObjectFifo(
        LINE_TY,
        depth=2,
        name="mid",
        packet=True,
        out_of_order=True,
        via_DMA=True,
    )
    of_out = ObjectFifo(LINE_TY, depth=2, name="out")

    sender = Worker(pass_through, fn_args=[of_in.cons(), of_mid.prod()])
    receiver = Worker(add_one, fn_args=[of_mid.cons(), of_out.prod()])

    def sequence(src, dst, in_handle, out_handle):
        in_handle.fill(src)
        out_handle.drain(dst, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])

    return Program(
        iron.get_current_device(), rt, workers=[sender, receiver]
    ).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-1000, 1000, size=(LEN,), dtype=np.int32)
    ref = src + 1

    src_tensor = iron.tensor(src, dtype=np.int32, device="npu")
    dst_tensor = iron.zeros(LEN, dtype=np.int32, device="npu")

    core_dma_push_queue(src_tensor, dst_tensor)
    if not np.array_equal(dst_tensor.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)

    # The first run compiles and loads; time the ones after it.
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        core_dma_push_queue(src_tensor, dst_tensor)
    elapsed = (time.perf_counter() - start) / runs
    checked = "with" if CHECK_QUEUE_SPACE else "without"
    print(f"TIME {elapsed * 1e6:.1f} us per run, {checked} the queue-space wait")
    print("PASS")


if __name__ == "__main__":
    main()
