# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Two ObjectFifos leave one tile over one sending channel. A channel cycling
# on one fifo's buffers never reaches the other's, so the core pushes the
# descriptor of each buffer it releases.
import sys

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_

LINE_LEN = 8
N_LINES = 16
LEN = N_LINES * LINE_LEN

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int32]]


def split_lines(of_in, of_even, of_odd):
    """Send alternate lines down the two fifos sharing the channel."""
    for _ in range_(sys.maxsize):
        for _ in range_(2):
            src = of_in.acquire(1)
            dst = of_even.acquire(1)
            for b in range_(LINE_LEN):
                dst[b] = src[b]
            of_in.release(1)
            of_even.release(1)

            src = of_in.acquire(1)
            dst = of_odd.acquire(1)
            for b in range_(LINE_LEN):
                dst[b] = src[b]
            of_in.release(1)
            of_odd.release(1)


def add(amount):
    def body(of_in, of_out):
        for _ in range_(sys.maxsize):
            src = of_in.acquire(1)
            dst = of_out.acquire(1)
            for b in range_(LINE_LEN):
                dst[b] = src[b] + amount
            of_in.release(1)
            of_out.release(1)

    return body


@iron.jit
def shared_send_channel(in_tensor: In, even_tensor: Out, odd_tensor: Out):
    tensor_ty = np.ndarray[(LEN,), np.dtype[np.int32]]
    half_ty = np.ndarray[(LEN // 2,), np.dtype[np.int32]]

    of_in = ObjectFifo(LINE_TY, depth=2, name="in")
    # Both leave the splitter over channel 0, which selects push_queue: the
    # core pushes the descriptor of each buffer it releases.
    of_even = ObjectFifo(LINE_TY, depth=2, name="even", packet=True, via_DMA=True)
    of_odd = ObjectFifo(LINE_TY, depth=2, name="odd", packet=True, via_DMA=True)
    of_even_out = ObjectFifo(LINE_TY, depth=2, name="even_out")
    of_odd_out = ObjectFifo(LINE_TY, depth=2, name="odd_out")

    splitter = Worker(
        split_lines,
        fn_args=[of_in.cons(), of_even.prod(channel=0), of_odd.prod(channel=0)],
    )
    even_worker = Worker(add(1), fn_args=[of_even.cons(), of_even_out.prod()])
    odd_worker = Worker(add(2), fn_args=[of_odd.cons(), of_odd_out.prod()])

    def sequence(src, even, odd, in_handle, even_handle, odd_handle):
        in_handle.fill(src)
        even_handle.drain(even, wait=True)
        odd_handle.drain(odd, wait=True)

    rt = Runtime(
        sequence,
        [
            tensor_ty,
            half_ty,
            half_ty,
            of_in.prod(),
            of_even_out.cons(),
            of_odd_out.cons(),
        ],
    )

    return Program(
        iron.get_current_device(),
        rt,
        workers=[splitter, even_worker, odd_worker],
    ).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-1000, 1000, size=(LEN,), dtype=np.int32)
    lines = src.reshape(N_LINES, LINE_LEN)
    even_ref = (lines[0::2] + 1).reshape(-1)
    odd_ref = (lines[1::2] + 2).reshape(-1)

    src_tensor = iron.tensor(src, dtype=np.int32, device="npu")
    even_tensor = iron.zeros(LEN // 2, dtype=np.int32, device="npu")
    odd_tensor = iron.zeros(LEN // 2, dtype=np.int32, device="npu")

    shared_send_channel(src_tensor, even_tensor, odd_tensor)
    if not np.array_equal(even_tensor.numpy(), even_ref):
        print("FAIL: even output mismatch")
        sys.exit(1)
    if not np.array_equal(odd_tensor.numpy(), odd_ref):
        print("FAIL: odd output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
