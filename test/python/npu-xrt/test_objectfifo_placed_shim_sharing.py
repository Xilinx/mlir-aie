# test_objectfifo_placed_shim_sharing.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""Three workers on one column, their shim ends left for the placer to place.

Their three outputs need three S2MM channels and the one shim tile has two.
The runtime sequence awaits each output before draining the next, so the
placer counts two of them once and allocation has them take turns on one
channel. Two workers read an input each; the third makes its own data, as
sending ends in one sequence never take turns (a shim MM2S task reports
itself complete before its words leave the shim).
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1

WORKERS = 3
N = 64
TILE = 16
MADE = 7


@iron.jit
def placed_shim_sharing(x: In, y: Out):
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(WORKERS * N,), np.dtype[np.int32]]

    def body(of_in, of_out, k):
        for _ in range_(N // TILE):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            for i in range_(TILE):
                b[i] = a[i] * 2 + k
            of_in.release(1)
            of_out.release(1)

    def make(of_out, k):
        for _ in range_(N // TILE):
            b = of_out.acquire(1)
            for i in range_(TILE):
                b[i] = k
            of_out.release(1)

    ins = [ObjectFifo(tile_ty, name=f"in{w}") for w in range(WORKERS - 1)]
    outs = [ObjectFifo(tile_ty, name=f"out{w}") for w in range(WORKERS)]
    workers = [
        Worker(body, fn_args=[ins[w].cons(), outs[w].prod(), w])
        for w in range(WORKERS - 1)
    ]
    workers.append(Worker(make, fn_args=[outs[-1].prod(), MADE]))

    def seq(x, y, in_hs, out_hs):
        for w in range(WORKERS):
            tap = TensorAccessPattern((WORKERS * N,), w * N, [1, 1, 1, N], [0, 0, 0, 1])
            tg = TaskGroup()
            if w < len(in_hs):
                in_hs[w].fill(x, tap, group=tg)
            out_hs[w].drain(y, tap, wait=True, group=tg)
            tg.finish()

    rt = Runtime(
        seq,
        [
            tensor_ty,
            tensor_ty,
            [f.prod() for f in ins],
            [f.cons() for f in outs],
        ],
    )
    return Program(NPU2Col1(), rt, workers=workers).resolve_program()


def test_objectfifo_placed_shim_sharing():
    x = iron.arange(WORKERS * N, dtype=np.int32)
    y = iron.zeros(WORKERS * N, dtype=np.int32, device="npu")
    placed_shim_sharing(x, y)
    y.to("cpu")

    ref = np.arange(WORKERS * N, dtype=np.int32) * 2
    ref += np.repeat(np.arange(WORKERS, dtype=np.int32), N)
    ref[(WORKERS - 1) * N :] = MADE
    np.testing.assert_array_equal(y.numpy(), ref)
