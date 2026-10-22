# test_benchmark_split.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""run_iters on the NPU splits a dispatch into its fixed overhead and the NPU's work.

An empty dispatch on the kernel's own context follows each timed call; the
NPU's time is each dispatch less the overhead's median. The design's output
is checked against numpy after the timed calls, the empty ones included.
"""

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.benchmark import run_iters

TILE = 16


@iron.jit
def add_one(a: In, b: Out):
    n = a.shape[0]
    tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(n // TILE):
            x = of_in.acquire(1)
            y = of_out.acquire(1)
            for i in range_(TILE):
                y[i] = x[i] + 1
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    ty = np.ndarray[(n,), np.dtype[np.int32]]
    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_a_dispatch_splits_into_its_overhead_and_the_npus_time():
    x = np.arange(4096, dtype=np.int32)
    a = iron.tensor(x, dtype=np.int32)
    b = iron.zeros((4096,), dtype=np.int32)
    result = run_iters(add_one, a, b, warmup=10, iters=20)
    np.testing.assert_array_equal(b.numpy(), x + 1)
    dispatch, overhead, npu = result.dispatch, result.dispatch_overhead, result.npu
    assert dispatch is not None and overhead is not None and npu is not None
    assert dispatch.n == overhead.n == npu.n == 20
    assert 0 < overhead.median_us < dispatch.median_us < result.e2e.median_us
    np.testing.assert_allclose(
        npu.samples_us, np.subtract(dispatch.samples_us, overhead.median_us)
    )
