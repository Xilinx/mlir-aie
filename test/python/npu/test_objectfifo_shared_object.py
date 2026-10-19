# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

# Several runtime-sequence transfers may share one ObjectFifo object: the far
# end's buffer descriptor completes once the stream carries a whole object,
# however many transfers it took. 128 + 896 elements straddle the 256-element
# object boundaries and make 4 whole objects together. A distribute's combined
# end completes per segment, so a fill may end between segments.

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime

OBJECT, FIRST, SECOND, SEGMENT = 256, 128, 896, 64
INT32 = np.iinfo(np.int32)


@iron.jit
def concat(x: In, y: In, c: Out, *, n: CompileTime[int] = OBJECT):
    of_in = ObjectFifo(np.ndarray[(n,), np.dtype[np.int32]], name="in")
    of_out = of_in.cons().forward(name="out")

    def sequence(x_h, y_h, c_h, in_prod, out_cons):
        in_prod.fill(x_h)
        in_prod.fill(y_h)
        out_cons.drain(c_h, wait=True)

    rt = Runtime(sequence, [x, y, c, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def split(a: In, c1: Out, c2: Out, *, n: CompileTime[int] = OBJECT):
    of_in = ObjectFifo(np.ndarray[(n,), np.dtype[np.int32]], name="in")
    of_out = of_in.cons().forward(name="out")

    def sequence(a_h, c1_h, c2_h, in_prod, out_cons):
        in_prod.fill(a_h)
        out_cons.drain(c1_h)
        out_cons.drain(c2_h, wait=True)

    rt = Runtime(sequence, [a, c1, c2, of_in.prod(), of_out.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


@iron.jit
def distribute(a: In, c0: Out, c1: Out, *, n: CompileTime[int] = SEGMENT):
    of_in = ObjectFifo(np.ndarray[(2 * n,), np.dtype[np.int32]], name="in")
    seg_ty = np.ndarray[(n,), np.dtype[np.int32]]
    out0, out1 = of_in.cons().split([0, n], obj_types=[seg_ty, seg_ty])

    def sequence(a_h, c0_h, c1_h, in_prod, out0_cons, out1_cons):
        in_prod.fill(a_h)
        out0_cons.drain(c0_h)
        out1_cons.drain(c1_h, wait=True)

    rt = Runtime(sequence, [a, c0, c1, of_in.prod(), out0.cons(), out1.cons()])
    return Program(iron.get_current_device(), rt).resolve_program()


def test_fills_concatenate_into_shared_objects():
    rng = np.random.default_rng(0)
    x, y = (
        rng.integers(INT32.min, INT32.max, size, np.int32, endpoint=True)
        for size in (FIRST, SECOND)
    )
    c = iron.zeros((FIRST + SECOND,), dtype=np.int32, device="npu")
    concat(
        iron.tensor(x, device="npu"),
        iron.tensor(y, device="npu"),
        c,
    )
    assert np.array_equal(c.numpy(), np.concatenate([x, y]))


def test_drains_split_shared_objects():
    rng = np.random.default_rng(2)
    a = rng.integers(INT32.min, INT32.max, FIRST + SECOND, np.int32, endpoint=True)
    c1 = iron.zeros((FIRST,), dtype=np.int32, device="npu")
    c2 = iron.zeros((SECOND,), dtype=np.int32, device="npu")
    split(iron.tensor(a, device="npu"), c1, c2)
    assert np.array_equal(c1.numpy(), a[:FIRST])
    assert np.array_equal(c2.numpy(), a[FIRST:])


def test_fill_ends_on_a_distribute_segment():
    rng = np.random.default_rng(3)
    a = rng.integers(INT32.min, INT32.max, 3 * SEGMENT, np.int32, endpoint=True)
    segments = a.reshape(-1, SEGMENT)
    c0 = iron.zeros((2 * SEGMENT,), dtype=np.int32, device="npu")
    c1 = iron.zeros((SEGMENT,), dtype=np.int32, device="npu")
    distribute(iron.tensor(a, device="npu"), c0, c1)
    assert np.array_equal(c0.numpy(), segments[0::2].reshape(-1))
    assert np.array_equal(c1.numpy(), segments[1])
