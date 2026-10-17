# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

# Several runtime-sequence transfers may share one ObjectFifo object: the far
# end's buffer descriptor completes once the stream carries a whole object,
# however many transfers it took. 128 + 896 elements straddle the 256-element
# object boundaries and make 4 whole objects together.

import aie.iron as iron
import numpy as np
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime

OBJECT, FIRST, SECOND = 256, 128, 896


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


def _random(size, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-(2**31), 2**31 - 1, size=size, dtype=np.int32)


def test_fills_concatenate_into_shared_objects():
    x, y = _random(FIRST, seed=0), _random(SECOND, seed=1)
    c = iron.zeros((FIRST + SECOND,), dtype=np.int32, device="npu")
    concat(
        iron.tensor(x, device="npu"),
        iron.tensor(y, device="npu"),
        c,
    )
    assert np.array_equal(c.numpy(), np.concatenate([x, y]))


def test_drains_split_shared_objects():
    a = _random(FIRST + SECOND, seed=2)
    c1 = iron.zeros((FIRST,), dtype=np.int32, device="npu")
    c2 = iron.zeros((SECOND,), dtype=np.int32, device="npu")
    split(iron.tensor(a, device="npu"), c1, c2)
    assert np.array_equal(c1.numpy(), a[:FIRST])
    assert np.array_equal(c2.numpy(), a[FIRST:])
