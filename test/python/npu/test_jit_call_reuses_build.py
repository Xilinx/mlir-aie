# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings
"""A design called again reuses the build its first call made."""

import numpy as np

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime
from aie.utils import get_current_device
from aie.utils.compile import NPU_CACHE_HOME

N = 32


class Sizes:
    @staticmethod
    def copy_twice():
        return N


@iron.jit
def copy_twice(a: In, b: Out):
    # The attribute shares the design's name, so the key walk reads the design.
    ty = np.ndarray[(Sizes.copy_twice(),), np.dtype[np.int32]]
    of_in = ObjectFifo(ty)
    of_out = of_in.cons().forward()

    def sequence(a, b, a_in, b_out):
        a_in.fill(a)
        b_out.drain(b, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(get_current_device(), rt).resolve_program()


def test_a_second_call_reuses_the_first_calls_build():
    want = np.arange(N, dtype=np.int32)
    a = iron.tensor(want)
    builds = []
    for _ in range(2):
        b = iron.tensor(np.zeros(N, dtype=np.int32))
        copy_twice(a, b)
        np.testing.assert_array_equal(b.numpy(), want)
        builds.append({p.name for p in NPU_CACHE_HOME.iterdir() if p.is_dir()})
    assert builds[1] == builds[0]
