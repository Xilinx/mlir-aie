# test_kernels_e2e.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""Starter end-to-end tests for the IRON kernel library factories.

``test/python/test_kernels.py`` is purely declarative: it pins each factory's
returned ``ExternalFunction`` (name, arg types, shapes, error paths) without
ever invoking ``aiecc``.  That catches API regressions but cannot catch
mismatches between the factory's metadata and the actual C++ kernel — e.g.
wrong exported function name, wrong compile flag, or DMA-alignment issues
like the bfloat16 ``reduce_max`` output that this test pins.

These tests compile + run each covered factory and verify output.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_

# ---------------------------------------------------------------------------
# kernels.passthrough — already exercised by passthrough_kernel.py / 00_memcpy.
# This test pins the int32 path as the canonical "factory works end to end" check.
# ---------------------------------------------------------------------------


def test_passthrough_int32_e2e():
    """Compile + run kernels.passthrough(dtype=int32) and verify output == input."""
    LINE = 1024
    N = 4 * LINE  # 4 LINE-sized chunks

    @iron.jit
    def passthrough_design(x: In, y: Out, *, n: CompileTime[int]):
        line_ty = np.ndarray[(LINE,), np.dtype[np.int32]]
        vec_ty = np.ndarray[(n,), np.dtype[np.int32]]

        of_in = ObjectFifo(line_ty, name="in")
        of_out = ObjectFifo(line_ty, name="out")
        passthrough_fn = kernels.passthrough(tile_size=LINE, dtype=np.int32)

        def core(of_in, of_out, fn):
            for _ in range_(n // LINE):
                ein = of_in.acquire(1)
                eout = of_out.acquire(1)
                fn(ein, eout, LINE)
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core, [of_in.cons(), of_out.prod(), passthrough_fn])

        rt = Runtime()

        def sequence(a, b):
            of_in.prod().fill(a)
            of_out.cons().drain(b, wait=True)

        rt.sequence(sequence, [vec_ty, vec_ty])

        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    x = iron.arange(N, dtype=np.int32, device="npu")
    y = iron.zeros(N, dtype=np.int32, device="npu")
    passthrough_design(x, y, n=N)
    np.testing.assert_array_equal(y.numpy(), x.numpy())


# ---------------------------------------------------------------------------
# kernels.reduce_max(dtype=bfloat16) — pins the alignment fix.
#
# The library's bf16 path produces an out_size of 2 elements (4 bytes) so the
# output tile satisfies the NPU's 4-byte shim-DMA alignment.  Before the fix,
# this code path was unreachable from the kernel library — callers had to
# build an ExternalFunction by hand (see the prior 02_vector_reduce_max).
# ---------------------------------------------------------------------------


def test_reduce_max_bfloat16_output_alignment_e2e():
    """kernels.reduce_max(dtype=bfloat16) must compile, run, and write the right value."""
    TILE = 1024

    @iron.jit
    def reduce_max_design(x: In, y: Out, *, n: CompileTime[int]):
        in_ty = np.ndarray[(n,), np.dtype[bfloat16]]
        tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]

        # Library-decided output shape; verifies the alignment fix is active.
        reduce_fn = kernels.reduce_max(tile_size=TILE, dtype=bfloat16)
        out_ty = reduce_fn.arg_types()[1]
        assert out_ty.__args__[0] == (
            2,
        ), "bfloat16 reduce_max output must be padded to 2 elements for DMA alignment"

        of_in = ObjectFifo(tile_ty, name="in")
        of_out = ObjectFifo(out_ty, name="out")

        def core(of_in, of_out, fn):
            for _ in range_(n // TILE):
                ein = of_in.acquire(1)
                eout = of_out.acquire(1)
                fn(ein, eout, TILE)
                of_in.release(1)
                of_out.release(1)

        worker = Worker(core, [of_in.cons(), of_out.prod(), reduce_fn])

        rt = Runtime()

        def sequence(a, b):
            of_in.prod().fill(a)
            of_out.cons().drain(b, wait=True)

        rt.sequence(sequence, [in_ty, out_ty])

        return Program(
            iron.get_current_device(), rt, workers=[worker]
        ).resolve_program()

    x = iron.arange(TILE, dtype=bfloat16, device="npu")
    y = iron.zeros(2, dtype=bfloat16, device="npu")
    reduce_max_design(x, y, n=TILE)
    # Last call's per-tile reduction is the max of the final tile; for arange
    # input that's the largest element.
    expected = bfloat16(TILE - 1)
    assert y[0] == expected, f"reduce_max wrote {y[0]} but max of input was {expected}"
