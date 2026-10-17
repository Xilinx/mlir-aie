# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""The exact (erf) GELU over every bf16 input, held to its contract.

``gelu_sized(exact=True)`` runs ``gelu_erf.cc`` on each of the 65536 bf16
bit patterns, 4096 a call, and every output is judged against
``gelu_erf_ref`` by the factory's own tolerance.
"""

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, jit, kernels
from aie.iron.controlflow import range_
from aie.utils.verify import compare

_TILE, _CALLS = 4096, 16


@jit
def gelu_erf_all(x_in: In, y_out: Out):
    """``_CALLS`` calls of the exact GELU, one tile each."""
    tile_ty = np.ndarray[(_TILE,), np.dtype[bfloat16]]
    gelu = kernels.gelu_sized(_TILE, exact=True)
    of_x = ObjectFifo(tile_ty, name="x", depth=2)
    of_y = ObjectFifo(tile_ty, name="y", depth=2)

    def core(of_x, of_y, gelu):
        for _ in range_(_CALLS):
            gelu(of_x.acquire(1), of_y.acquire(1), _TILE)
            of_x.release(1)
            of_y.release(1)

    worker = Worker(core, fn_args=[of_x.cons(), of_y.prod(), gelu])
    host_ty = np.ndarray[(_TILE * _CALLS,), np.dtype[bfloat16]]

    def sequence(x_h, y_h, xf, yf):
        xf.fill(x_h)
        yf.drain(y_h, wait=True)

    rt = Runtime(sequence, [host_ty, host_ty, of_x.prod(), of_y.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_gelu_erf_matches_erf_gelu_at_every_bf16():
    x = np.arange(1 << 16, dtype=np.uint16).view(bfloat16)
    y_t = iron.zeros((x.size,), dtype=bfloat16)
    gelu_erf_all(iron.tensor(x, dtype=bfloat16), y_t)
    got = y_t.numpy()
    want = kernels.gelu_erf_ref(x)
    tolerance = kernels.gelu_sized(_TILE, exact=True).contract.tolerance
    verdict = compare(got, want, tolerance)
    assert verdict, verdict.detail
