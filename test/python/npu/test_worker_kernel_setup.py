# test_worker_kernel_setup.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.iron.kernels as kernels
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker, jit
from aie.iron.controlflow import range_

TILE = 1024
TILES = 4


@jit
def bf16_mul(a: In, b: In, c: Out):
    mul = kernels.mul(tile_size=TILE)
    tile_ty = np.ndarray[(TILE,), np.dtype[bfloat16]]
    tensor_ty = np.ndarray[(TILE * TILES,), np.dtype[bfloat16]]
    of_a, of_b, of_c = (ObjectFifo(tile_ty, name=n) for n in ("a", "b", "c"))

    def core_fn(of_a, of_b, of_c, mul):
        for _ in range_(TILES):
            ea, eb, ec = of_a.acquire(1), of_b.acquire(1), of_c.acquire(1)
            mul(ea, eb, ec)
            of_a.release(1)
            of_b.release(1)
            of_c.release(1)

    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod(), mul])

    def sequence(A, B, C, a_h, b_h, c_h):
        a_h.fill(A)
        b_h.fill(B)
        c_h.drain(C, wait=True)

    rt = Runtime(
        sequence,
        [tensor_ty, tensor_ty, tensor_ty, of_a.prod(), of_b.prod(), of_c.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def test_worker_runs_the_kernels_rounding_setup():
    """A Worker calling ``kernels.mul`` rounds its bf16 products to nearest even.

    The product of two bf16 values is exact in fp32, so a core in the mode the
    contract's ``setup`` names matches numpy's cast bit for bit; one left in
    its boot mode (floor) truncates about half of them by one ulp.
    """
    rng = np.random.default_rng(0)
    a, b = (rng.uniform(-4, 4, TILE * TILES).astype(bfloat16) for _ in range(2))
    ta = iron.tensor(a, dtype=bfloat16, device="npu")
    tb = iron.tensor(b, dtype=bfloat16, device="npu")
    tc = iron.tensor(np.zeros(TILE * TILES, bfloat16), dtype=bfloat16, device="npu")
    bf16_mul(ta, tb, tc)
    want = (a.astype(np.float32) * b.astype(np.float32)).astype(bfloat16)
    got = tc.numpy()
    differ = np.flatnonzero(got.view(np.uint16) != want.view(np.uint16))
    assert differ.size == 0, (
        f"{differ.size}/{got.size} words differ from the nearest-even product, "
        f"first at {differ[0]}: got {got[differ[0]]}, want {want[differ[0]]}"
    )
