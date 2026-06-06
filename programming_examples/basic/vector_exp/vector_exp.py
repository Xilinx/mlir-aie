# vector_exp/vector_exp.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector exp(x) — IRON + ``@iron.jit``, 4 cores, bfloat16.

Demonstrates the IRON kernel library's LUT-backed bf16 exp kernel
(``kernels.bf16_exp``).  Each of 4 cores runs the kernel on its own
1024-element tile; the runtime split/join the work across the cores.
"""

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker, kernels
from aie.iron.controlflow import range_
from aie.utils.verify import count_mismatches

_TILE = 1024  # hard-coded by kernels.bf16_exp's underlying C++ kernel
_N_CORES = 4


@iron.jit
def vector_exp(
    x: In,
    y: Out,
    *,
    N: Compile[int],
):
    n = _TILE
    n_cores = _N_CORES
    tiles = (N // n) // n_cores

    tensor_ty = np.ndarray[(N,), np.dtype[bfloat16]]
    memtile_ty = np.ndarray[(n * n_cores,), np.dtype[bfloat16]]
    tile_ty = np.ndarray[(n,), np.dtype[bfloat16]]

    exp_fn = kernels.bf16_exp(tile_size=n)

    A_fifo = ObjectFifo(memtile_ty, name="inA")
    C_fifo = ObjectFifo(memtile_ty, name="outC")
    a_fifos = A_fifo.cons().split(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )
    c_fifos = C_fifo.prod().join(
        offsets=[n * i for i in range(n_cores)], obj_types=[tile_ty] * n_cores
    )

    def core_fn(a_in, c_out, exp_fn):
        for _ in range_(tiles):
            elem_out = c_out.acquire(1)
            elem_in_a = a_in.acquire(1)
            exp_fn(elem_in_a, elem_out)
            a_in.release(1)
            c_out.release(1)

    workers = [
        Worker(core_fn, fn_args=[a_fifos[i].cons(), c_fifos[i].prod(), exp_fn])
        for i in range(n_cores)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a_in, c_out):
        rt.start(*workers)
        rt.fill(A_fifo.prod(), a_in)
        rt.drain(C_fifo.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def main():
    # Test every possible bfloat16 value by reinterpreting each uint16 as bf16.
    N = 65536
    a_np = np.arange(N, dtype=np.uint16).view(bfloat16)

    a = iron.tensor(a_np, dtype=bfloat16, device="npu")
    c = iron.zeros(N, dtype=bfloat16, device="npu")

    vector_exp(a, c, N=N)

    # The AIE kernel is a LUT approximation; verify with the canonical
    # 12.8% relative tolerance, stopping at the first inf/nan (the LUT's
    # behaviour outside its defined input range isn't part of the contract).
    with np.errstate(over="ignore", invalid="ignore"):
        ref = np.exp(a_np.astype(np.float32))
    errors, n_checked = count_mismatches(c.numpy(), ref)

    if errors:
        print(f"FAIL: {errors} of {n_checked} samples outside 12.8% relative tolerance")
        sys.exit(1)
    print(f"PASS!  ({n_checked} samples verified within 12.8% relative tolerance)")


if __name__ == "__main__":
    main()
