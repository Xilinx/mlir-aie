# section-3/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-3 first end-to-end program — ``@iron.jit`` (vector * scalar).

Streams a 4096-element int32 vector through a single compute tile in
1024-element tiles, multiplying each element by a per-call scalar
``factor``.  The C++ kernel lives in ``vector_scalar_mul.cc`` and is
auto-built into the JIT cache via ``ExternalFunction(source_file=...)``.

Three modes:

* ``python3 vector_scalar_mul.py`` — JIT-compile + run + verify on the
  attached NPU.
* ``python3 vector_scalar_mul.py --emit-mlir`` — print the lowered MLIR
  (no NPU access).
* ``python3 vector_scalar_mul.py --xclbin-path X --insts-path Y`` —
  compile-only, drop the xclbin/insts pair for the explicit-XRT
  walkthrough (used by ``test.cpp`` and ``test.py``).
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    Compile,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

tensor_size = 4096
tile_size = 1024

tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

_KERNEL_SRC = Path(__file__).resolve().parent / "vector_scalar_mul.cc"


@iron.jit
def vector_scalar_mul(a_in: In, f_in: In, c_out: Out):
    scale_fn = ExternalFunction(
        "vector_scalar_mul_aie_scalar",
        source_file=str(_KERNEL_SRC),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
    )

    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_fn(of_in, of_factor, of_out, scale):
        elem_factor = of_factor.acquire(1)
        for _ in range_(tensor_size // tile_size):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    my_worker = Worker(
        core_fn, [of_in.cons(), of_factor.cons(), of_out.prod(), scale_fn]
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a, f, c):
        rt.start(my_worker)
        rt.fill(of_in.prod(), a)
        rt.fill(of_factor.prod(), f)
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_inputs():
    a_in = iron.arange(1, tensor_size + 1, dtype=np.int32, device="npu")
    f_in = iron.full((1,), 3, dtype=np.int32, device="npu")
    return a_in, f_in


def _run_and_verify(opts):
    a_in, f_in = _make_inputs()
    c_out = iron.zeros(tensor_size, dtype=np.int32, device="npu")
    vector_scalar_mul(a_in, f_in, c_out)
    assert_pass(
        c_out.numpy(),
        a_in.numpy() * f_in.numpy()[0],
        fail_msg="vector_scalar_mul output mismatch",
    )


def main():
    p = argparse.ArgumentParser(prog="Section 3 vector * scalar example")
    add_compile_args(p, with_emit_mlir=True)
    opts = p.parse_args()
    run_design_cli(
        vector_scalar_mul,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
