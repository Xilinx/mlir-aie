# section-4/section-4b/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-4b trace example — ``@iron.jit`` (vector * scalar).

Same vector*scalar design as section-3/4a; this version exposes a
``trace_size`` knob and calls ``rt.enable_trace(trace_size,
workers=[my_worker])`` to capture core-tile trace events into a DDR
buffer.  See README.md for a walk-through of the customizable
``coretile_events`` / ``coremem_events`` / ``memtile_events`` /
``shimtile_events`` keyword arguments and the ``PortEvent`` /
``MemTilePortEvent`` / ``ShimTilePortEvent`` classes.

Three modes (same as section-3/4a):

* ``python3 vector_scalar_mul.py [-t N]`` — JIT-compile + run on the
  attached NPU; with ``-t N`` (bytes) trace is enabled.
* ``python3 vector_scalar_mul.py --emit-mlir`` — print the lowered MLIR.
* ``python3 vector_scalar_mul.py --xclbin-path X --insts-path Y`` —
  compile-only, drop the xclbin/insts pair for the explicit-XRT
  walkthrough.
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
from aie.iron.device import device_from_args
from aie.utils.hostruntime.argparse import add_compile_args, add_trace_arg
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

tensor_size = 4096
tile_size = 1024

tensor_ty = np.ndarray[(tensor_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

_KERNEL_SRC = Path(__file__).resolve().parent / "vector_scalar_mul.cc"


@iron.jit
def vector_scalar_mul(
    a_in: In,
    f_in: In,
    c_out: Out,
    *,
    trace_size: Compile[int] = 0,
):
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
        if trace_size > 0:
            rt.enable_trace(trace_size, workers=[my_worker])
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
    vector_scalar_mul(a_in, f_in, c_out, trace_size=opts.trace_size)
    assert_pass(
        c_out.numpy(),
        a_in.numpy() * f_in.numpy()[0],
        fail_msg="vector_scalar_mul output mismatch",
    )


def _emit_mlir(opts):
    a_in = iron.zeros(tensor_size, dtype=np.int32, device="npu")
    f_in = iron.zeros(1, dtype=np.int32, device="npu")
    c_out = iron.zeros(tensor_size, dtype=np.int32, device="npu")
    print(vector_scalar_mul.as_mlir(a_in, f_in, c_out, trace_size=opts.trace_size))


def main():
    p = argparse.ArgumentParser(prog="Section 4b trace example")
    add_compile_args(p, with_emit_mlir=True)
    add_trace_arg(p)
    opts = p.parse_args()
    run_design_cli(
        vector_scalar_mul,
        opts,
        compile_kwargs=lambda o: dict(trace_size=o.trace_size),
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
