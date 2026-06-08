# section-4/section-4a/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""Section-4a timer example — ``@iron.jit`` (vector * scalar).

Same vector*scalar design as section-3, used here to teach NPU and
end-to-end timing. The default ``_run_and_verify`` wraps the
``@iron.jit`` call in ``aie.utils.benchmark.run_iters`` which reports
``avg/min/max us`` for both the NPU-side timing (from the kernel
result) and end-to-end host time. Override the loop counts with
``--warmup`` / ``--iters``.

Three modes (same as section-3):

* ``python3 vector_scalar_mul.py [--iters N --warmup K]`` — JIT-compile
  + run + verify + benchmark on the attached NPU.
* ``python3 vector_scalar_mul.py --emit-mlir`` — print the lowered MLIR.
* ``python3 vector_scalar_mul.py --xclbin-path X --insts-path Y`` —
  compile-only.
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import (
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import device_from_args
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
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


def _run_and_verify(opts):
    a_in = iron.arange(1, tensor_size + 1, dtype=np.int32, device="npu")
    f_in = iron.full((1,), 3, dtype=np.int32, device="npu")
    c_out = iron.zeros(tensor_size, dtype=np.int32, device="npu")

    bench = run_iters(
        vector_scalar_mul,
        a_in,
        f_in,
        c_out,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    assert_pass(
        c_out.numpy(),
        a_in.numpy() * f_in.numpy()[0],
        fail_msg="vector_scalar_mul output mismatch",
    )

    print()
    print_benchmark(bench)
    print("\nPASS!")


def _emit_mlir(opts):
    a_in = iron.zeros(tensor_size, dtype=np.int32, device="npu")
    f_in = iron.zeros(1, dtype=np.int32, device="npu")
    c_out = iron.zeros(tensor_size, dtype=np.int32, device="npu")
    print(vector_scalar_mul.as_mlir(a_in, f_in, c_out))


def main():
    p = argparse.ArgumentParser(prog="Section 4a timing example")
    add_compile_args(p, with_emit_mlir=True)
    add_benchmark_args(p, default_warmup=4, default_iters=10)
    opts = p.parse_args()
    run_design_cli(
        vector_scalar_mul,
        opts,
        compile_kwargs={},
        run_and_verify=_run_and_verify,
        emit_mlir=_emit_mlir,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
