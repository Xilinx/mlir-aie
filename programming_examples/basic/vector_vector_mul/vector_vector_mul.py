# vector_vector_mul/vector_vector_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector-vector multiply — IRON API design with ``@iron.jit`` compilation.

Two int32 vectors are multiplied element-wise on a single AIE compute tile,
in tile-of-16 sub-vectors fed via three depth-2 ObjectFifos (two in, one out).

Driven via the standard 3-mode CLI:

* default          — JIT-compile + run on the attached NPU + verify.
* ``--xclbin-path`` / ``--insts-path`` — compile-only, used by the
  Makefile so a C++ testbench can drive the design.
* ``--emit-mlir``  — print MLIR to stdout for the selected ``--dev``
  (e.g. ``-d xcvc1902 --emit-mlir`` for the VCK5000 toolchain).
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import device_from_args
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass


@iron.jit
def vector_vector_mul(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: Compile[int],
    dtype: Compile[type] = np.int32,
):
    n = 16
    if num_elements % n != 0:
        raise ValueError(f"num_elements ({num_elements}) must be a multiple of {n}")
    n_tiles = num_elements // n

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(n,), np.dtype[dtype]]

    of_in1 = ObjectFifo(tile_ty, name="in1")
    of_in2 = ObjectFifo(tile_ty, name="in2")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in1, of_in2, of_out):
        for _ in range_(n_tiles):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(n):
                elem_out[i] = elem_in1[i] * elem_in2[i]
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in1.cons(), of_in2.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(worker)
        rt.fill(of_in1.prod(), A)
        rt.fill(of_in2.prod(), B)
        rt.drain(of_out.cons(), C, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _compile_kwargs(opts):
    return dict(num_elements=opts.num_elements, dtype=np.int32)


def _run_and_verify(opts):
    input0 = iron.randint(0, 100, (opts.num_elements,), dtype=np.int32, device="npu")
    input1 = iron.randint(0, 100, (opts.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    bench = run_iters(
        vector_vector_mul,
        input0,
        input1,
        output,
        num_elements=opts.num_elements,
        dtype=input0.dtype,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected = input0.numpy() * input1.numpy()
    assert_pass(
        expected,
        output.numpy(),
        fail_msg="output does not match a * b",
        print_pass=False,
    )

    print()
    print_benchmark(bench)
    print("PASS!")


def main():
    p = argparse.ArgumentParser(prog="AIE Vector-Vector Multiply")
    add_compile_args(p, dev_choices=("npu", "npu2", "xcvc1902"), with_emit_mlir=True)
    add_benchmark_args(p)
    p.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=256,
        help="Total elements per input vector (must be a multiple of 16).",
    )
    opts = p.parse_args()
    run_design_cli(
        vector_vector_mul,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: device_from_args(o, n_cols=1),
    )


if __name__ == "__main__":
    main()
