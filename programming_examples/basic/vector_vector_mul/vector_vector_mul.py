# vector_vector_mul/vector_vector_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector-vector multiply — Iron API design with ``@iron.jit`` compilation.

Two int32 vectors are multiplied element-wise on a single AIE compute tile,
in tile-of-16 sub-vectors fed via three depth-2 ObjectFifos (two in, one out).

This script has two modes:

* default  — JIT-compiles the design and runs it on the attached NPU,
  then verifies the result against ``a * b`` computed on the host.
* ``--emit-mlir-vck5000`` — emits MLIR for the VCK5000 (XCVC1902) toolchain
  to consume via ``aiecc``; the design body is shared between the two paths.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import XCVC1902
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.verify import assert_pass


def _build_design(dev, num_elements, dtype):
    """Build the vector-vector-multiply IRON design and resolve to MLIR.

    Shared by the JIT path (NPU) and the ``--emit-mlir-vck5000`` path.
    """
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
        # Worker wraps this body in `while True` by default (while_true=True),
        # so the inner loop just iterates over one full vector's worth of tiles.
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

    return Program(dev, rt).resolve_program()


@iron.jit
def vector_vector_mul(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: Compile[int],
    dtype: Compile[type],
):
    return _build_design(iron.get_current_device(), num_elements, dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emit-mlir-vck5000",
        action="store_true",
        help=(
            "Emit MLIR targeting the VCK5000 (XCVC1902) to stdout instead of "
            "JIT-running on the NPU. Consumed by the Makefile's `vck5000` target."
        ),
    )
    p.add_argument(
        "-n",
        "--num-elements",
        type=int,
        default=256,
        help="Total elements per input vector (must be a multiple of 16).",
    )
    p.add_argument("-w", "--warmup", type=int, default=10)
    p.add_argument("-i", "--iters", type=int, default=20)
    opts = p.parse_args()

    if opts.emit_mlir_vck5000:
        # VCK5000 toolchain consumes the printed MLIR via aiecc.
        print(_build_design(XCVC1902(), opts.num_elements, np.int32))
        return

    # NPU JIT path: build random inputs on the device, run, verify on host.
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
    assert_pass(expected, output.numpy(), fail_msg="output does not match a * b", print_pass=False)

    print()
    print_benchmark(bench)
    print("PASS!")


if __name__ == "__main__":
    main()
