# vector_scalar_mul/vector_scalar_mul.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Vector scalar multiply — Iron API design with ``@iron.jit`` compilation.

A single AIE compute core scales ``a`` by a runtime scalar ``factor`` to produce
``c = a * factor``.  Default config: 4096-element ``int16`` vector tiled into
four 1024-element sub-vectors.  Driven both as a standalone script (jit + run +
verify) and from the per-sibling ``Makefile`` via ``--xclbin-path`` /
``--insts-path`` compile-only mode.
"""

import argparse
import os
import sys

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
from aie.iron.device import NPU1Col1, NPU2
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime import set_current_device

_KERNELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "aie_kernels", "aie2")
)


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2()


@iron.jit
def vector_scalar_mul(
    A: In,
    F: In,
    C: Out,
    *,
    in1_size: Compile[int],
    int_bit_width: Compile[int] = 16,
    vectorized: Compile[bool] = True,
    trace_size: Compile[int] = 0,
    use_chess: Compile[bool] = False,
):
    in1_dtype = np.int16 if int_bit_width == 16 else np.int32
    tensor_size = in1_size // np.dtype(in1_dtype).itemsize
    num_sub_vectors = 4
    tile_size = tensor_size // num_sub_vectors

    tensor_ty = np.ndarray[(tensor_size,), np.dtype[in1_dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[in1_dtype]]
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    func_type = "vector" if vectorized else "scalar"
    scale = ExternalFunction(
        f"vector_scalar_mul_{func_type}",
        source_file=os.path.join(_KERNELS_DIR, "scale.cc"),
        arg_types=[tile_ty, tile_ty, scalar_ty, np.int32],
        include_dirs=[_KERNELS_DIR],
        compile_flags=[f"-DBIT_WIDTH={int_bit_width}"],
        use_chess=use_chess,
    )

    of_in = ObjectFifo(tile_ty, name="in")
    of_factor = ObjectFifo(scalar_ty, name="infactor")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_factor, of_out, scale_fn):
        elem_factor = of_factor.acquire(1)
        for _ in range_(num_sub_vectors):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            scale_fn(elem_in, elem_out, elem_factor, tile_size)
            of_in.release(1)
            of_out.release(1)
        of_factor.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_factor.cons(), of_out.prod(), scale],
        trace=(1 if trace_size > 0 else 0),
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, scalar_ty, tensor_ty) as (a_in, f_in, c_out):
        if trace_size > 0:
            rt.enable_trace(trace_size)
        rt.start(worker)
        rt.fill(of_in.prod(), a_in)
        rt.fill(of_factor.prod(), f_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Scalar Multiplication")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-i1s", "--in1_size", type=int, default=8192, help="bytes")
    p.add_argument("-i2s", "--in2_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument("-os", "--out_size", type=int, default=8192, help="bytes (== in1_size)")
    p.add_argument("-bw", "--int_bit_width", type=int, default=16, choices=[16, 32])
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument("-t", "--trace_size", type=int, default=0)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument("-w", "--warmup", type=int, default=2)
    p.add_argument("-i", "--iters", type=int, default=5)
    return p


def _validate(opts):
    if opts.in1_size % 128 != 0 or opts.in1_size < 1024:
        sys.exit(
            "in1_size must be a multiple of 128 (len multiple of 64) and >= 1024"
        )
    if opts.in2_size != 4:
        sys.exit("in2_size must be 4 (1 x int32 scalar)")
    if opts.out_size != opts.in1_size:
        sys.exit("out_size must equal in1_size")


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_scalar_mul.specialize(
        in1_size=opts.in1_size,
        int_bit_width=opts.int_bit_width,
        trace_size=opts.trace_size,
        use_chess=bool(opts.use_chess),
    )
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    in1_dtype = np.int16 if opts.int_bit_width == 16 else np.int32
    tensor_size = opts.in1_size // np.dtype(in1_dtype).itemsize

    rng = np.random.default_rng(0)
    a_np = rng.integers(0, 100, size=(tensor_size,), dtype=in1_dtype)
    f_np = np.array([3], dtype=np.int32)
    c_np = np.zeros((tensor_size,), dtype=in1_dtype)

    a_t = iron.tensor(a_np, dtype=in1_dtype, device="npu")
    f_t = iron.tensor(f_np, dtype=np.int32, device="npu")
    c_t = iron.tensor(c_np, dtype=in1_dtype, device="npu")

    bench = run_iters(
        vector_scalar_mul,
        a_t,
        f_t,
        c_t,
        in1_size=opts.in1_size,
        int_bit_width=opts.int_bit_width,
        use_chess=bool(opts.use_chess),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected = (a_np.astype(np.int64) * 3).astype(in1_dtype)
    actual = c_t.numpy()
    if not np.array_equal(actual, expected):
        sys.exit("FAIL! output does not match a * factor")

    print()
    print_benchmark(bench)
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    _validate(opts)
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
