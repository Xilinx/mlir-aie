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
four 1024-element sub-vectors.  The design body delegates to
``aie.iron.algorithms.transform_typed``, which handles the
ObjectFifo / Worker / Runtime plumbing (including trace) for any
``(input_tile, output_tile, *param_tensors, tile_size)``-shaped
ExternalFunction.

Driven both as a standalone script (jit + run + verify) and from the per-
sibling ``Makefile`` via ``--xclbin-path`` / ``--insts-path`` compile-only
mode.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out, kernels
from aie.iron.algorithms import transform_typed
from aie.iron.device import device_from_args
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import (
    add_benchmark_args,
    add_compile_args,
    add_trace_arg,
)
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


@iron.jit
def vector_scalar_mul(
    A: In,
    C: Out,
    F: In,
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
    scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

    scale = kernels.scale(
        tile_size=tile_size,
        dtype=in1_dtype,
        vectorized=vectorized,
        use_chess=use_chess,
    )

    return transform_typed(
        scale, tensor_ty, scalar_ty, tile_size=tile_size, trace_size=trace_size
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Scalar Multiplication")
    add_compile_args(p)
    p.add_argument("-i1s", "--in1_size", type=int, default=8192, help="bytes")
    p.add_argument("-i2s", "--in2_size", type=int, default=4, help="bytes (always 4)")
    p.add_argument(
        "-os", "--out_size", type=int, default=8192, help="bytes (== in1_size)"
    )
    p.add_argument("-bw", "--int_bit_width", type=int, default=16, choices=[16, 32])
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    add_trace_arg(p)
    add_benchmark_args(p)
    return p


def _validate(opts):
    if opts.in1_size % 128 != 0 or opts.in1_size < 1024:
        sys.exit("in1_size must be a multiple of 128 (len multiple of 64) and >= 1024")
    if opts.in2_size != 4:
        sys.exit("in2_size must be 4 (1 x int32 scalar)")
    if opts.out_size != opts.in1_size:
        sys.exit("out_size must equal in1_size")


def _compile_kwargs(opts):
    return dict(
        in1_size=opts.in1_size,
        int_bit_width=opts.int_bit_width,
        trace_size=opts.trace_size,
        use_chess=bool(opts.use_chess),
    )


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
        c_t,
        f_t,
        in1_size=opts.in1_size,
        int_bit_width=opts.int_bit_width,
        use_chess=bool(opts.use_chess),
        warmup=opts.warmup,
        iters=opts.iters,
    )

    expected = (a_np.astype(np.int64) * 3).astype(in1_dtype)
    actual = c_t.numpy()
    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        fail_msg="output does not match a * factor",
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        vector_scalar_mul,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
        validate=_validate,
    )


if __name__ == "__main__":
    main()
