# dma_transpose/dma_transpose.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""DMA transpose — Iron API design with ``@iron.jit`` compilation.

Reads an ``M x K`` int32 matrix from host memory in column-major tile order
(via ``TensorTiler2D.simple_tiler(..., tile_col_major=True)``) and writes it
back row-major, producing the transpose with no compute on the AIE core — the
shim DMA does all the work via per-stream stride/wrap configuration.

Driven both as a standalone script (jit + run + verify) and from the per-
sibling ``Makefile`` via ``--xclbin-path`` / ``--insts-path`` compile-only
mode.
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2Col1, AnyComputeTile
from aie.helpers.taplib import TensorTiler2D
from aie.utils.benchmark import print_benchmark, run_iters
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    return NPU1Col1() if dev_str == "npu" else NPU2Col1()


@iron.jit
def dma_transpose(
    A: In,
    B: In,
    C: Out,
    *,
    M: Compile[int],
    K: Compile[int],
):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    tap_in = TensorTiler2D.simple_tiler((M, K), tile_col_major=True)[0]

    of_in = ObjectFifo(tensor_ty)
    of_out = of_in.cons().forward(AnyComputeTile)

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _b_unused, c_out):
        rt.fill(of_in.prod(), a_in, tap_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE DMA Transpose")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", type=int, default=64)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument("-w", "--warmup", type=int, default=2)
    p.add_argument("-i", "--iters", type=int, default=5)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="Produce a file (transpose_data.png) showing data access order",
    )
    return p


def _validate(opts):
    if (opts.M * opts.K) % 1024 != 0:
        sys.exit("M * K must be a multiple of 1024 (test.cpp host constraint)")


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = dma_transpose.specialize(M=opts.M, K=opts.K)
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _generate_access_map(opts):
    tap_in = TensorTiler2D.simple_tiler((opts.M, opts.K), tile_col_major=True)[0]
    tap_in.visualize(file_path="transpose_data.png", show_tile=False)


def _run_and_verify(opts):
    rng = np.random.default_rng(0)
    a_np = rng.integers(-1_000_000, 1_000_000, size=(opts.M, opts.K), dtype=np.int32)
    b_np = np.zeros((opts.M, opts.K), dtype=np.int32)  # unused 2nd buffer
    c_np = np.zeros((opts.M, opts.K), dtype=np.int32)

    a_t = iron.tensor(a_np.reshape(-1), dtype=np.int32, device="npu")
    b_t = iron.tensor(b_np.reshape(-1), dtype=np.int32, device="npu")
    c_t = iron.tensor(c_np.reshape(-1), dtype=np.int32, device="npu")

    bench = run_iters(
        dma_transpose,
        a_t,
        b_t,
        c_t,
        M=opts.M,
        K=opts.K,
        warmup=opts.warmup,
        iters=opts.iters,
    )

    # Column-major read of an (M,K) matrix produces an (M,K) buffer storing
    # A^T flattened row-major in C — i.e. C reshaped to (K,M) equals A.T.
    expected = a_np.T.reshape(-1)
    actual = c_t.numpy()
    if not np.array_equal(actual, expected):
        sys.exit("FAIL! output does not match A.T")

    print()
    print_benchmark(bench)
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.generate_access_map:
        _generate_access_map(opts)
        return
    _validate(opts)
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
