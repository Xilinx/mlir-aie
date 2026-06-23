# row_wise_bias_add/row_wise_bias_add.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Row-wise bias add — IRON API design with ``@iron.jit`` compilation.

The C++ kernel (``kernel.cc``) adds a per-column bias vector (``1 x N``)
to every row of an ``M x N`` ``float32`` matrix.  Tiling is ``(m, n)``;
the kernel is parameterized at compile time on ``DIM_m`` / ``DIM_n``
(passed via ``ExternalFunction.compile_flags``), so each specialization
gets its own ``.o`` named with a content hash — no separate Makefile
``build/kernel.o`` step.

Two invocation modes:

  * standalone:   ``python3 row_wise_bias_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
"""

import argparse
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import device_from_args
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass

_KERNEL_SRC = str(Path(__file__).parent / "kernel.cc")


@iron.jit
def row_wise_bias_add(
    inp: In,
    bias: In,
    out: Out,
    *,
    M: CompileTime[int] = 768,
    N: CompileTime[int] = 2304,
    m: CompileTime[int] = 96,
    n: CompileTime[int] = 32,
):
    assert M % m == 0
    assert N % n == 0

    tensor_ty = np.ndarray[(m * n,), np.dtype[np.float32]]
    bias_ty = np.ndarray[(n,), np.dtype[np.float32]]
    in_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    bias_full_ty = np.ndarray[(N,), np.dtype[np.float32]]

    kernel_func = ExternalFunction(
        "row_wise_bias_add_f32_f32",
        source_file=_KERNEL_SRC,
        arg_types=[tensor_ty, bias_ty, tensor_ty],
        compile_flags=[f"-DDIM_m={m}", f"-DDIM_n={n}"],
    )

    in_fifo = ObjectFifo(tensor_ty, name="in_fifo")
    bias_fifo = ObjectFifo(bias_ty, name="bias_fifo")
    out_fifo = ObjectFifo(tensor_ty, name="out_fifo")

    def core_fn(in_fifo, bias_fifo, out_fifo, kernel_func):
        for _ in range_(N // n):
            elem_bias = bias_fifo.acquire(1)
            for _ in range_(M // m):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                kernel_func(elem_in, elem_bias, elem_out)
                out_fifo.release(1)
                in_fifo.release(1)
            bias_fifo.release(1)

    worker = Worker(
        core_fn,
        fn_args=[in_fifo.cons(), bias_fifo.cons(), out_fifo.prod(), kernel_func],
    )

    tap = TensorTiler2D.group_tiler(
        (M, N), (m, n), (M // m, N // n), tile_group_col_major=True
    )[0]
    bias_tap = TensorTiler2D.group_tiler((1, N), (1, n), (1, N // n))[0]

    rt = Runtime()
    with rt.sequence(in_ty, bias_full_ty, in_ty) as (a, b, c):
        rt.start(worker)
        rt.fill(in_fifo.prod(), a, tap)
        rt.fill(bias_fifo.prod(), b, bias_tap)
        rt.drain(out_fifo.cons(), c, tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Row-Wise Bias Add")
    add_compile_args(p)
    p.add_argument("-M", "--M", type=int, default=768)
    p.add_argument("-N", "--N", type=int, default=2304)
    p.add_argument("-m", "--m", type=int, default=96)
    p.add_argument("-n", "--n", type=int, default=32)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, N=opts.N, m=opts.m, n=opts.n)


def _run_and_verify(opts):
    in_t = iron.arange(
        opts.M * opts.N, shape=(opts.M, opts.N), dtype=np.float32, device="npu"
    )
    bias_np = 3 * np.arange(opts.N, dtype=np.float32)
    bias_t = iron.tensor(bias_np, dtype=np.float32, device="npu")
    out_t = iron.zeros_like(in_t)

    row_wise_bias_add(in_t, bias_t, out_t, **_compile_kwargs(opts))

    expected = in_t.numpy() + bias_np[None, :]
    actual = out_t.numpy().reshape(in_t.shape)
    assert_pass(actual, expected, fail_msg="output does not match in + bias (per-row)")


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        row_wise_bias_add,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=device_from_args,
    )


if __name__ == "__main__":
    main()
