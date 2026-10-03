# row_wise_bias_add/row_wise_bias_add.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Row-wise bias add / affine cast — IRON API designs with ``@iron.jit``.

The C++ kernel (``kernel.cc``) adds a per-column bias vector (``1 x N``)
to every row of an ``M x N`` ``float32`` matrix.  Tiling is ``(m, n)``;
the kernel is parameterized at compile time on ``DIM_m`` / ``DIM_n``
(passed via ``ExternalFunction.compile_flags``), so each specialization
gets its own ``.o`` named with a content hash — no separate Makefile
``build/kernel.o`` step.

``--op affine_cast`` selects a second design in the same kernel file:
a per-column affine transform (``out = in*gamma + beta``) narrowed to
``bfloat16``. gamma and beta are packed into one ``[2*n]`` buffer per
column-block (see ``kernel.cc``), since an AIE2 tile has only two input
DMA channels and ``in`` already takes one.

Two invocation modes:

  * standalone:   ``python3 row_wise_bias_add.py``
  * compile-only: ``... --xclbin-path=PATH --insts-path=PATH``  (NPU Makefile)
"""

import argparse
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorTiler2D
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_pass
from ml_dtypes import bfloat16

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

    def sequence(a, b, c, in_h, bias_h, out_h):
        in_h.fill(a, tap)
        bias_h.fill(b, bias_tap)
        out_h.drain(c, tap, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, bias_full_ty, in_ty, in_fifo.prod(), bias_fifo.prod(), out_fifo.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


@iron.jit
def row_wise_affine_cast(
    inp: In,
    gb: In,
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
    out_tile_ty = np.ndarray[(m * n,), np.dtype[bfloat16]]
    gb_tile_ty = np.ndarray[(2 * n,), np.dtype[np.float32]]
    in_ty = np.ndarray[(M * N,), np.dtype[np.float32]]
    out_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]
    gb_full_ty = np.ndarray[(2 * N,), np.dtype[np.float32]]

    kernel_func = ExternalFunction(
        "row_wise_affine_cast_f32_bf16",
        source_file=_KERNEL_SRC,
        arg_types=[tensor_ty, gb_tile_ty, out_tile_ty],
        compile_flags=[f"-DDIM_m={m}", f"-DDIM_n={n}"],
    )

    in_fifo = ObjectFifo(tensor_ty, name="in_fifo")
    gb_fifo = ObjectFifo(gb_tile_ty, name="gb_fifo")
    out_fifo = ObjectFifo(out_tile_ty, name="out_fifo")

    def core_fn(in_fifo, gb_fifo, out_fifo, kernel_func):
        for _ in range_(N // n):
            elem_gb = gb_fifo.acquire(1)
            for _ in range_(M // m):
                elem_in = in_fifo.acquire(1)
                elem_out = out_fifo.acquire(1)
                kernel_func(elem_in, elem_gb, elem_out)
                out_fifo.release(1)
                in_fifo.release(1)
            gb_fifo.release(1)

    worker = Worker(
        core_fn,
        fn_args=[in_fifo.cons(), gb_fifo.cons(), out_fifo.prod(), kernel_func],
    )

    tap = TensorTiler2D.group_tiler(
        (M, N), (m, n), (M // m, N // n), tile_group_col_major=True
    )[0]
    # gamma/beta ride as ONE [1, 2N] array, block-interleaved: for column-block
    # j, elements [j*2n : j*2n+n) are gamma[jn:(j+1)n) and [j*2n+n : (j+1)*2n)
    # are beta[jn:(j+1)n) — see _run_and_verify_affine_cast for the host pack.
    gb_tap = TensorTiler2D.group_tiler((1, 2 * N), (1, 2 * n), (1, N // n))[0]

    def sequence(a, b, c, in_h, gb_h, out_h):
        in_h.fill(a, tap)
        gb_h.fill(b, gb_tap)
        out_h.drain(c, tap, wait=True)

    rt = Runtime(
        sequence,
        [in_ty, gb_full_ty, out_ty, in_fifo.prod(), gb_fifo.prod(), out_fifo.cons()],
    )

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Row-Wise Bias Add")
    add_compile_args(p)
    p.add_argument(
        "--op",
        choices=["bias_add", "affine_cast"],
        default="bias_add",
        help="bias_add: out = in + bias (f32). affine_cast: out = "
        "bfloat16(in*gamma + beta)",
    )
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


def _run_and_verify_affine_cast(opts):
    rng = np.random.default_rng(0)
    in_np = rng.uniform(-4.0, 4.0, size=(opts.M, opts.N)).astype(np.float32)
    gamma_np = rng.uniform(0.5, 2.0, size=(opts.N,)).astype(np.float32)
    beta_np = rng.uniform(-1.0, 1.0, size=(opts.N,)).astype(np.float32)

    # Block-interleave to match gb_tap's [1, 2N] tiling: column-block j gets
    # gamma[jn:(j+1)n) then beta[jn:(j+1)n) as one contiguous 2n run.
    n = opts.n
    gb_np = np.empty(2 * opts.N, dtype=np.float32)
    for j in range(opts.N // n):
        gb_np[2 * j * n : 2 * j * n + n] = gamma_np[j * n : (j + 1) * n]
        gb_np[2 * j * n + n : 2 * (j + 1) * n] = beta_np[j * n : (j + 1) * n]

    in_t = iron.tensor(in_np, dtype=np.float32, device="npu")
    gb_t = iron.tensor(gb_np, dtype=np.float32, device="npu")
    out_t = iron.zeros(opts.M * opts.N, dtype=bfloat16, device="npu")

    row_wise_affine_cast(in_t, gb_t, out_t, **_compile_kwargs(opts))

    # ml_dtypes rounds to nearest even, matching the kernel's conv_even.
    expected = (in_np * gamma_np[None, :] + beta_np[None, :]).astype(bfloat16)
    actual = out_t.numpy().reshape(opts.M, opts.N)
    assert_pass(actual, expected, fail_msg="affine_cast output mismatch")


def main():
    opts = _make_argparser().parse_args()
    if opts.op == "affine_cast":
        run_design_cli(
            row_wise_affine_cast,
            opts,
            compile_kwargs=_compile_kwargs,
            run_and_verify=_run_and_verify_affine_cast,
            device=device_from_args,
        )
    else:
        run_design_cli(
            row_wise_bias_add,
            opts,
            compile_kwargs=_compile_kwargs,
            run_and_verify=_run_and_verify,
            device=device_from_args,
        )


if __name__ == "__main__":
    main()
