#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Dynamic single-core matrix multiply — IRON ``@iron.jit`` design.

A minimal delta over ``../single_core/single_core.py``: the problem size
``(M, K, N)`` is supplied to the runtime sequence as SSA ``i32`` values rather
than baked in at compile time, so one compiled design serves any
multiple-of-tile shape at runtime.

Two things change versus the static design:

1. The runtime sequence takes ``M, K, N`` as scalar arguments and derives its
   loop trip counts / DMA geometry from them with ``range_`` / ``if_`` (which
   emit ``scf.for`` / ``scf.if``) and arithmetic on the SSA values.
2. The core reads its loop trip counts from an RTP :class:`Buffer` the host
   populates at the start of the sequence, so one fixed ELF runs any size.

Because the sequence sizes are runtime values, the NPU program can't be frozen
to a static ``insts.bin``; the design is driven through the aiecc TXN-C++ flow
(see ``single_core_dynamic_txn.py``), which emits a ``generate_txn_sequence``
host function that rebuilds the instruction stream for each ``(M, K, N)``.
"""

import argparse

import numpy as np

import aie.iron as iron
from aie.iron import (
    Buffer,
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.dialects import arith
from aie.extras.dialects.arith import constant
from aie.extras import types as T
from aie.helpers.dialects.scf import if_
from aie.helpers.taplib import TensorTiler2D
from aie.utils.hostruntime.argparse import add_compile_args


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def single_core_dynamic(
    A: In,
    B: In,
    C: Out,
    *,
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    max_m: CompileTime[int] = 4096,
    max_k: CompileTime[int] = 4096,
    max_n: CompileTime[int] = 4096,
):
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), "input and output dtypes must both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), "output dtype must be equal or larger to input dtype"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    # Host buffer types are sized to the compiled-in maximum; the runtime only
    # streams the (M, K, N) sub-region requested at call time.
    A_ty = np.ndarray[(max_m * max_k,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(max_k * max_n,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(max_m * max_n,), np.dtype[dtype_out]]
    a_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    b_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    c_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    inA = ObjectFifo(a_ty, name="inA")
    a_dims = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    memA = inA.cons().forward(name="memA", dims_to_stream=a_dims)

    inB = ObjectFifo(b_ty, name="inB")
    if b_col_maj:
        b_dims = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    else:
        b_dims = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    memB = inB.cons().forward(name="memB", dims_to_stream=b_dims)

    memC = ObjectFifo(c_ty, name="memC")
    c_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    outC = memC.cons().forward(name="outC", dims_to_stream=c_dims)

    # RTP buffer the host writes loop trip counts into:
    #   rtp[0] = K_div_k  (inner-K accumulation count)
    #   rtp[1] = tiles    (outer output-tile count = M_div_m * N_div_n)
    rtp = Buffer(
        np.ndarray[(2,), np.dtype[np.int32]],
        name="rtp",
        use_write_rtp=True,
    )

    def core_fn(of_a, of_b, of_c, my_rtp, zero, matmul):
        # Trip counts come from RTP, not Python ints, so the compiled core runs
        # any (M, K, N) the host writes. The outer loop runs to the largest
        # supported tile count; the host-written `tiles` bounds the real work.
        K_div_k = my_rtp[0]
        tiles = my_rtp[1]
        for _ in range_(tiles):
            elem_out = of_c.acquire(1)
            zero(elem_out)
            for _ in range_(K_div_k):
                elem_in_a = of_a.acquire(1)
                elem_in_b = of_b.acquire(1)
                matmul(elem_in_a, elem_in_b, elem_out)
                of_a.release(1)
                of_b.release(1)
            of_c.release(1)

    worker = Worker(
        core_fn,
        [memA.cons(), memB.cons(), memC.prod(), rtp, zero_kernel, matmul_kernel],
        stack_size=0xD00,
        dynamic_objfifo_lowering=True,
    )

    rows_per_block = 4

    rt = Runtime()

    def sequence(A, B, C, M, K, N):
        M_div_m = M // m
        K_div_k = K // k
        N_div_n = N // n
        tiles = M_div_m * N_div_n

        rtp[0] = K_div_k
        rtp[1] = tiles

        prev = None
        for tile_row_block in range_(iron.ceildiv(max_m // m, rows_per_block)):
            for pingpong in [0, 1]:
                row_base = (
                    tile_row_block * rows_per_block + pingpong * rows_per_block // 2
                )
                C_row_offset = row_base * m * N
                # Clamp the rows handled this half-block to what remains.
                num_tile_rows = arith.minsi(
                    constant(rows_per_block // 2, T.i32()),
                    M_div_m - row_base,
                )
                with if_(num_tile_rows > 0, hasElse=False):
                    tg = TaskGroup()
                    outC.cons().drain(
                        C,
                        offset=C_row_offset,
                        sizes=[num_tile_rows, N_div_n, m, n],
                        strides=[m * N, n, N, 1],
                        group=tg,
                        wait=True,
                    )
                    for tile_row in range(rows_per_block // 2):
                        A_row_offset = (row_base + tile_row) * m * K

                        def emit_ab():
                            inA.prod().fill(
                                A,
                                offset=A_row_offset,
                                sizes=[N_div_n, K_div_k, m, k],
                                strides=[0, k, K, 1],
                                group=tg,
                            )
                            if not b_col_maj:
                                b_sizes = [N_div_n, K_div_k, k, n]
                                b_strides = [n, k * N, N, 1]
                            else:
                                b_sizes = [N_div_n, K_div_k, n, k]
                                b_strides = [n * K, k, K, 1]
                            inB.prod().fill(
                                B,
                                sizes=b_sizes,
                                strides=b_strides,
                                group=tg,
                            )

                        if tile_row == 0:
                            emit_ab()
                        else:
                            with if_(num_tile_rows > tile_row, hasElse=False):
                                emit_ab()

                    if prev is not None:
                        prev.resolve()
                    prev = tg
        if prev is not None:
            prev.resolve()

    rt.sequence(sequence, [A_ty, B_ty, C_ty, T.i32, T.i32, T.i32])

    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication (Single Core, Dynamic)",
        description=(
            "Single-core GEMM whose (M, K, N) are runtime SSA values, so one "
            "compiled design serves any multiple-of-tile shape."
        ),
    )
    add_compile_args(p, short_dev=None)
    p.add_argument("-m", type=int, default=32)
    p.add_argument("-k", type=int, default=32)
    p.add_argument("-n", type=int, default=32)
    p.add_argument(
        "--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="bf16"
    )
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="f32",
    )
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument("--max-m", type=int, default=4096)
    p.add_argument("--max-k", type=int, default=4096)
    p.add_argument("--max-n", type=int, default=4096)
    return p


def main():
    opts = _make_argparser().parse_args()
    # Emit MLIR for the TXN flow (see single_core_dynamic_txn.py to compile it).
    mlir_module = single_core_dynamic.as_mlir(
        None,
        None,
        None,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        max_m=opts.max_m,
        max_k=opts.max_k,
        max_n=opts.max_n,
    )
    print(mlir_module)


if __name__ == "__main__":
    main()
