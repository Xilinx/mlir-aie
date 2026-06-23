# n32_core.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
"""32-core (8 cols x 4 rows) asymmetric-tile-buffered bf16/bfp16 GEMM —
``@iron.jit`` IRON design.

Each AIE2P core's L1 A-tile is (m//4, k); the L2 A buffer pre-fetches
``mtk = 512`` columns at a time and the memtile fan-out splits it into
four A-subtiles per matmul cycle. B is bfp16ebs8; C is bf16. Strix-only.
"""

import argparse
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.dialects.aiex import v8bfp16ebs8
from aie.helpers.taplib import TensorTiler2D

import aie.iron as iron
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils.hostruntime.argparse import (
    device_from_args,
    add_compile_args,
)
from aie.utils.hostruntime.cli import run_design_cli

_KERNEL_SRC = Path(__file__).resolve().parent / "mm_bfp_mixed.cc"
_AIE_KERNELS_INC = Path(__file__).resolve().parents[5] / "aie_kernels"


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def n32_core_gemm(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int] = 4096,
    K: CompileTime[int] = 4096,
    N: CompileTime[int] = 2048,
    m: CompileTime[int] = 128,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 128,
):
    n_aie_cols = 8
    n_aie_rows = 4
    r, s, t = 8, 8, 8
    mtk = 512

    A_l2_ty = np.ndarray[(m, mtk), np.dtype[bfloat16]]
    B_l2_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l2_ty = np.ndarray[(n_aie_rows * m, n), np.dtype[bfloat16]]
    A_l1_ty = np.ndarray[(m // 4, k), np.dtype[bfloat16]]
    B_l1_ty = np.ndarray[(k, n // 8), np.dtype[v8bfp16ebs8]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[bfloat16]]

    kernel_flags = [
        f"-DDIM_M={m}",
        f"-DDIM_K={k}",
        f"-DDIM_N={n}",
        f"-I{_AIE_KERNELS_INC}",
    ]

    zero_kernel = ExternalFunction(
        "zero_kernel_bf16",
        source_file=str(_KERNEL_SRC),
        arg_types=[C_l1_ty],
        compile_flags=kernel_flags + ["-DZERO_ONLY"],
        use_chess=True,
    )
    matmul_kernel = ExternalFunction(
        "matmul_vectorized_different_datatypes",
        source_file=str(_KERNEL_SRC),
        arg_types=[A_l1_ty, B_l1_ty, C_l1_ty],
        compile_flags=kernel_flags + ["-DMATMUL_ONLY"],
        use_chess=True,
    )

    A_l3l2_fifos = [None] * n_aie_rows
    A_l2l1_fifos = [None] * n_aie_rows
    B_l3l2_fifos = [None] * n_aie_cols
    B_l2l1_fifos = [None] * n_aie_cols
    C_l1l2_fifos = [[None] * n_aie_cols for _ in range(n_aie_rows)]
    C_l2l3_fifos = [None] * n_aie_cols

    # Input A: 4 shim-rows, each shim→memtile carries an L2 strip and
    # the memtile then fans out the L1 sub-tiles to all 8 cores in that row.
    a_l3l2_dims = [(m, k), (mtk // k, m * k), (k, 1)]
    a_l2l1_in_dims = [
        (mtk // k * 4, m * k // 4),
        (k // s, s),
        (m // 4, k),
        (s, 1),
    ]
    a_l2l1_out_dims = [
        (k // s, r * s),
        (m // 4 // r, r * k),
        (r * s, 1),
    ]

    for row in range(n_aie_rows):
        A_l3l2_fifos[row] = ObjectFifo(
            A_l2_ty,
            name=f"A_L3L2_{row}",
            depth=2,
            dims_from_stream_per_cons=a_l3l2_dims,
        )
        # forward() emits an ObjectFifoLink at the memtile. The new
        # (A_l2l1) fifo's producer-side `dims_to_stream` is the memtile
        # TX layout, and per-cons `dims_from_stream` is the layout each
        # of the 8 compute-tile consumers reads from the stream.
        A_l2l1_fifos[row] = (
            A_l3l2_fifos[row]
            .cons()
            .forward(
                obj_type=A_l1_ty,
                name=f"A_L2L1_{row}",
                depth=2,
                dims_to_stream=a_l2l1_in_dims,
                dims_from_stream=a_l2l1_out_dims,
            )
        )

    # Input B: 8 cols, each shim→memtile→fan to 4 rows of cores.
    for col in range(n_aie_cols):
        B_l3l2_fifos[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=2)
        B_l2l1_fifos[col] = (
            B_l3l2_fifos[col]
            .cons()
            .forward(obj_type=B_l1_ty, name=f"B_L2L1_{col}", depth=2)
        )

    # Output C: per-col, 4 cores join at memtile, memtile→shim with the
    # final layout transform.
    c_l2l3_dims = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    for col in range(n_aie_cols):
        C_l2l3_fifos[col] = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=2, dims_to_stream=c_l2l3_dims
        )
        of_offsets = [m * n * i for i in range(n_aie_rows)]
        c_tmp_fifos = (
            C_l2l3_fifos[col]
            .prod()
            .join(
                of_offsets,
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[1] * n_aie_rows,
            )
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j][col] = c_tmp_fifos[j]

    tiles_per_core = (M // m) * (N // n) // (n_aie_cols * n_aie_rows)

    def core_fn(in_a, in_b, out_c, zero, matmul):
        for _ in range_(tiles_per_core) if tiles_per_core > 1 else range(1):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(K // k):
                elem_in_b = in_b.acquire(1)
                for _ in range(4):
                    elem_in_a = in_a.acquire(1)
                    matmul(elem_in_a, elem_in_b, elem_out)
                    in_a.release(1)
                in_b.release(1)
            out_c.release(1)

    workers = Worker.grid(
        n_aie_rows,
        n_aie_cols,
        lambda row, col: Worker(
            core_fn,
            [
                A_l2l1_fifos[row].cons(),
                B_l2l1_fifos[col].cons(),
                C_l1l2_fifos[row][col].prod(),
                zero_kernel,
                matmul_kernel,
            ],
            stack_size=0xD00,
        ),
    )

    A_ty = np.ndarray[(M * K,), np.dtype[bfloat16]]
    B_ty = np.ndarray[(K * N // 8,), np.dtype[v8bfp16ebs8]]
    C_ty = np.ndarray[(M * N,), np.dtype[bfloat16]]

    A_taps = TensorTiler2D.group_tiler((M, K), (m, mtk), (1, K // mtk))
    B_taps = TensorTiler2D.group_tiler((1, N * K // 8), (1, n * K // 8), (1, 1))
    C_taps = TensorTiler2D.group_tiler((M, N), (n_aie_rows * m, n), (1, 1))

    num_row_tile = M // m // n_aie_rows
    num_col_tile = N // n // n_aie_cols
    num_groups = num_row_tile * num_col_tile
    tb_max_n_rows = 4

    rt = Runtime()
    with rt.sequence(A_ty, B_ty, C_ty) as (a, b, c):
        rt.start(*[w for row in workers for w in row])
        # 4-slot rotating task-group pipeline. Each slot is one
        # iteration's (4 A-fills + 8 B-fills + 8 C-drains). A/B fills are
        # wait=False (free at finish), C drains are wait=True (await +
        # free at finish). The interleaved finish_task_group calls below
        # match the original placed-style rotation 1:1, so the lowered
        # MLIR sequence is structurally identical to the hand-managed
        # dma_start/await/free sequence:
        #
        #   group_idx % 4 == 1 (after the first wrap): finish slots 2,3
        #   group_idx % 4 == 3:                        finish slots 0,1
        #
        # This caps in-flight BDs at exactly the original's 4-group
        # window (no extra BD pressure introduced by the IRON wrapper).
        slots = [None] * tb_max_n_rows
        for group_idx in range(num_groups):
            slot_idx = group_idx % tb_max_n_rows
            tg = rt.task_group()
            slots[slot_idx] = tg

            a_base_idx = (group_idx // num_col_tile) * n_aie_rows
            for row in range(n_aie_rows):
                rt.fill(
                    A_l3l2_fifos[row].prod(),
                    a,
                    tap=A_taps[a_base_idx + row],
                    task_group=tg,
                    wait=False,
                )
            b_base_idx = (group_idx % num_col_tile) * n_aie_cols
            for col in range(n_aie_cols):
                rt.fill(
                    B_l3l2_fifos[col].prod(),
                    b,
                    tap=B_taps[b_base_idx + col],
                    task_group=tg,
                    wait=False,
                )
            c_base_idx = group_idx * n_aie_cols
            for col in range(n_aie_cols):
                rt.drain(
                    C_l2l3_fifos[col].cons(),
                    c,
                    tap=C_taps[c_base_idx + col],
                    task_group=tg,
                    wait=True,
                )

            if slot_idx == 1 and group_idx != 1:
                rt.finish_task_group(slots[2])
                rt.finish_task_group(slots[3])
            if slot_idx == 3:
                rt.finish_task_group(slots[0])
                rt.finish_task_group(slots[1])

        # Drain the two slots still pending at the end of the rotation.
        rt.finish_task_group(slots[2])
        rt.finish_task_group(slots[3])

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(
        prog="AIE 32-Core ATB GEMM (bf16/bfp16/bf16)",
    )
    add_compile_args(p, default_dev="npu2")
    p.add_argument("-M", type=int, default=4096)
    p.add_argument("-K", type=int, default=4096)
    p.add_argument("-N", type=int, default=2048)
    p.add_argument("-m", type=int, default=128)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=128)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, K=opts.K, N=opts.N, m=opts.m, k=opts.k, n=opts.n)


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        n32_core_gemm,
        opts,
        compile_kwargs=_compile_kwargs,
        device=lambda o: device_from_args(o, n_cols=None),
    )


if __name__ == "__main__":
    main()
