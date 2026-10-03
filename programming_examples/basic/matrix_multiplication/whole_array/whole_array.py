#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiply — IRON API design with ``@iron.jit`` compilation.

A 4xN_cols AIE array computes ``C = A @ B`` (optionally with ``B`` column-major
or ``C`` column-major).  Each compute tile owns one (m, n) output sub-tile and
streams (m, k) x (k, n) inputs through three layers of ObjectFifos.

``M``, ``K`` and ``N`` are ``DispatchTime`` scalars. Specialized
(``whole_array.specialize(M=.., K=.., N=..)``) they are constants and the
runtime sequence unrolls. Left free, the design compiles once and every call
rebuilds the instruction stream on the host from the same xclbin, in C++, with
no Python in the loop: the taps come out as arithmetic on ``M``, ``K`` and
``N``, and the row-block loop stays rolled. The tensors the design is called
with size its host buffers, so any shape whose ``A``, ``B`` and ``C`` fit them
(packed row-major at the front) runs on the same xclbin, and a shape that
breaks a constraint or does not fit is refused before it reaches the NPU.

Per-core trip counts (``K // k`` and the number of output tiles a core
produces) reach the workers as runtime parameters: each worker owns an RTP
buffer the sequence writes before releasing a barrier the worker waits on.
The worker steps the barrier back off once it has read them, so its next
dispatch waits for that dispatch's values instead of reusing these.

The script has three modes:

* ``--xclbin-path=... --insts-path=...`` — compile the design ahead-of-time
  for ``-M -K -N`` and write artifacts to the given paths (bypasses the JIT
  cache).  Used by ``makefile-common`` so ``test.cpp`` + ``sweep.sh`` can
  drive the design via ``make``.
* ``--dynamic MxKxN ...`` — compile once with dispatch-time shapes and run
  every listed shape on the NPU, verifying each against numpy.
* default — JIT-compile for ``-M -K -N`` + run on the attached NPU + verify
  against numpy.
"""

import argparse
from typing import NamedTuple

import aie.iron as iron
import aie.iron.kernels as kernels
import numpy as np
from aie.helpers.npdtypes import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    WorkerRuntimeBarrier,
    require,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark


class Tilings(NamedTuple):
    """The A, B and C tilings the runtime sequence fills and drains."""

    A: TensorAccessPattern
    B: TensorAccessPattern
    C: TensorAccessPattern


def tile_matrices(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj) -> Tilings:
    """Tile A, B and C for the runtime sequence.

    ``A[i]`` is the i-th row of A tiles, walked ``N // n // n_aie_cols`` times;
    ``B[j]`` is the j-th column of B tiles; ``C[i, j]`` is the C tile at row
    block ``i`` and column ``j``. ``M``, ``K`` and ``N`` may be ints or
    dispatch-time scalars.
    """
    n_aie_rows = 4
    n_A_tiles_per_shim = max(1, n_aie_rows // n_aie_cols)

    A_tiles = (
        TensorAccessPattern.full((M, K))
        .tile((m * n_A_tiles_per_shim, k))
        .repeat(N // n // n_aie_cols)
        .permute((1, 0, 2, 3, 4))
    )
    if b_col_maj:
        B_tiles = TensorAccessPattern.full((N, K)).tile((n, k))
    else:
        B_tiles = TensorAccessPattern.full((K, N)).tile((k, n)).permute((1, 0, 2, 3))
    if c_col_maj:
        # A row block is n_aie_rows tiles of the transposed (N, M) layout.
        C_tiles = (
            TensorAccessPattern.full((N, M))
            .tile((n, m))
            .split(1, n_aie_rows)
            .permute((1, 0, 2, 3, 4))
        )
    else:
        C_tiles = TensorAccessPattern.full((M, N)).tile((m * n_aie_rows, n))
    return Tilings(A_tiles, B_tiles, C_tiles)


def step_transfers(tilings: Tilings, row, n_aie_cols: int) -> list:
    """Return the shim transfers of row block ``row``.

    Each transfer is ``(tensor, col, tap)``: the shim column ``col`` fills
    ``"A"`` or ``"B"``, or drains ``"C"``, with the access pattern ``tap``.
    ``row`` may be a dispatch-time scalar.
    """
    n_shim_mem_A = min(4, n_aie_cols)
    transfers = []
    for col in range(n_aie_cols):
        transfers.append(("C", col, tilings.C[row, col::n_aie_cols]))
        if col < n_shim_mem_A:
            transfers.append(("A", col, tilings.A[row * n_shim_mem_A + col]))
        transfers.append(("B", col, tilings.B[col::n_aie_cols]))
    return transfers


@iron.jit
def whole_array(
    A: In,
    B: In,
    C: Out,
    *,
    M: DispatchTime[np.int32],
    K: DispatchTime[np.int32],
    N: DispatchTime[np.int32],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    emulate_bf16_mmul_with_bfp16: CompileTime[bool] = False,
    use_chess: CompileTime[bool] = False,
    scalar: CompileTime[bool] = False,
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    fifo_depth = 2
    n_shim_mem_A = min(n_aie_rows, n_aie_cols)
    n_A_tiles_per_shim = max(1, n_aie_rows // n_aie_cols)

    dtype_in = np_ndarray_type_get_dtype(A)
    dtype_out = np_ndarray_type_get_dtype(C)

    assert np.issubdtype(dtype_in, np.integer) == np.issubdtype(
        dtype_out, np.integer
    ), f"Input dtype ({dtype_in}) and output dtype ({dtype_out}) must either both be integral or both be float"
    assert (
        np.dtype(dtype_out).itemsize >= np.dtype(dtype_in).itemsize
    ), f"Output dtype ({dtype_out}) must be equal or larger to input dtype ({dtype_in})"

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
        emulate_bf16_mmul_with_bfp16=emulate_bf16_mmul_with_bfp16,
        vectorized=not scalar,
    )
    zero_kernel = kernels.zero(m * n, dtype_out, use_chess=use_chess)
    r, s, t = matmul_kernel.mac_dims
    dims = matmul_kernel.stream_dims
    assert m % r == 0
    assert k % s == 0
    assert n % t == 0

    dev = iron.get_current_device()
    assert dev is not None
    if n_aie_cols > dev.cols:
        raise ValueError(
            f"n_aie_cols={n_aie_cols} but the device has {dev.cols} columns"
        )
    # Element offsets into the buffers are i32 dispatch-time arithmetic.
    for name, ty in (("A", A), ("B", B), ("C", C)):
        if not np.prod(np_ndarray_type_get_shape(ty)) < 2**31:
            raise ValueError(f"{name} must hold fewer than 2**31 elements")

    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    # Shim -> memtile -> core fifos: A per row, B per column, C per [row][col].
    A_l3l2_fifos: list[ObjectFifo] = []
    A_l2l1_fifos: list[ObjectFifo] = []
    B_l3l2_fifos: list[ObjectFifo] = []
    B_l2l1_fifos: list[ObjectFifo] = []
    C_l1l2_fifos: list[list[ObjectFifo]] = [[] for _ in range(n_aie_rows)]
    C_l2l3_fifos: list[ObjectFifo] = []

    for i in range(n_shim_mem_A):
        a_l3l2 = ObjectFifo(A_l2_ty, name=f"A_L3L2_{i}", depth=fifo_depth)
        A_l3l2_fifos.append(a_l3l2)
        start_row = i * n_A_tiles_per_shim
        stop_row = start_row + n_A_tiles_per_shim
        A_l2l1_fifos.extend(
            a_l3l2.cons().split(
                [m * k * j for j in range(stop_row - start_row)],
                obj_types=[A_l1_ty] * (stop_row - start_row),
                names=[f"A_L2L1_{row}" for row in range(start_row, stop_row)],
                to_stream=[dims.A] * (stop_row - start_row),
            )
        )

    for col in range(n_aie_cols):
        b_l3l2 = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l3l2_fifos.append(b_l3l2)
        B_l2l1_fifos.append(
            b_l3l2.cons().forward(
                obj_type=B_l1_ty, name=f"B_L2L1_{col}", to_stream=dims.B
            )
        )
        c_l2l3 = ObjectFifo(
            C_l2_ty, name=f"C_L2L3_{col}", depth=fifo_depth, to_stream=dims.C
        )
        C_l2l3_fifos.append(c_l2l3)
        c_tmp_fifos = c_l2l3.prod().join(
            [m * n * i for i in range(n_aie_rows)],
            obj_types=[C_l1_ty] * n_aie_rows,
            names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
            depths=[fifo_depth] * n_aie_rows,
        )
        for j in range(n_aie_rows):
            C_l1l2_fifos[j].append(c_tmp_fifos[j])

    # Each worker's trip counts arrive as runtime parameters: rtp[0] = K // k,
    # rtp[1] = output tiles per core. The sequence writes them, then releases
    # the barrier the worker waits on at the top of every dispatch.
    rtps = [
        [
            Buffer(rtp_ty, name=f"rtp_{row}_{col}", use_write_rtp=True)
            for col in range(n_aie_cols)
        ]
        for row in range(n_aie_rows)
    ]
    barriers = [
        [WorkerRuntimeBarrier() for _ in range(n_aie_cols)] for _ in range(n_aie_rows)
    ]

    def core_fn(in_a, in_b, out_c, zero, matmul, rtp, barrier):
        barrier.wait_for_value(1)
        k_iters = rtp[0]
        n_tiles = rtp[1]
        # The wait leaves the barrier at 1; step it off so the next dispatch
        # blocks until that dispatch's own set(1) instead of reusing these.
        barrier.release_with_value(1)
        for _ in range_(n_tiles):
            elem_out = out_c.acquire(1)
            zero(elem_out)
            for _ in range_(k_iters):
                elem_in_a = in_a.acquire(1)
                elem_in_b = in_b.acquire(1)
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
                rtps[row][col],
                barriers[row][col],
            ],
            stack_size=0xD00,
        ),
    )
    flat_workers = [w for row in workers for w in row]

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, M, K, N, A_hs, B_hs, C_hs):
        # A ValueError on a specialized shape; a refused dispatch on a live one.
        require(M > 0, "M must be positive")
        require(K > 0, "K must be positive")
        require(N > 0, "N must be positive")
        require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")
        require(K % k == 0, "K must be a multiple of k")
        require(N % (n * n_aie_cols) == 0, "N must be a multiple of n * n_aie_cols")

        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = K // k
                rtps[row][col][1] = (M // m) * (N // n) // n_aie_cores
                barriers[row][col].set(1)

        # Four row blocks in flight: each block's transfers form a group that
        # is finished only after the next three blocks' are issued.
        tilings = tile_matrices(M, K, N, m, k, n, n_aie_cols, b_col_maj, c_col_maj)
        for row, tg in TaskGroup.pipelined(M // m // n_aie_rows, depth=4):
            for tensor, col, tap in step_transfers(tilings, row, n_aie_cols):
                if tensor == "C":
                    C_hs[col].drain(C, tap=tap, wait=True, group=tg)
                elif tensor == "A":
                    A_hs[col].fill(A, tap=tap, group=tg)
                else:
                    B_hs[col].fill(B, tap=tap, group=tg)

    rt = Runtime(sequence, [A, B, C, M, K, N, A_prods, B_prods, C_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Matrix Multiplication (Whole Array)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument(
        "--dynamic",
        type=lambda s: tuple(int(x) for x in s.split("x")),
        nargs="+",
        metavar="MxKxN",
        help="compile once with dispatch-time M, K and N and run each shape; "
        "the host buffers are sized for the largest A, B and C among them",
    )
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--emulate-bf16-mmul-with-bfp16", type=int, choices=[0, 1], default=0
    )
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i16",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    p.add_argument(
        "--scalar",
        type=int,
        choices=[0, 1],
        default=0,
        help="use scalar (non-vector) matmul/zero kernels for debugging smaller sizes",
    )
    add_benchmark_args(p)
    return p


def _buffer_shapes(opts):
    """Return the A, B and C buffer shapes: one shape's, or the largest of ``--dynamic``."""
    if opts.dynamic:
        return (
            (max(M * K for M, K, _ in opts.dynamic),),
            (max(K * N for _, K, N in opts.dynamic),),
            (max(M * N for M, _, N in opts.dynamic),),
        )
    B_shape = (opts.N, opts.K) if opts.b_col_maj else (opts.K, opts.N)
    return (opts.M, opts.K), B_shape, (opts.M, opts.N)


def _compile_kwargs(opts):
    dtypes = [str_to_dtype(opts.dtype_in)] * 2 + [str_to_dtype(opts.dtype_out)]
    tensor_types = {
        name: np.ndarray[shape, np.dtype[dtype]]
        for name, shape, dtype in zip("ABC", _buffer_shapes(opts), dtypes)
    }
    shape = {} if opts.dynamic else dict(M=opts.M, K=opts.K, N=opts.N)
    return dict(
        **tensor_types,
        **shape,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        emulate_bf16_mmul_with_bfp16=bool(opts.emulate_bf16_mmul_with_bfp16),
        use_chess=bool(opts.use_chess),
        scalar=bool(opts.scalar),
    )


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)
    # The same kernel the design binds; kernels.mm is memoized, so this is
    # the object the design built, asked for its tolerance.
    tolerance = kernels.mm(
        dim_m=opts.m,
        dim_k=opts.k,
        dim_n=opts.n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(opts.b_col_maj),
    ).contract.tolerance

    design = whole_array.specialize(**_compile_kwargs(opts))
    rng = np.random.default_rng(1726250518)
    A_t, B_t, C_t = (
        iron.zeros(buf, dtype=dt, device="npu")
        for buf, dt in zip(_buffer_shapes(opts), (dtype_in, dtype_in, dtype_out))
    )
    for M, K, N in opts.dynamic or [(opts.M, opts.K, opts.N)]:
        B_shape = (N, K) if opts.b_col_maj else (K, N)
        if np.issubdtype(dtype_in, np.integer):
            info = np.iinfo(dtype_in)
            lo, hi = info.min // 4, info.max // 4
            A_np = rng.integers(lo, hi, size=(M, K), dtype=dtype_in)
            B_np = rng.integers(lo, hi, size=B_shape, dtype=dtype_in)
        else:
            A_np = (rng.random((M, K)) * 4.0).astype(dtype_in)
            B_np = (rng.random(B_shape) * 4.0).astype(dtype_in)
        # Each matrix sits packed row-major at the front of its buffer.
        A_t.numpy_view().flat[: A_np.size] = A_np.flat
        B_t.numpy_view().flat[: B_np.size] = B_np.flat
        shape = {}
        if opts.dynamic:
            print(f"M={M} K={K} N={N}")
            shape = dict(M=M, K=K, N=N)

        bench = run_iters(
            design,
            A_t,
            B_t,
            C_t,
            **shape,
            warmup=opts.warmup,
            iters=opts.iters,
        )

        expected = kernels.mm_ref(A_np, B_np.T if opts.b_col_maj else B_np)
        live = C_t.numpy().reshape(-1)[: M * N]
        if opts.c_col_maj:
            actual, expected = live.reshape(N, M), expected.T
        else:
            actual = live.reshape(M, N)
        assert_close_with_benchmark(
            actual,
            expected.astype(dtype_out),
            bench=bench,
            ops=2.0 * M * K * N,
            tolerance=tolerance,
            fail_msg="output does not match A @ B",
            mismatch_indices=True,
        )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        # NPU1 binds the ColN variant matching n_aie_cols; NPU2 keeps the
        # full array for the placer.
        device=lambda o: from_name(
            o.dev, n_cols=o.n_aie_cols if o.dev == "npu" else None
        ),
    )


if __name__ == "__main__":
    main()
