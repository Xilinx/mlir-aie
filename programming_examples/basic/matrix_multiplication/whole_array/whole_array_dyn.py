#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiplication with dispatch-time M, K and N.

The design in whole_array.py bakes the problem shape into the compiled
artifact. This variant takes the shape as `DispatchTime` scalars: it compiles
once, and every call rebuilds the instruction stream on the host from the
same xclbin, in C++, in microseconds, with no Python in the loop. The tiling
logic is the same taplib algebra as the static design, evaluated inside the
runtime-sequence body on staged values: taps come out as arithmetic on `M`,
`K`, `N`, the time-block loop stays rolled (`range_`) with the in-flight
step's `TaskGroup` carried as an iter_arg so two halves stay in flight as in
the static design, the ragged last row block is a peeled `if_`, and the shape
constraints the static design asserts become `require` guards that refuse an
illegal dispatch before anything reaches the NPU.

The one thing the compiled design must still know is how large the host
buffers are. The runtime sequence's arguments are typed memrefs, and
`In`/`Out` tensors are not visible while the design is generated, so their
sizes are given as `A_elements`, `B_elements` and `C_elements`. Any shape
whose `A`, `B` and `C` fit those buffers (packed row-major at the front)
runs on the same xclbin; a larger one is refused by a guard. The command
line derives the sizes from the shapes it is asked to run.

Per-core trip counts (`K // k` and the number of output tiles a core
produces) reach the workers as runtime parameters: each worker owns an RTP
buffer the sequence writes before releasing a barrier the worker waits on.
The worker steps the barrier back off once it has read them, so its next
dispatch waits for that dispatch's values instead of reusing these.

A fully static specialization (`whole_array_dyn.specialize(M=.., K=.., N=..)`)
takes the ordinary static path and issues exactly the transfers
whole_array.py issues for that shape; `tests/dispatch_txn.py` checks the
dynamic builder against it with `aie.utils.txn_trace`.
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import TensorAccessPattern
from aie.helpers.taplib.symbolic import require
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
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import if_, range_, yield_
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark
from whole_array import _device_for


@iron.jit
def whole_array_dyn(
    A: In,
    B: In,
    C: Out,
    *,
    M: DispatchTime[np.int32],
    K: DispatchTime[np.int32],
    N: DispatchTime[np.int32],
    A_elements: CompileTime[int],
    B_elements: CompileTime[int],
    C_elements: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
    b_col_maj: CompileTime[int] = 0,
    c_col_maj: CompileTime[int] = 0,
    use_chess: CompileTime[bool] = False,
):
    n_aie_rows = 4
    n_aie_cores = n_aie_rows * n_aie_cols
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        b_col_maj=bool(b_col_maj),
        c_col_maj=bool(c_col_maj),
        use_chess=use_chess,
    )
    zero_kernel = kernels.zero(m * n, dtype_out, use_chess=use_chess)
    dims = matmul_kernel.stream_dims

    dev = iron.get_current_device()
    if n_aie_cols > dev.cols:
        raise ValueError(
            f"n_aie_cols={n_aie_cols} but the device has {dev.cols} columns"
        )
    # Element offsets into the buffers are i32 dispatch-time arithmetic.
    for name, size in (("A", A_elements), ("B", B_elements), ("C", C_elements)):
        if not 0 < size < 2**31:
            raise ValueError(f"{name}_elements={size} must be in [1, 2**31)")

    fifo_depth = 2
    n_shim_mem_A = n_aie_rows if n_aie_cols > n_aie_rows else n_aie_cols
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols if n_aie_cols < 4 else 1

    A_ty = np.ndarray[(A_elements,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(B_elements,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(C_elements,), np.dtype[dtype_out]]
    A_l2_ty = np.ndarray[(m * k * n_A_tiles_per_shim,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

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
                to_stream=[dims.A or []] * (stop_row - start_row),
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

    tb_max_n_rows = 4 if not c_col_maj else 2
    tb_n_rows = tb_max_n_rows // 2

    A_prods = [f.prod() for f in A_l3l2_fifos]
    B_prods = [f.prod() for f in B_l3l2_fifos]
    C_conses = [f.cons() for f in C_l2l3_fifos]

    def sequence(A, B, C, M, K, N, A_hs, B_hs, C_hs):
        # The static design's asserts, as guards: a ValueError on a static
        # specialization, a refused dispatch (no stream) on the dynamic path.
        require(M > 0, "M must be positive")
        require(K > 0, "K must be positive")
        require(N > 0, "N must be positive")
        require(M % (m * n_aie_rows) == 0, "M must be a multiple of m * n_aie_rows")
        require(K % k == 0, "K must be a multiple of k")
        require(N % (n * n_aie_cols) == 0, "N must be a multiple of n * n_aie_cols")
        # x * y <= size as x <= size // y: the i32 product could overflow.
        require(M <= A_elements // K, "A (M x K) does not fit A_elements")
        require(K <= B_elements // N, "B (K x N) does not fit B_elements")
        require(M <= C_elements // N, "C (M x N) does not fit C_elements")

        k_iters = K // k
        n_tiles_per_core = (M // m) * (N // n) // n_aie_cores
        for row in range(n_aie_rows):
            for col in range(n_aie_cols):
                rtps[row][col][0] = k_iters
                rtps[row][col][1] = n_tiles_per_core
                barriers[row][col].set(1)

        # The same tilers as whole_array.py, on the live shape. Every grid
        # size, stride and index below is staged arithmetic on M, K, N.
        A_tiles = (
            TensorAccessPattern.full((M, K))
            .tile((m * n_A_tiles_per_shim, k))
            .group((1, K // k))
            .repeat(N // n // n_aie_cols)
        )
        if b_col_maj:
            B_tiles = (
                TensorAccessPattern.full((N, K))
                .tile((n, k))
                .group((N // n // n_aie_cols, K // k), steps=(n_aie_cols, 1))
            )
        else:
            B_tiles = (
                TensorAccessPattern.full((K, N))
                .tile((k, n))
                .group(
                    (K // k, N // n // n_aie_cols),
                    steps=(1, n_aie_cols),
                    order="col",
                )
            )
        if c_col_maj:
            C_tiles = (
                TensorAccessPattern.full((N, M))
                .tile((n, m))
                .order("col")
                .group((N // n // n_aie_cols, n_aie_rows), steps=(n_aie_cols, 1))
            )
        else:
            # partial: the last group may hold fewer than tb_n_rows row blocks.
            C_tiles = (
                TensorAccessPattern.full((M, N))
                .tile((m * n_aie_rows, n))
                .group(
                    (tb_n_rows, N // n // n_aie_cols),
                    steps=(1, n_aie_cols),
                    partial=True,
                )
            )

        n_row_tiles = M // m // n_aie_rows
        n_full_steps = n_row_tiles // tb_n_rows
        n_ragged_rows = n_row_tiles % tb_n_rows

        def issue_step(step, n_rows):
            """Issue one time block half: n_rows (static) row blocks per column."""
            tg = TaskGroup()
            row_base = step * tb_n_rows
            for col in range(n_aie_cols):
                C_hs[col].drain(
                    C, tap=C_tiles[step * n_aie_cols + col], wait=True, group=tg
                )
                for tile_row in range(n_rows):
                    tile_offset = (
                        (row_base + tile_row) * n_shim_mem_A + col
                    ) % A_tiles.num_steps
                    if col < n_aie_rows:
                        A_hs[col].fill(A, tap=A_tiles[tile_offset], group=tg)
                    B_hs[col].fill(B, tap=B_tiles[col], group=tg)
            return tg

        # Two time-block halves in flight, as in whole_array.py: a step's group
        # is finished only after the next step's is issued. The in-flight group
        # rides the loop as an iter_arg, so the loop stays rolled over the
        # dispatch-time trip count.
        with if_(n_full_steps > 0):
            prev = issue_step(0, tb_n_rows)
            last = prev
            for iv, prev, last in range_(
                1, n_full_steps, iter_args=[prev], insert_yield=False
            ):
                current = issue_step(iv, tb_n_rows)
                prev.finish()
                yield_([current])
            last.finish()
        if tb_n_rows > 1:
            with if_(n_ragged_rows > 0):
                issue_step(n_full_steps, 1).finish()

    rt = Runtime(sequence, [A_ty, B_ty, C_ty, M, K, N, A_prods, B_prods, C_conses])
    return Program(dev, rt, workers=flat_workers).resolve_program()


def main():
    p = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication (Whole Array, dispatch-time shape)"
    )
    add_compile_args(p, short_dev=None)
    p.add_argument(
        "--shapes",
        type=lambda s: tuple(int(x) for x in s.split("x")),
        nargs="+",
        default=[(512, 512, 512)],
        metavar="MxKxN",
        help="shapes to run on one compiled design; the host buffers are "
        "sized for the largest A, B and C among them",
    )
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--b-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--c-col-maj", type=int, choices=[0, 1], default=0)
    p.add_argument("--dtype_in", type=str, choices=["bf16", "i8", "i16"], default="i16")
    p.add_argument(
        "--dtype_out",
        type=str,
        choices=["bf16", "i8", "i16", "f32", "i32"],
        default="i32",
    )
    p.add_argument("--use-chess", type=int, choices=[0, 1], default=0)
    add_benchmark_args(p)
    opts = p.parse_args()

    compile_kwargs = dict(
        A_elements=max(M * K for M, K, _ in opts.shapes),
        B_elements=max(K * N for _, K, N in opts.shapes),
        C_elements=max(M * N for M, _, N in opts.shapes),
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
        b_col_maj=opts.b_col_maj,
        c_col_maj=opts.c_col_maj,
        use_chess=bool(opts.use_chess),
    )

    def run_and_verify(opts):
        dtype_in = str_to_dtype(opts.dtype_in)
        dtype_out = str_to_dtype(opts.dtype_out)
        tolerance = kernels.mm(
            dim_m=opts.m,
            dim_k=opts.k,
            dim_n=opts.n,
            input_dtype=dtype_in,
            output_dtype=dtype_out,
            b_col_maj=bool(opts.b_col_maj),
        ).contract.tolerance
        rng = np.random.default_rng(1726250518)
        A_t = iron.zeros((compile_kwargs["A_elements"],), dtype=dtype_in, device="npu")
        B_t = iron.zeros((compile_kwargs["B_elements"],), dtype=dtype_in, device="npu")
        C_t = iron.zeros((compile_kwargs["C_elements"],), dtype=dtype_out, device="npu")
        for M, K, N in opts.shapes:
            print(f"M={M} K={K} N={N}")
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
            A_t.numpy_view()[: A_np.size] = A_np.reshape(-1)
            B_t.numpy_view()[: B_np.size] = B_np.reshape(-1)
            bench = run_iters(
                whole_array_dyn,
                A_t,
                B_t,
                C_t,
                M=M,
                K=K,
                N=N,
                **compile_kwargs,
                warmup=opts.warmup,
                iters=opts.iters,
            )
            expected = kernels.mm_ref(A_np, B_np.T if opts.b_col_maj else B_np)
            live = C_t.numpy()[: M * N]
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

    run_design_cli(
        whole_array_dyn,
        opts,
        compile_kwargs=compile_kwargs,
        run_and_verify=run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
    )


if __name__ == "__main__":
    main()
