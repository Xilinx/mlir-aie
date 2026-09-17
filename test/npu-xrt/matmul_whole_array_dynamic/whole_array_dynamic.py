#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Whole-array matrix multiply with a *dynamic* runtime sequence.

Same device/worker/ObjectFifo structure as ``whole_array.py``, but the host
runtime sequence is written with ``range_`` loops and SSA arithmetic over the
problem dimensions M/K/N instead of Python-unrolled ``TensorTiler2D`` taps. The
DMAs use ``fifo.fill``/``fifo.drain`` with runtime-valued sizes / strides /
offsets, so a single body serves both lowerings.

One design, two lowerings, selected by explicit specialization:

* **static** — call ``specialize(M=..., K=..., N=...)``. The bounds are constant, so
  ``aie-unroll-runtime-sequence-loops`` flattens the loops and everything folds
  to the same BDs the ``TensorTiler2D`` version emits (binary TXN path).
* **dynamic** — bind compile-time K, and pass M/N at execution time. The ``scf.for`` loops
  survive to the EmitC path (``--aie-npu-to-cpp``), so one xclbin runs many
  shapes; the C++ builder assembles the TXN per call. K is fixed because the
  workers' reduction depth is compiled into their programs.

The BD size/stride/offset formulas match the explicit-math form of the design
(see the module docstring in ``whole_array.py`` for the tiling picture).
"""

import argparse

import aie.iron as iron
import numpy as np
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.iron import (
    CompileTime,
    DispatchTime,
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
from aie.iron.device import from_name
from aie.utils.benchmark import run_iters
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark

# Fixed tiling constants (compile-time).
N_AIE_ROWS = 4


def _device_for(dev_str, n_aie_cols):
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


@iron.jit
def whole_array_dynamic(
    A: In,
    B: In,
    C: Out,
    M: DispatchTime[np.int32],
    N: DispatchTime[np.int32],
    *,
    K: CompileTime[int],
    A_elements: CompileTime[int],
    B_elements: CompileTime[int],
    C_elements: CompileTime[int],
    m: CompileTime[int] = 64,
    k: CompileTime[int] = 64,
    n: CompileTime[int] = 32,
    n_aie_cols: CompileTime[int] = 4,
    dtype_in_str: CompileTime[str] = "i16",
    dtype_out_str: CompileTime[str] = "i16",
):
    """One generator: prebind M/N for static lowering, pass them per call otherwise.

    A_elements/B_elements/C_elements describe physical host-buffer capacities,
    not additional problem dimensions. In/Out tensors are execution-time values,
    so their allocation sizes must be supplied separately for MLIR generation.
    K is compile-time because the workers' reduction depth is fixed. M/N can vary
    within those capacities, using packed prefixes of the host buffers.
    """
    if any(size <= 0 for size in (A_elements, B_elements, C_elements)):
        raise ValueError("Host-buffer element capacities must be positive.")
    if K <= 0 or K % k:
        raise ValueError("K must be positive and divisible by k.")
    if isinstance(M, (int, np.integer)) and not (
        0 < M and int(M) * int(K) <= A_elements and M % (m * N_AIE_ROWS) == 0
    ):
        raise ValueError("M must fit the capacity and be divisible by m * N_AIE_ROWS.")
    if isinstance(N, (int, np.integer)) and not (
        0 < N and int(K) * int(N) <= B_elements and N % (n * n_aie_cols) == 0
    ):
        raise ValueError("N must fit the capacity and be divisible by n * n_aie_cols.")
    if isinstance(M, (int, np.integer)) and isinstance(N, (int, np.integer)):
        if int(M) * int(N) > C_elements:
            raise ValueError("M * N must fit the output buffer capacity.")
    dtype_in = str_to_dtype(dtype_in_str)
    dtype_out = str_to_dtype(dtype_out_str)

    n_aie_rows = N_AIE_ROWS
    assert n_aie_cols in (1, 2, 4), "n_aie_cols must be 1, 2, or 4."

    matmul_kernel = kernels.mm(
        dim_m=m,
        dim_k=k,
        dim_n=n,
        input_dtype=dtype_in,
        output_dtype=dtype_out,
        vectorized=True,
    )
    zero_kernel = matmul_kernel.zero
    r, s, t = matmul_kernel.mac_dims

    # L3 host tensors are flat (the runtime sequence indexes them via BD
    # sizes/strides). Max-capacity sized so one xclbin serves many shapes.
    A_ty = np.ndarray[(A_elements,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(B_elements,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(C_elements,), np.dtype[dtype_out]]
    # A L2 buffer carries one (m x k) tile per compute row; per column a shim
    # feeds all n_aie_rows rows (A is broadcast to every column), split into
    # per-row L1 tiles.
    A_l2_ty = np.ndarray[(m * k * n_aie_rows,), np.dtype[dtype_in]]
    B_l2_ty = np.ndarray[(k * n,), np.dtype[dtype_in]]
    C_l2_ty = np.ndarray[(m * n * n_aie_rows,), np.dtype[dtype_out]]
    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]

    fifo_depth = 2

    # --- ObjectFifos (compile-time; one column-band per AIE column) -----------
    # The n_aie_cols columns band the output N dimension: column `col` computes
    # C[:, col-band] = A @ B[:, col-band]. A is broadcast to every column (one
    # A shim DMA per column feeding that column's n_aie_rows rows); B and C are
    # banded in N, so each column has its own B/C shim FIFOs. Names carry the
    # column index so the MLIR symbols stay unique.
    A_l3l2 = [None] * n_aie_cols
    A_l2l1_fifos = [None] * n_aie_cols  # [col][row]
    B_l3l2 = [None] * n_aie_cols
    B_l2l1 = [None] * n_aie_cols
    C_l2l3 = [None] * n_aie_cols
    C_l1l2_fifos = [None] * n_aie_cols  # [col][row]

    for col in range(n_aie_cols):
        A_l3l2[col] = ObjectFifo(A_l2_ty, name=f"A_L3L2_{col}", depth=fifo_depth)
        A_l2l1_fifos[col] = (
            A_l3l2[col]
            .cons()
            .split(
                [m * k * j for j in range(n_aie_rows)],
                obj_types=[A_l1_ty] * n_aie_rows,
                names=[f"A_L2L1_{col}_{row}" for row in range(n_aie_rows)],
                dims_to_stream=[
                    [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
                    for _ in range(n_aie_rows)
                ],
            )
        )

        B_l3l2[col] = ObjectFifo(B_l2_ty, name=f"B_L3L2_{col}", depth=fifo_depth)
        B_l2l1[col] = (
            B_l3l2[col]
            .cons()
            .forward(
                obj_type=B_l1_ty,
                name=f"B_L2L1_{col}",
                dims_to_stream=[(k // s, s * n), (n // t, t), (s, n), (t, 1)],
            )
        )

        C_l2l3[col] = ObjectFifo(
            C_l2_ty,
            name=f"C_L2L3_{col}",
            depth=fifo_depth,
            dims_to_stream=[(m // r, r * n), (r, t), (n // t, r * t), (t, 1)],
        )
        C_l1l2_fifos[col] = (
            C_l2l3[col]
            .prod()
            .join(
                [m * n * i for i in range(n_aie_rows)],
                obj_types=[C_l1_ty] * n_aie_rows,
                names=[f"C_L1L2_{col}_{row}" for row in range(n_aie_rows)],
                depths=[fifo_depth] * n_aie_rows,
            )
        )

    # --- Compute workers (compile-time) --------------------------------------
    # The core body produces one C tile per invocation. Worker wraps it in
    # while_true by default, so the number of tiles a core produces is driven
    # entirely by how many the host DMA feeds/drains (objectfifo acquire()
    # backpressure) -- NOT a compile-time count. That is what lets a single
    # xclbin serve runtime M/N: only the K reduction depth (K // k) is baked in.
    def core_fn(in_a, in_b, out_c, zero, matmul):
        elem_out = out_c.acquire(1)
        zero(elem_out)
        for _ in range_(int(K) // k):
            elem_in_a = in_a.acquire(1)
            elem_in_b = in_b.acquire(1)
            matmul(elem_in_a, elem_in_b, elem_out)
            in_a.release(1)
            in_b.release(1)
        out_c.release(1)

    workers = []
    for col in range(n_aie_cols):
        for row in range(n_aie_rows):
            workers.append(
                Worker(
                    core_fn,
                    [
                        A_l2l1_fifos[col][row].cons(),
                        B_l2l1[col].cons(),
                        C_l1l2_fifos[col][row].prod(),
                        zero_kernel,
                        matmul_kernel,
                    ],
                    stack_size=0xD00,
                )
            )

    # --- Runtime sequence: range_ + fill/drain, one body for both lowerings ---
    # The body's M/K/N are declared as inputs to Runtime(seq, [...]):
    # K is always constant; only M/N can remain runtime inputs.
    #   unbound: passed as np.int32 types -> runtime i32 block args, so the
    #                  scf.for survives to the EmitC path; one xclbin, many shapes.
    #   specialized: passed as Python ints -> folded arith.constant, so
    #                  the range_ bounds are constant, aie-unroll-runtime-sequence-
    #                  loops flattens the loops, and everything folds to the static
    #                  binary path.
    # The SAME seq body runs either way; only how M/K/N enter differs.
    m_ar = m * n_aie_rows

    def seq(A, B, C, M_val, K_val, N_val, A_prods, B_prods, C_conses):
        i32 = np_dtype_to_mlir_type(np.int32)
        i64 = np_dtype_to_mlir_type(np.int64)

        # dma_bd operand widths differ: sizes/strides are i64 (DynamicIndexList),
        # offset/len are i32. Provide M/K/N in both widths.
        M64 = arith.extsi(i64, M_val)
        K64 = arith.extsi(i64, K_val)
        N64 = arith.extsi(i64, N_val)

        # M/K/N and all tile dims are non-negative, so truncated (divsi) and floor
        # division agree. divsi is used explicitly (not Python //, which emits
        # arith.floordivsi) because the EmitC C++ TXN path lowers divsi but not
        # floordivsi -- the dynamic lowering needs plain integer division.
        def _divsi(val, d):
            return arith.divsi(val, arith.constant(d, i64))

        row_blocks = _divsi(M64, m_ar)  # C row-blocks (loop trip count)
        k_tiles = _divsi(K64, k)  # A/B d1 count
        # Each column owns 1/n_aie_cols of the N tiles (its own N-band). Requires
        # N % (n * n_aie_cols) == 0 (a runtime value; documented, not checkable).
        n_tiles_col = _divsi(N64, n * n_aie_cols)  # A/B d0 count per column
        # N elements per column band. At n_aie_cols == 1 this is just N, so col 0
        # stays byte-identical to the single-column design (no extra div/mul/add).
        N_per_col = (
            N_val if n_aie_cols == 1 else arith.trunci(i32, _divsi(N64, n_aie_cols))
        )

        for rb in range_(row_blocks):
            rb_i32 = arith.index_cast(rb, to=i32)  # offset operand is i32
            for col in range(n_aie_cols):
                # col 0's N-offset is a literal 0 (no add emitted), so its BDs
                # match the single-column design; col > 0 adds col * (N/n_aie_cols).
                rb_n_off = rb_i32 * (m_ar * N_val)
                if col == 0:
                    C_off, B_off = rb_n_off, 0
                else:
                    col_n_off = arith.constant(col, i32) * N_per_col
                    C_off, B_off = rb_n_off + col_n_off, col_n_off

                # One task group per (row-block, col): await C, free A and B.
                tg = TaskGroup()

                # C output (drained, waited):
                #   C_offset = rb * m_ar * N + col * (N / n_aie_cols)
                C_conses[col].drain(
                    C,
                    sizes=[1, n_tiles_col, m_ar, n],
                    strides=[m_ar * N64, n, N64, 1],
                    offset=C_off,
                    transfer_len=m_ar * N_per_col,  # m_ar * n * n_tiles_col
                    wait=True,
                    group=tg,
                )

                # A input (broadcast across columns; no col term):
                #   A_offset = rb * m_ar * K
                A_prods[col].fill(
                    A,
                    sizes=[n_tiles_col, k_tiles, m_ar, k],
                    strides=[0, k, K64, 1],
                    offset=rb_i32 * (m_ar * K_val),
                    transfer_len=m_ar * K_val,
                    group=tg,
                )

                # B input (banded in N): B_offset = col * (N / n_aie_cols)
                B_prods[col].fill(
                    B,
                    sizes=[n_tiles_col, k_tiles, k, n],
                    strides=[n, k * N64, N64, 1],
                    offset=B_off,
                    transfer_len=n * K_val,  # k * n * k_tiles
                    group=tg,
                )

                tg.finish()

    # dynamic -> the three runtime scalars (i32 block args); static -> the ints,
    # which fold to arith.constant so the range_ bounds are compile-time.
    # fifos the body drives, one prod/cons handle per column, passed as fn_args.
    A_prods = [f.prod() for f in A_l3l2]
    B_prods = [f.prod() for f in B_l3l2]
    C_conses = [f.cons() for f in C_l2l3]
    rt = Runtime(
        seq,
        [A_ty, B_ty, C_ty, M, K, N, A_prods, B_prods, C_conses],
    )

    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="Whole-array matmul (dynamic runtime seq)")
    add_compile_args(p, short_dev=None, with_emit_mlir=True)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    p.add_argument("--dtype_in", type=str, choices=["i16"], default="i16")
    p.add_argument("--dtype_out", type=str, choices=["i16", "i32"], default="i16")
    add_benchmark_args(p)
    return p


def _run_and_verify(opts):
    dtype_in = str_to_dtype(opts.dtype_in)
    dtype_out = str_to_dtype(opts.dtype_out)
    rng = np.random.default_rng(1726250518)
    info = np.iinfo(dtype_in)
    A_np = rng.integers(
        info.min // 4, info.max // 4, size=(opts.M, opts.K), dtype=dtype_in
    )
    B_np = rng.integers(
        info.min // 4, info.max // 4, size=(opts.K, opts.N), dtype=dtype_in
    )
    A_t = iron.tensor(A_np, dtype=dtype_in, device="npu")
    B_t = iron.tensor(B_np, dtype=dtype_in, device="npu")
    C_t = iron.zeros((opts.M, opts.N), dtype=dtype_out, device="npu")

    bench = run_iters(
        whole_array_dynamic.specialize(**_compile_kwargs(opts)),
        A_t,
        B_t,
        C_t,
        warmup=opts.warmup,
        iters=opts.iters,
    )
    expected = (A_np.astype(np.int64) @ B_np.astype(np.int64)).astype(dtype_out)
    actual = C_t.numpy().reshape(opts.M, opts.N)
    assert_close_with_benchmark(
        actual,
        expected,
        bench=bench,
        ops=2.0 * opts.M * opts.K * opts.N,
        fail_msg="output does not match A @ B",
        mismatch_indices=True,
    )


def _compile_kwargs(opts):
    return dict(
        A_elements=opts.M * opts.K,
        B_elements=opts.K * opts.N,
        C_elements=opts.M * opts.N,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
    )


def main():
    opts = _make_argparser().parse_args()
    run_design_cli(
        whole_array_dynamic,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
    )


if __name__ == "__main__":
    main()
