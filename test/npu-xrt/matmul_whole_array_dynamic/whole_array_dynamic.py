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

One design, two lowerings, selected by whether M/K/N are bound:

* **static** — pass Python ints for M/K/N. The ``range_`` bounds are constant, so
  ``aie-unroll-runtime-sequence-loops`` flattens the loops and everything folds
  to the same BDs the ``TensorTiler2D`` version emits (binary TXN path).
* **dynamic** — pass M/K/N as runtime ``i32`` arguments. The ``scf.for`` loops
  survive to the EmitC path (``--aie-npu-to-cpp``), so one xclbin runs many
  shapes; the C++ builder assembles the TXN per call.

The BD size/stride/offset formulas match the explicit-math form of the design
(see the module docstring in ``whole_array.py`` for the tiling picture).
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import (
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    kernels,
    str_to_dtype,
)
from aie.iron.controlflow import range_
from aie.iron.device import from_name
from aie.extras.dialects import arith
from aie.helpers.util import np_dtype_to_mlir_type
from aie.utils.hostruntime.argparse import add_benchmark_args, add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie.utils.verify import assert_close_with_benchmark
from aie.utils.benchmark import run_iters

# Fixed tiling constants (compile-time). M/K/N are the runtime problem dims.
N_AIE_ROWS = 4


def _device_for(dev_str, n_aie_cols):
    return from_name(dev_str, n_cols=n_aie_cols if dev_str == "npu" else None)


def _build_design(
    dev, M, K, N, m, k, n, n_aie_cols, dtype_in_str, dtype_out_str, dynamic=True
):
    """Build the whole-array matmul with a range_-based runtime sequence.

    dynamic=True  -> M/K/N are runtime i32 args (scf survives, EmitC path).
    dynamic=False -> M/K/N fold to constants (loops unroll, binary path).
    The same runtime-sequence body is used for both.
    """
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
    A_ty = np.ndarray[(M * K,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(K * N,), np.dtype[dtype_in]]
    C_ty = np.ndarray[(M * N,), np.dtype[dtype_out]]
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
        for _ in range_(K // k):
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
    # The body's M/K/N are declared as inputs to rt.sequence():
    #   dynamic=True : passed as np.int32 types -> runtime i32 block args, so the
    #                  scf.for survives to the EmitC path; one xclbin, many shapes.
    #   dynamic=False: passed as the Python ints M/K/N -> folded arith.constant, so
    #                  the range_ bounds are constant, aie-unroll-runtime-sequence-
    #                  loops flattens the loops, and everything folds to the static
    #                  binary path.
    # The SAME seq body runs either way; only how M/K/N enter differs.
    m_ar = m * n_aie_rows

    def seq(A, B, C, M_val, K_val, N_val):
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
                C_l2l3[col].cons().drain(
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
                A_l3l2[col].prod().fill(
                    A,
                    sizes=[n_tiles_col, k_tiles, m_ar, k],
                    strides=[0, k, K64, 1],
                    offset=rb_i32 * (m_ar * K_val),
                    transfer_len=m_ar * K_val,
                    group=tg,
                )

                # B input (banded in N): B_offset = col * (N / n_aie_cols)
                B_l3l2[col].prod().fill(
                    B,
                    sizes=[n_tiles_col, k_tiles, k, n],
                    strides=[n, k * N64, N64, 1],
                    offset=B_off,
                    transfer_len=n * K_val,  # k * n * k_tiles
                    group=tg,
                )

                tg.finish()

    rt = Runtime()
    # dynamic -> declare M/K/N as runtime i32 types; static -> pass the ints.
    mkn = [np.int32, np.int32, np.int32] if dynamic else [M, K, N]
    rt.sequence(seq, [A_ty, B_ty, C_ty, *mkn])

    return Program(dev, rt, workers=workers).resolve_program()


# Static @iron.jit entry: M/K/N are CompileTime, so the range_ loops unroll and
# the design compiles to the binary path (recompiles per shape). This is the
# runnable-on-HW form today; the dynamic (runtime-M/K/N) form lowers to a C++
# TXN builder (verified at the MLIR level) and awaits an end-to-end dynamic
# dispatch driver.
from aie.iron import CompileTime, In, Out  # noqa: E402


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def whole_array_dynamic(
    A: In,
    B: In,
    C: Out,
    *,
    M: CompileTime[int],
    K: CompileTime[int],
    N: CompileTime[int],
    m: CompileTime[int],
    k: CompileTime[int],
    n: CompileTime[int],
    n_aie_cols: CompileTime[int],
    dtype_in_str: CompileTime[str],
    dtype_out_str: CompileTime[str],
):
    return _build_design(
        iron.get_current_device(),
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        dynamic=False,
    )


def _make_argparser():
    p = argparse.ArgumentParser(prog="Whole-array matmul (dynamic runtime seq)")
    add_compile_args(p, short_dev=None)
    p.add_argument("-M", type=int, default=512)
    p.add_argument("-K", type=int, default=512)
    p.add_argument("-N", type=int, default=512)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4], default=4)
    p.add_argument("--dtype_in", type=str, choices=["i16"], default="i16")
    p.add_argument("--dtype_out", type=str, choices=["i16", "i32"], default="i16")
    p.add_argument(
        "-o",
        "--emit-dynamic-mlir",
        type=str,
        default=None,
        help="Write the dynamic (runtime M/K/N) MLIR to this file and exit. "
        "aiecc consumes it to build the xclbin for end-to-end HW dispatch "
        "(the C++ TXN builder path, no @iron.jit).",
    )
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
        whole_array_dynamic,
        A_t,
        B_t,
        C_t,
        M=opts.M,
        K=opts.K,
        N=opts.N,
        m=opts.m,
        k=opts.k,
        n=opts.n,
        n_aie_cols=opts.n_aie_cols,
        dtype_in_str=opts.dtype_in,
        dtype_out_str=opts.dtype_out,
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
    if opts.emit_dynamic_mlir:
        dev = _device_for(opts.dev, opts.n_aie_cols)
        # Set the device so kernels.mm() resolves the correct arch (aie2p for
        # npu2); otherwise arch detection falls back to aie2 and the emitted
        # link_with names a kernel .o built for the wrong architecture.
        iron.set_current_device(dev)
        module = _build_design(
            dev,
            opts.M,
            opts.K,
            opts.N,
            opts.m,
            opts.k,
            opts.n,
            opts.n_aie_cols,
            opts.dtype_in,
            opts.dtype_out,
            dynamic=True,
        )
        with open(opts.emit_dynamic_mlir, "w") as f:
            f.write(str(module))
        return
    run_design_cli(
        whole_array_dynamic,
        opts,
        compile_kwargs=_compile_kwargs,
        run_and_verify=_run_and_verify,
        device=lambda o: _device_for(o.dev, o.n_aie_cols),
    )


if __name__ == "__main__":
    main()
