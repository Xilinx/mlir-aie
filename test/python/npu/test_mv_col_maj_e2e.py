# test_mv_col_maj_e2e.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings

"""The column-major bf16 matvec returns the row-major one's bits.

``mv_bf16.cc`` built with ``-DA_COL_MAJ`` reads A stored ``(K, M)`` and sums
in the row-major kernel's order, so the two are layouts of one kernel rather
than two approximations of ``A @ b``. Each case runs both on the device, the
row-major kernel four rows a call over the whole ``K`` and the column-major
one over ``A.T`` in ``chunk``-row calls that carry their sums in ``acc``, and
compares the bits. The comparison is device against device: aie2p's float
adder is not IEEE, so no host model reproduces either kernel's bits.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.iron import CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.iron.buffer import Buffer
from ml_dtypes import bfloat16


@iron.jit
def _col_maj_design(
    a_in: In,
    b_in: In,
    c_out: Out,
    *,
    dim_m: CompileTime[int],
    dim_k: CompileTime[int],
    chunk: CompileTime[int],
    vec_size: CompileTime[int],
):
    kernel = kernels.mv_col_maj(dim_m, chunk, vec_size=vec_size)
    _, a_ty, b_ty, acc_ty, c_ty = kernel.arg_types()
    of_a = ObjectFifo(a_ty, name="a")
    of_b = ObjectFifo(b_ty, name="b")
    of_c = ObjectFifo(c_ty, name="c")
    # NaN until the FIRST call overwrites it: a kernel that read the sums it
    # was meant to start would return NaN.
    acc = Buffer(
        acc_ty, name="acc", initial_value=np.full(vec_size * dim_m, np.nan, np.float32)
    )
    calls = dim_k // chunk

    def core(of_a, of_b, of_c, acc, kernel):
        c = of_c.acquire(1)
        for i in range(calls):
            flags = (kernels.MV_COL_MAJ_FIRST if i == 0 else 0) | (
                kernels.MV_COL_MAJ_LAST if i == calls - 1 else 0
            )
            a, b = of_a.acquire(1), of_b.acquire(1)
            kernel(flags, a, b, acc, c)
            of_a.release(1)
            of_b.release(1)
        of_c.release(1)

    worker = Worker(core, [of_a.cons(), of_b.cons(), of_c.prod(), acc, kernel])
    a_host = np.ndarray[(dim_k * dim_m,), np.dtype[bfloat16]]
    b_host = np.ndarray[(dim_k,), np.dtype[bfloat16]]

    def seq(a, b, c, h_a, h_b, h_c):
        h_a.fill(a)
        h_b.fill(b)
        h_c.drain(c, wait=True)

    rt = Runtime(seq, [a_host, b_host, c_ty, of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _row_maj(a, b, vec_size):
    """The row-major kernel over ``a`` (M, K), four rows a call."""
    dim_m, dim_k = a.shape
    calls = dim_m // 4
    fn = kernels.mv(
        dim_m=4,
        dim_k=dim_k,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        vec_size=vec_size,
    )
    design = kd.design(
        kernels.mv,
        calls=calls,
        dim_m=4,
        dim_k=dim_k,
        input_dtype=bfloat16,
        output_dtype=bfloat16,
        vec_size=vec_size,
    )
    inputs = [a.reshape(calls, 4, dim_k), np.broadcast_to(b, (calls, dim_k))]
    ins, out = kd.upload(
        inputs,
        kd.output_size(fn, calls=calls),
        fn.output_dtype(),
        fn=fn,
        poison=True,
    )
    design(*ins, out)
    return out.numpy().copy()


def _col_maj(a_t, b, chunk, vec_size):
    """The column-major kernel over ``a_t`` (K, M), ``chunk`` rows a call."""
    dim_k, dim_m = a_t.shape
    a = iron.tensor(np.ascontiguousarray(a_t).ravel(), dtype=bfloat16, device="npu")
    b = iron.tensor(b, dtype=bfloat16, device="npu")
    c = iron.tensor(np.full(dim_m, np.nan, bfloat16), dtype=bfloat16, device="npu")
    _col_maj_design(a, b, c, dim_m=dim_m, dim_k=dim_k, chunk=chunk, vec_size=vec_size)
    return c.numpy().copy()


# (vec_size, M, K, chunk). The first is llama 3.2 1B's attention context: 64
# head dims over 2048 cached positions, 128 a call. The rest cover each
# VEC_SIZE, a single FIRST|LAST call, and two 64-wide output blocks per row.
# Below 64 lanes K stays within four vectors: past that the row-major kernel
# runs its mac loop, whose rows past the first come out wrong at 16 and 32
# lanes, so there is nothing right to be bit-identical to.
_SHAPES = [
    (64, 64, 2048, 128),
    (64, 64, 128, 128),
    (64, 64, 512, 64),
    (32, 32, 128, 32),
    (16, 16, 64, 16),
    (32, 128, 128, 32),
]


@pytest.mark.parametrize(
    "vec_size, dim_m, dim_k, chunk",
    _SHAPES,
    ids=[f"r{r}-m{m}-k{k}-chunk{c}" for r, m, k, c in _SHAPES],
)
def test_col_maj_matvec_returns_the_row_major_bits(vec_size, dim_m, dim_k, chunk):
    rng = np.random.default_rng(dim_m * dim_k + vec_size)
    # Signed terms over a wide range of exponents, so sums cancel and an
    # order the kernels did not share would round differently.
    scale = np.exp2(rng.integers(-6, 7, size=(dim_m, dim_k)))
    a = (rng.standard_normal((dim_m, dim_k)) * scale).astype(bfloat16)
    b = rng.standard_normal(dim_k).astype(bfloat16)
    row_maj = _row_maj(a, b, vec_size)
    col_maj = _col_maj(a.T, b, chunk, vec_size)
    expected = a.astype(np.float32) @ b.astype(np.float32)
    # Not a tolerance test: this only proves both computed A @ b...
    np.testing.assert_allclose(
        row_maj.astype(np.float32),
        expected,
        rtol=0.05,
        atol=0.05 * np.abs(expected).max(),
    )
    # ...and this is the claim.
    np.testing.assert_array_equal(col_maj.view(np.uint16), row_maj.view(np.uint16))
