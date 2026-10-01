# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Numeric gates for softmax.cc's batched entry points on AIE2P.

``softmax_rows_bf16`` runs ``softmax_bf16`` over each row of a block, and
``softmax_rows_causal_bf16`` first masks every key past the row's own query
position, the first row being query ``row_offset``. Both are held to
``kernels.softmax``'s tolerance at the row length.
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
    jit,
    kernels,
)
from aie.utils.compile.utils import resolve_target_arch
from aie.utils.verify import nearly_equal

_ROWS, _ROW_LEN = 8, 64


@jit
def softmax_rows(
    x_in: In,
    y_out: Out,
    *,
    causal: CompileTime[bool] = False,
    row_offset: CompileTime[int] = 0,
):
    """One call of a batched entry point over a ``_ROWS`` x ``_ROW_LEN`` block."""
    block_ty = np.ndarray[(_ROWS * _ROW_LEN,), np.dtype[bfloat16]]
    scalars = [np.int32] * (3 if causal else 2)
    symbol = "softmax_rows_causal_bf16" if causal else "softmax_rows_bf16"
    softmax = kernels.softmax(_ROW_LEN)
    fn = softmax.object_file.bind(symbol, [block_ty, block_ty, *scalars])

    of_x = ObjectFifo(block_ty, name="x", depth=1)
    of_y = ObjectFifo(block_ty, name="y", depth=1)

    def core(of_x, of_y, fn):
        args = (_ROWS, _ROW_LEN, row_offset) if causal else (_ROWS, _ROW_LEN)
        fn(of_x.acquire(1), of_y.acquire(1), *args)
        of_x.release(1)
        of_y.release(1)

    worker = Worker(core, fn_args=[of_x.cons(), of_y.prod(), fn])

    def sequence(x_h, y_h, x_fifo, y_fifo):
        x_fifo.fill(x_h)
        y_fifo.drain(y_h, wait=True)

    rt = Runtime(sequence, [block_ty, block_ty, of_x.prod(), of_y.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _reference(x, causal, row_offset):
    """Row-wise softmax, over each row's keys up to its own position when causal."""
    scores = x.astype(np.float32)
    if causal:
        keys = np.arange(_ROW_LEN)[None, :]
        rows = row_offset + np.arange(_ROWS)[:, None]
        scores = np.where(keys <= rows, scores, -np.inf)
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@pytest.mark.parametrize(
    "causal,row_offset",
    [
        (False, 0),  # every row whole
        (True, 0),  # the diagonal of the first block
        (True, 20),  # rows starting mid-sequence, the mask mid-vector
        (True, 60),  # the last rows, whose mask runs past the row
    ],
)
def test_softmax_rows_match_row_wise_softmax(causal, row_offset):
    if resolve_target_arch(iron.get_current_device()) == "aie2":
        pytest.skip("the batched entry points are AIE2P's")
    rng = np.random.default_rng(20260930 + row_offset)
    x = rng.uniform(-4, 4, (_ROWS, _ROW_LEN)).astype(bfloat16)

    y_t = iron.zeros((_ROWS * _ROW_LEN,), dtype=bfloat16)
    softmax_rows(
        iron.tensor(x.reshape(-1), dtype=bfloat16),
        y_t,
        causal=causal,
        row_offset=row_offset,
    )
    got = y_t.numpy().astype(np.float32).reshape(_ROWS, _ROW_LEN)

    tol = kernels.softmax(_ROW_LEN).contract.tolerance
    assert tol.rtol is not None and tol.atol is not None
    ref = _reference(x, causal, row_offset)
    assert nearly_equal(got, ref, rtol=tol.rtol, atol=tol.atol).all()
