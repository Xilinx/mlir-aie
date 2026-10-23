# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# RUN: %run_on_npu2_hrx% %pytest %s
# REQUIRES: xrt_python_bindings || hrx_python_bindings
"""Numeric gate for softmax.cc's ``softmax_rows_banded_f32_bf16``: a row's
mask computed from ``window`` and its own query position (``row_offset + r``,
wrapped every ``period`` rows), not read from memory, as EG2 audio
attention's windowed-and-relative softmax needs (S21a, pr222/s21a-mask.md).
"""

import numpy as np
import pytest
from ml_dtypes import bfloat16, finfo

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
from aie.iron.kernels.activation import softmax_f32_bound, softmax_ref

_ROWS, _ROW_LEN = 16, 32


@jit
def softmax_rows_banded(
    x_in: In,
    addend_in: In,
    y_out: Out,
    *,
    window: CompileTime[int],
    period: CompileTime[int],
    row_offset: CompileTime[int],
    soft_cap: CompileTime[float],
):
    """One call of ``softmax_rows_banded_f32_bf16`` over a ``_ROWS`` x
    ``_ROW_LEN`` block.
    """
    f32_ty = np.ndarray[(_ROWS * _ROW_LEN,), np.dtype[np.float32]]
    bf16_ty = np.ndarray[(_ROWS * _ROW_LEN,), np.dtype[bfloat16]]
    softmax = kernels.softmax(
        _ROW_LEN, input_dtype=np.float32, soft_cap=soft_cap, addend=True
    )
    fn = softmax.object_file.bind(
        "softmax_rows_banded_f32_bf16",
        [f32_ty, f32_ty, bf16_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    of_x = ObjectFifo(f32_ty, name="x", depth=1)
    of_addend = ObjectFifo(f32_ty, name="addend", depth=1)
    of_y = ObjectFifo(bf16_ty, name="y", depth=1)

    def core(of_x, of_addend, of_y, fn):
        fn(
            of_x.acquire(1),
            of_addend.acquire(1),
            of_y.acquire(1),
            _ROWS,
            _ROW_LEN,
            window,
            period,
            row_offset,
        )
        of_x.release(1)
        of_addend.release(1)
        of_y.release(1)

    worker = Worker(core, fn_args=[of_x.cons(), of_addend.cons(), of_y.prod(), fn])

    def sequence(x_h, addend_h, y_h, x_fifo, addend_fifo, y_fifo):
        x_fifo.fill(x_h)
        addend_fifo.fill(addend_h)
        y_fifo.drain(y_h, wait=True)

    rt = Runtime(
        sequence,
        [f32_ty, f32_ty, bf16_ty, of_x.prod(), of_addend.prod(), of_y.cons()],
    )
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def _band(window, period, row_offset, rows):
    """``[lo, hi)`` a row keeps, ``audio.py``'s ``self.mask`` restated as
    bounds: the same ``np.where`` the operator's docstring derives from.
    """
    q = (row_offset + np.arange(rows)) % period
    parity = q % 2
    lo = np.maximum(parity, window - 1 - q + parity)
    hi = window + parity
    return lo, hi


def _row_mask(window, period, row_offset, rows, row_len):
    """The band restated as the explicit bf16 mask ``softmax_ref`` and
    ``softmax_f32_bound`` (the factory's own reference and tolerance) take,
    so this test is held to the same contract the masked path is.
    """
    lo, hi = _band(window, period, row_offset, rows)
    cols = np.arange(row_len)[None, :]
    band = (cols >= lo[:, None]) & (cols < hi[:, None])
    return np.where(band, bfloat16(0.0), finfo(bfloat16).min)


@pytest.mark.parametrize(
    "window,period,row_offset",
    [
        (12, 16, 0),  # period divides the block exactly, no wrap mid-block
        (12, 768, 0),  # EG2's own window and period, the sequence's start
        (12, 768, 390),  # EG2's shape, well past the start-of-sequence effect
        (12, 11, 3),  # period smaller than the block: wraps more than once
    ],
)
def test_softmax_rows_banded_matches_windowed_softmax(window, period, row_offset):
    cap = 50.0
    rng = np.random.default_rng(20261014 + row_offset + period)
    x = rng.uniform(-80, 80, (_ROWS, _ROW_LEN)).astype(np.float32)
    addend = rng.uniform(-80, 80, (_ROWS, _ROW_LEN)).astype(np.float32)

    y_t = iron.zeros((_ROWS * _ROW_LEN,), dtype=bfloat16)
    softmax_rows_banded(
        iron.tensor(x.reshape(-1), dtype=np.float32),
        iron.tensor(addend.reshape(-1), dtype=np.float32),
        y_t,
        window=window,
        period=period,
        row_offset=row_offset,
        soft_cap=cap,
    )
    got = y_t.numpy().astype(np.float32).reshape(_ROWS, _ROW_LEN)

    mask = _row_mask(window, period, row_offset, _ROWS, _ROW_LEN)
    ref = softmax_ref(x, addend, mask, tile_size=_ROW_LEN, soft_cap=cap)
    bound = softmax_f32_bound(x, addend, mask, tile_size=_ROW_LEN, soft_cap=cap)
    assert np.all(np.abs(got - ref) <= bound), (got - ref, bound)
