# test_attention_merge.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""attention_merge's factory and reference (no NPU required): windows of keys,
each attended alone as mha.cc's online softmax leaves it and merged one per
call, are attention over every key so far.
"""

import ml_dtypes
import numpy as np
import pytest
from aie.iron.kernels import attention_merge, attention_merge_ref
from aie.iron.kernels.linalg import attention_merge_bound
from ml_dtypes import bfloat16

ROWS, DIM, WINDOWS, KEYS = 32, 64, 4, 64


def test_each_call_is_attention_over_every_key_so_far():
    rng = np.random.default_rng(0)
    scale = float(bfloat16(np.log2(np.e) / np.sqrt(DIM)))
    q = rng.standard_normal((ROWS, DIM))
    k = rng.standard_normal((WINDOWS * KEYS, DIM))
    v = rng.standard_normal((WINDOWS * KEYS, DIM))
    a = q @ k.T
    o = np.empty((WINDOWS, ROWS, DIM))
    state = np.empty((WINDOWS, 2, ROWS))
    for j in range(WINDOWS):
        keys = np.s_[j * KEYS : (j + 1) * KEYS]
        m = a[:, keys].max(axis=1)
        p = np.exp2((a[:, keys] - m[:, None]) * scale)
        state[j] = m, p.sum(axis=1)
        o[j] = p @ v[keys] / state[j, 1][:, None]
    got = attention_merge_ref(
        o.reshape(WINDOWS, -1), state.reshape(WINDOWS, -1), scale, rows=ROWS, dim=DIM
    )
    for j in range(WINDOWS):
        seen = a[:, : (j + 1) * KEYS]
        p = np.exp2((seen - seen.max(axis=1, keepdims=True)) * scale)
        whole = p @ v[: (j + 1) * KEYS] / p.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(got[j].reshape(ROWS, DIM), whole, rtol=1e-12)


def test_a_row_no_window_kept_is_zero_within_a_zero_bound():
    o = np.ones((WINDOWS, ROWS * DIM))
    state = np.zeros((WINDOWS, 2, ROWS))
    state[:, 0] = float(ml_dtypes.finfo(bfloat16).min)
    state = state.reshape(WINDOWS, -1)
    shape = dict(rows=ROWS, dim=DIM)
    assert not attention_merge_ref(o, state, 0.5, **shape).any()
    assert np.all(attention_merge_bound(o, state, 0.5, **shape) <= 2.0**-126)


@pytest.mark.parametrize(
    "shape, match",
    [
        (dict(rows=48), "rows must be a positive multiple of 32"),
        (dict(dim=16), "dim must be a positive multiple of 32"),
    ],
)
def test_the_factory_refuses_shapes_its_vectors_do_not_tile(shape, match):
    with pytest.raises(ValueError, match=match):
        attention_merge(**shape)
