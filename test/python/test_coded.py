# test_coded.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""aie.utils.coded: the coded-row bf16 weight format (no NPU)."""

import numpy as np
import pytest
from aie.utils import coded
from ml_dtypes import bfloat16


def _bf16(values):
    return np.asarray(values, dtype=np.float32).astype(bfloat16)


def _row_of(hi_values, lo_values=None):
    """A single bf16 row built directly from high/low bytes (bypassing float rounding)."""
    hi = np.asarray(hi_values, dtype=np.uint16)
    lo = (
        np.zeros_like(hi)
        if lo_values is None
        else np.asarray(lo_values, dtype=np.uint16)
    )
    bits = (hi << 8) | lo
    return bits.view(bfloat16).reshape(1, -1)


def test_row_bytes_matches_the_design_tiers():
    assert coded.row_bytes(2048, capacity=15) == 3136
    assert coded.row_bytes(2048, capacity=31) == 3200
    assert coded.row_bytes(2048, capacity=63) == 3328
    # K % 128 != 0 pads the code plane to the next 64-byte boundary.
    assert coded.row_bytes(64, capacity=15) == 192
    with pytest.raises(ValueError):
        coded.row_bytes(63)
    with pytest.raises(ValueError):
        coded.row_bytes(0)


def test_round_trip_over_every_bf16_bit_pattern():
    # Reshaping the full 16-bit sweep into 64-wide rows keeps every row's
    # high byte constant (256 values / 64 = 4 rows per high byte), so this
    # exercises NaN payloads, Inf, -0 and subnormals with zero exceptions.
    bits = np.arange(65536, dtype=np.uint16)
    w = bits.view(bfloat16).reshape(1024, 64)
    back = coded.decode_rows(coded.encode_rows(w), 64)
    np.testing.assert_array_equal(back.view(np.uint16), bits.reshape(1024, 64))


@pytest.mark.parametrize("K", [64, 128, 8192])
def test_round_trip_random_rows_at_several_k(K):
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((5, K)) * 4).astype(bfloat16)
    coded_rows = coded.encode_rows(w)
    assert coded_rows.shape == (5, coded.row_bytes(K))
    back = coded.decode_rows(coded_rows, K)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))


def test_round_trip_preserves_leading_batch_dims():
    rng = np.random.default_rng(1)
    w = (rng.standard_normal((3, 2, 7, 128)) * 4).astype(bfloat16)
    coded_rows = coded.encode_rows(w)
    assert coded_rows.shape == (3, 2, 7, coded.row_bytes(128))
    back = coded.decode_rows(coded_rows, 128)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))


def test_chunking_crosses_a_chunk_boundary():
    # Real row counts past one internal chunk, not a patched chunk size, so
    # this exercises the loop in encode_rows/decode_rows that _CHUNK_BUDGET
    # otherwise hides from every other test here (K=64 rows all fit in one
    # chunk below this size).
    K = 64
    chunk = coded._chunk_rows(K)
    M = chunk + 5
    rng = np.random.default_rng(4)
    w = (rng.standard_normal((M, K)) * 4).astype(bfloat16)
    coded_rows = coded.encode_rows(w)
    back = coded.decode_rows(coded_rows, K)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))

    # An over-capacity row at the first local index of the second chunk
    # must still be named by its global row index.
    hi = np.concatenate([np.full(48, 60, np.uint16), np.arange(16, dtype=np.uint16)])
    bad_row = _row_of(hi)
    w2 = np.concatenate([w[:chunk], bad_row, w[chunk:]], axis=0)
    with pytest.raises(ValueError, match=rf"row {chunk} holds 16 exceptions"):
        coded.encode_rows(w2, capacity=15)


def test_row_with_exactly_15_exceptions_packs():
    # 49 elements share one high byte (the window covers them); 15 more each
    # hold a distinct, isolated high byte the windows cannot also cover.
    hi = np.concatenate([np.full(49, 60, np.uint16), np.arange(15, dtype=np.uint16)])
    w = _row_of(hi)
    out = coded.encode_rows(w, capacity=15)
    t = -(-(3 * 64 // 2) // 64) * 64
    assert out[0, t + 2] == 15
    np.testing.assert_array_equal(
        coded.decode_rows(out, 64).view(np.uint16), w.view(np.uint16)
    )


def test_row_needing_16_exceptions_raises_naming_the_row():
    hi = np.concatenate([np.full(48, 60, np.uint16), np.arange(16, dtype=np.uint16)])
    w = np.concatenate([_row_of(np.full(64, 60, np.uint16)), _row_of(hi)], axis=0)
    with pytest.raises(ValueError, match=r"row 1 holds 16 exceptions"):
        coded.encode_rows(w, capacity=15)


def test_single_exponent_row_round_trips():
    # Every element in [1, 2), so the exponent field is fixed and only the
    # mantissa (and the row's one sign) varies.
    rng = np.random.default_rng(2)
    w = (1.0 + rng.random(8192)).astype(bfloat16).reshape(1, 8192)
    back = coded.decode_rows(coded.encode_rows(w), 8192)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))


def test_alternating_signs_round_trip():
    rng = np.random.default_rng(3)
    mag = (rng.standard_normal(512) * 3).astype(bfloat16).astype(np.float32)
    signs = np.where(np.arange(512) % 2 == 0, 1.0, -1.0)
    w = (np.abs(mag) * signs).astype(bfloat16).reshape(1, 512)
    back = coded.decode_rows(coded.encode_rows(w), 512)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))


def test_encode_rejects_bad_inputs():
    with pytest.raises(ValueError):
        coded.encode_rows(np.zeros((2, 64), np.float32))
    with pytest.raises(ValueError):
        coded.encode_rows(_bf16(np.zeros(64)).reshape(64))
    with pytest.raises(ValueError):
        coded.encode_rows(_bf16(np.zeros((2, 63))))
    with pytest.raises(ValueError):
        coded.decode_rows(np.zeros((2, 192), np.uint8), 63)


def test_byte_layout_of_a_hand_computed_row():
    # 63 elements at hi=0x3F (1.0's bf16 high byte), lo=0x80; one outlier at
    # hi=0x00, lo=0x00, far from the cluster so no 8-wide window covers both.
    # The pos window search (counts[0]=1, counts[63]=63) ties over bases
    # 56..63 (all see the cluster's 63 hits); argmax keeps the lowest,
    # base_pos=56. The neg half is empty, so base_neg defaults to 128. The
    # outlier's high byte falls in neither window and becomes exception 0,
    # at column 63.
    hi = np.concatenate([np.full(63, 0x3F, np.uint16), [0x00]])
    lo = np.concatenate([np.full(63, 0x80, np.uint16), [0x00]])
    w = _row_of(hi, lo)
    out = coded.encode_rows(w, capacity=15)
    assert out.shape == (1, 192)

    expected = np.zeros(192, np.uint8)
    expected[:63] = 0x80  # lo plane, cluster
    expected[63] = 0x00  # lo plane, outlier
    expected[64:95] = 0x77  # code nibbles: both cluster (code 7 each)
    expected[95] = 0x07  # cluster (7), then the outlier's code 0 in the high nibble
    # 96..127 is alignment padding (K=64 is not a multiple of 128), left zero.
    expected[128] = 56  # base_pos
    expected[129] = 128  # base_neg
    expected[130] = 1  # n_exc
    expected[132] = 63  # exception 0: column index low byte
    expected[134] = 0x00  # exception 0: the outlier's true high byte
    np.testing.assert_array_equal(out[0], expected)

    back = coded.decode_rows(out, 64)
    np.testing.assert_array_equal(back.view(np.uint16), w.view(np.uint16))
