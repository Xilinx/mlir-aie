# test_q4nx_dequant.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""q4nx nibble order, geometry, narrowing and BFP storage (no NPU required)."""

import math
from pathlib import Path

import numpy as np
import pytest
from aie.iron import In, Out, kernels
from aie.iron.device import NPU1Col1
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16


def _pack(scales, mins, nibbles):
    """Independent writer: matrices are [n, k], parameters are [group, n]."""
    m, k = nibbles.shape
    params = np.concatenate([scales.ravel(), mins.ravel()]).astype(bfloat16)
    result = bytearray(params.view(np.uint16).astype("<u2").tobytes())
    for row in range(0, m, 16):
        for col in range(k):
            for n in range(row, row + 16, 2):
                result.append(int(nibbles[n, col]) | (int(nibbles[n + 1, col]) << 4))
    return np.frombuffer(result, dtype=np.uint8)


def _oracle(scales, mins, nibbles, *, group, ct_k):
    """Scalar arithmetic and block traversal, independent of the vector codec."""
    m, k = nibbles.shape
    out = []
    for ks in range(0, k, ct_k):
        for ns in range(0, m, 8):
            for step in range(ks, ks + ct_k, 8):
                for n in range(ns, ns + 8):
                    values = []
                    for col in range(step, step + 8):
                        x = np.float32(
                            float(scales[col // group, n]) * int(nibbles[n, col])
                            + float(mins[col // group, n])
                        )
                        nearest = bfloat16(x)
                        if float(nearest) > float(x):
                            nearest = np.nextafter(nearest, bfloat16(-np.inf))
                        values.append(float(nearest))
                    exponent = max(
                        int(np.float32(v).view(np.uint32) >> 23) & 255 for v in values
                    )
                    out.append(exponent)
                    out.extend(
                        math.floor(math.ldexp(v, 133 - exponent)) & 255 for v in values
                    )
    return np.array(out, dtype=np.uint8)


def test_low_nibble_first_and_exponents_shared_across_k():
    q = np.array([np.arange(8) + 8 * (n % 2) for n in range(16)], dtype=np.uint8)
    payload = _pack(np.ones((1, 16)), np.zeros((1, 16)), q)
    assert payload[64] == 0x80
    out = kernels.q4nx_dequant_ref(
        payload, m_tile=16, k_tile=8, group=8, ct_k=8
    ).reshape(16, 9)
    for n in range(16):
        assert out[n, 0] == (130 if n % 2 else 129)
        np.testing.assert_array_equal(out[n, 1:], q[n] * (8 if n % 2 else 16))


def test_literal_default_payload():
    payload = np.array([0x80, 0x3F] * 256 + [0] * 512 + [0x21] * 4096, np.uint8)
    expected = np.array(([0x7F] + [0x40] * 8 + [0x80] + [0x40] * 8) * 512, np.uint8)
    np.testing.assert_array_equal(kernels.q4nx_dequant_ref(payload), expected)


@pytest.mark.parametrize(
    "m,k,group,ct_k",
    [(16, 8, 8, 8), (32, 256, 32, 128), (48, 96, 24, 32), (16, 48, 16, 24)],
)
def test_group_row_slice_and_batch_layout(m, k, group, ct_k):
    rng = np.random.default_rng(6)
    payloads, expected = [], []
    for _ in range(2):
        scales = rng.uniform(-2, 2, (k // group, m)).astype(bfloat16)
        mins = rng.uniform(-8, 8, (k // group, m)).astype(bfloat16)
        q = rng.integers(0, 16, (m, k), dtype=np.uint8)
        payloads.append(_pack(scales, mins, q))
        expected.append(_oracle(scales, mins, q, group=group, ct_k=ct_k))
    payloads = np.array(payloads)
    # Accept non-contiguous host views without reinterpreting their strides.
    storage = np.empty((2, 2 * payloads.shape[1]), dtype=np.uint8)
    storage[:, ::2] = payloads
    actual = kernels.q4nx_dequant_ref(
        storage[:, ::2], m_tile=m, k_tile=k, group=group, ct_k=ct_k
    )
    assert actual.dtype == np.uint8
    assert actual.shape == (2, m * k * 9 // 8)
    np.testing.assert_array_equal(actual, expected)


def test_bf16_floor_before_bfp_rounding_for_both_signs():
    scales = np.full((1, 16), 1 / 256, dtype=bfloat16)
    mins = np.ones((1, 16), dtype=bfloat16)
    q = np.full((16, 8), 3, dtype=np.uint8)
    scales[0, 1] = -1 / 256
    mins[0, 1] = -1
    q[1] = 1
    actual = kernels.q4nx_dequant_ref(
        _pack(scales, mins, q), m_tile=16, k_tile=8, group=8, ct_k=8
    ).reshape(16, 9)
    np.testing.assert_array_equal(actual[0], [127] + [64] * 8)
    np.testing.assert_array_equal(actual[1], [127] + [191] * 8)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(m_tile=0), "m_tile"),
        (dict(k_tile=-8), "k_tile"),
        (dict(group=0), "group"),
        (dict(ct_k=0), "ct_k"),
        (dict(m_tile=True), "m_tile"),
        (dict(k_tile=256.0), "k_tile"),
        (dict(group="32"), "group"),
        (dict(s=4), "s and t"),
        (dict(t=16), "s and t"),
        (dict(m_tile=24), "m_tile"),
        (dict(ct_k=12), "ct_k"),
        (dict(group=12), "group"),
        (dict(ct_k=24), "k_tile"),
        (dict(group=24), "k_tile"),
        (dict(m_tile=2**30), "int offsets"),
    ],
)
def test_bad_geometry_is_rejected_by_factory_and_reference(kwargs, match):
    with pytest.raises(ValueError, match=match):
        kernels.q4nx_dequant(**kwargs)
    with pytest.raises(ValueError, match=match):
        kernels.q4nx_dequant_ref(np.zeros(1, np.uint8), **kwargs)


def test_payload_size_and_nonfinite_parameters():
    geometry = dict(m_tile=16, k_tile=8, group=8, ct_k=8)
    for bad in (np.uint8(0), np.zeros(127, np.uint8), np.zeros((2, 129), np.uint8)):
        with pytest.raises(ValueError, match="128 bytes"):
            kernels.q4nx_dequant_ref(bad, **geometry)
    q = np.zeros((16, 8), dtype=np.uint8)
    for param in ("scale", "min"):
        scales, mins = np.ones((1, 16)), np.zeros((1, 16))
        (scales if param == "scale" else mins)[0, 0] = np.inf
        with pytest.raises(ValueError, match="scales and minima must be finite"):
            kernels.q4nx_dequant_ref(_pack(scales, mins, q), **geometry)


def test_empty_batch_and_overflow():
    geometry = dict(m_tile=np.int32(16), k_tile=8, group=8, ct_k=8)
    empty = kernels.q4nx_dequant_ref(np.empty((0, 128), np.uint8), **geometry)
    assert empty.shape == (0, 144)
    scales = np.full((1, 16), 2.0**125, dtype=bfloat16)
    payload = _pack(scales, np.zeros((1, 16)), np.full((16, 8), 15, np.uint8))
    with pytest.raises(ValueError, match="dequantized bf16 values must be finite"):
        kernels.q4nx_dequant_ref(payload, **geometry)


def test_factory_metadata_and_finite_sample(npu2_device):
    fn = kernels.q4nx_dequant()
    assert fn._original_name == "q4nx_dequant_bfp"
    assert fn._name.endswith("_q4nx_dequant_bfp")
    assert Path(fn._source_file).name == "q4nx_dequant.cc"
    assert Path(fn._source_file).parent.name == "quant"
    assert fn.arg_shape(0) == (5120,)
    assert fn.arg_shape(1) == (9216,)
    assert fn.arg_dtype(0) == fn.arg_dtype(1) == np.uint8
    for name, value in dict(
        M_TILE=32, K_TILE=256, GROUP=32, CT_K=128, S=8, T=8
    ).items():
        assert f"-DQ4NX_{name}={value}" in fn._compile_flags
    contract = fn.contract
    assert contract.roles == (In, Out)
    assert contract.setup is None
    assert contract.unsupported is None
    assert contract.tolerance.kind == "exact"
    assert contract.tolerance.max_mismatch_frac == 0
    assert contract.ops_per_call == 16384
    assert contract.acc_dtype == np.float32
    assert contract.reduction == 1
    assert contract.stack_bytes is None
    sample = contract.sample(np.random.default_rng(8), 3)
    assert len(sample) == 1
    assert sample[0].shape == (3, 5120)
    assert sample[0].dtype == np.uint8
    params = sample[0][:, :1024].copy().view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    assert np.isfinite(params).all()
    assert (params < 0).any() and (params > 0).any()
    expected = contract.reference(*sample)
    assert expected.shape == (3, 9216) and expected.dtype == np.uint8
    np.testing.assert_array_equal(expected, kernels.q4nx_dequant_ref(sample[0]))
    assert fn == kernels.q4nx_dequant()


def test_nondefault_factory_and_architecture(npu2_device):
    fn = kernels.q4nx_dequant(m_tile=48, k_tile=96, group=24, ct_k=32)
    assert fn.arg_shape(0) == (3072,)
    assert fn.arg_shape(1) == (5184,)
    sample = fn.contract.sample(np.random.default_rng(9), 2)
    assert sample[0].shape == (2, 3072)
    assert fn.contract.reference(*sample).shape == (2, 5184)
    set_current_device(NPU1Col1())
    with pytest.raises(NotImplementedError, match="only available on aie2p"):
        kernels.q4nx_dequant()
