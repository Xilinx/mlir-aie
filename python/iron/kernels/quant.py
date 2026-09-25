# kernels/quant.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Packed quantization kernel factories and byte-exact host references."""

from functools import partial

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils import bfp
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, _default_source_path, _detect_arch, _make_extern


def _geometry(m_tile, k_tile, group, ct_k, s, t):
    geometry = dict(m_tile=m_tile, k_tile=k_tile, group=group, ct_k=ct_k, s=s, t=t)
    for name, value in geometry.items():
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
        ):
            raise ValueError(f"q4nx_dequant: {name} must be a positive integer")
    if s != 8 or t != 8:
        raise ValueError("q4nx_dequant: s and t must both be 8")
    if m_tile % 16:
        raise ValueError("q4nx_dequant: m_tile must be a multiple of 16")
    if ct_k % s:
        raise ValueError("q4nx_dequant: ct_k must be a multiple of s")
    if group % s:
        raise ValueError("q4nx_dequant: group must be a multiple of s")
    if k_tile % ct_k:
        raise ValueError("q4nx_dequant: k_tile must be a multiple of ct_k")
    if k_tile % group:
        raise ValueError("q4nx_dequant: k_tile must be a multiple of group")
    geometry = {name: int(value) for name, value in geometry.items()}
    if geometry["m_tile"] * geometry["k_tile"] * 9 // 8 > np.iinfo(np.int32).max:
        raise ValueError("q4nx_dequant: geometry exceeds the kernel's int offsets")
    return geometry


def _bf16_floor(x):
    """Narrow float32 toward minus infinity, not numpy's bf16 nearest-even."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    upper = bits >> 16
    upper += ((bits >> 31 != 0) & (bits & 0xFFFF != 0)).astype(np.uint32)
    return (upper << 16).view(np.float32)


def q4nx_dequant_ref(payload, *, m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8):
    """Dequantize packed q4nx to the kernel's bfp16ebs8 output bytes.

    Input has shape ``(..., m_tile*k_tile//2 + 4*m_tile*k_tile//group)``.
    It contains little-endian bf16 scales, then bf16 minima, both indexed
    ``[k_group, n]``, followed by unsigned nibbles (low nibble first) indexed
    ``[n//16, k, n%16]``. Values are ``min + scale * nibble``.

    Accumulator results narrow to bf16 with floor rounding before BFP
    conversion. Output is uint8 with the same leading dimensions, indexed
    ``[k//ct_k, n//8, (k%ct_k)//8, n%8, 9]``: each nine-byte block is an
    exponent followed by eight signed mantissas for consecutive k values.
    Finite scales/minima and finite dequantized bf16 results are required.
    """
    geometry = _geometry(m_tile, k_tile, group, ct_k, s, t)
    m_tile, k_tile, group, ct_k, s, t = geometry.values()
    scales_count = m_tile * k_tile // group
    input_bytes = m_tile * k_tile // 2 + 4 * scales_count
    payload = np.asarray(payload, dtype=np.uint8)
    if payload.ndim == 0 or payload.shape[-1] != input_bytes:
        raise ValueError(f"q4nx_dequant_ref: expected {input_bytes} bytes per tile")
    lead = payload.shape[:-1]
    data = payload.reshape(-1, input_bytes)
    if not len(data):
        return np.empty((*lead, m_tile * k_tile * 9 // 8), dtype=np.uint8)
    params = np.ascontiguousarray(data[:, : 4 * scales_count]).view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    if not np.isfinite(params).all():
        raise ValueError("q4nx_dequant_ref: scales and minima must be finite")
    scales, mins = params.reshape(-1, 2, k_tile // group, m_tile).transpose(1, 0, 2, 3)
    packed = data[:, 4 * scales_count :]
    q = np.empty((len(data), m_tile * k_tile), dtype=np.uint8)
    q[:, 0::2], q[:, 1::2] = packed & 15, packed >> 4
    q = q.reshape(-1, m_tile // 16, k_tile, 16).transpose(0, 2, 1, 3)
    q = q.reshape(-1, k_tile, m_tile)
    # A bf16 scale times a four-bit integer is exact in float32. Float64
    # models the fused addition before its single float32 accumulator rounding.
    with np.errstate(over="ignore", invalid="ignore"):
        values = (
            np.repeat(scales, group, axis=1).astype(np.float64) * q
            + np.repeat(mins, group, axis=1).astype(np.float64)
        ).astype(np.float32)
        values = _bf16_floor(values)
    if not np.isfinite(values).all():
        raise ValueError("q4nx_dequant_ref: dequantized bf16 values must be finite")
    values = values.reshape(-1, k_tile // ct_k, ct_k // s, s, m_tile // t, t)
    values = values.transpose(0, 1, 4, 2, 5, 3)
    encoded = bfp.encode(values.reshape(len(data), -1))
    return encoded.reshape(*lead, m_tile * k_tile * 9 // 8)


def _q4nx_sample(rng, calls, *, m_tile, k_tile, group, **_):
    count = m_tile * k_tile // group
    scales = rng.uniform(-2, 2, (calls, count)).astype(bfloat16)
    mins = rng.uniform(-8, 8, (calls, count)).astype(bfloat16)
    params = np.concatenate([scales, mins], axis=1)
    params = params.view(np.uint16).astype("<u2").view(np.uint8)
    packed = rng.integers(0, 256, (calls, m_tile * k_tile // 2), dtype=np.uint8)
    return [np.concatenate([params, packed], axis=1)]


def q4nx_dequant(
    *, m_tile=32, k_tile=256, group=32, ct_k=128, s=8, t=8
) -> ExternalFunction:
    """AIE2P-only q4nx dequantization into GEMM B-operand BFP storage.

    ``m_tile`` counts n rows (a multiple of 16); ``k_tile`` must be divisible
    by both ``group`` and ``ct_k``. The latter two are positive multiples of
    eight; ``s`` and ``t`` must be eight. Groups need not divide k slices.
    See ``q4nx_dequant_ref`` for the packed input and output layouts.

    Both arguments are uint8 byte buffers, including the bfp16ebs8 output,
    so the generic harness compares the complete encoded result byte for
    byte rather than decoding or quantizing it again. The kernel saves,
    selects and restores floor rounding itself; no setup kernel is needed.
    """
    geometry = _geometry(m_tile, k_tile, group, ct_k, s, t)
    m_tile, k_tile, group, ct_k, s, t = geometry.values()
    if _detect_arch() != "aie2p":
        raise NotImplementedError("q4nx_dequant() is only available on aie2p.")
    input_bytes = m_tile * k_tile // 2 + 4 * m_tile * k_tile // group
    output_bytes = m_tile * k_tile * 9 // 8
    return _make_extern(
        "q4nx_dequant_bfp",
        _default_source_path("q4nx_dequant.cc", subdir="generic"),
        [
            np.ndarray[(input_bytes,), np.dtype[np.uint8]],
            np.ndarray[(output_bytes,), np.dtype[np.uint8]],
        ],
        compile_flags=[
            f"-DQ4NX_{name.upper()}={value}" for name, value in geometry.items()
        ],
        contract=KernelContract(
            # aiecc measured_stack_size (Peano 22). A group size that is not a
            # power of two makes the `/ GROUP` in the inner loop call __muldi3,
            # which needs 64 bytes more than the power-of-two geometries.
            stack_bytes=1280,
            roles=(In, Out),
            reference=partial(q4nx_dequant_ref, **geometry),
            sample=partial(_q4nx_sample, **geometry),
            tolerance=Tolerance.exact(
                note="floor bf16 narrowing and bfp16ebs8 conversion, byte for byte"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=1,
        ),
    )
