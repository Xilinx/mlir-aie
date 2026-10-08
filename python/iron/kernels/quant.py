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
from aie.utils.aie2p_emulation import bf16_floor
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Trace,
    _arch_traits,
    _kernel_source,
    _make_extern,
)

# FastFlowLM's q4nx block: Q4NX_M_TILE out-features by Q4NX_K_TILE
# in-features, with a scale and a minimum per Q4NX_GROUP in-features.
Q4NX_M_TILE = 32
Q4NX_K_TILE = 256
Q4NX_GROUP = 32
Q4NX_BLOCK_BYTES = (
    Q4NX_M_TILE * Q4NX_K_TILE // 2 + 4 * Q4NX_M_TILE * Q4NX_K_TILE // Q4NX_GROUP
)


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


def q4nx_unpack(blocks, *, m_tile=Q4NX_M_TILE, k_tile=Q4NX_K_TILE, group=Q4NX_GROUP):
    """Split q4nx blocks into their codes, scales and minima.

    ``blocks`` is uint8 of shape ``(..., block_bytes)``, one block per row.
    A block holds little-endian bf16 scales, then bf16 minima, both stored
    ``[k // group, m]``. 4-bit codes follow, stored ``[m // 16, k, m % 16]``,
    each byte low nibble first. A weight is ``min + scale * code``.

    Returns ``(codes, scales, mins)``: ``codes`` is uint8 ``[..., m_tile,
    k_tile]``, and ``scales`` and ``mins`` are float32 ``[..., m_tile, k_tile
    // group]``.
    """
    groups = k_tile // group
    param_bytes = 4 * m_tile * groups
    block_bytes = m_tile * k_tile // 2 + param_bytes
    blocks = np.ascontiguousarray(blocks, np.uint8)
    if blocks.ndim == 0 or blocks.shape[-1] != block_bytes:
        raise ValueError(f"q4nx_unpack: expected {block_bytes} bytes per block")
    lead = blocks.shape[:-1]
    params = np.ascontiguousarray(blocks[..., :param_bytes]).view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    params = np.swapaxes(params.reshape(*lead, 2, groups, m_tile), -1, -2)
    scales, mins = np.moveaxis(params, -3, 0)
    packed = blocks[..., param_bytes:]
    codes = np.empty((*lead, m_tile * k_tile), np.uint8)
    codes[..., 0::2], codes[..., 1::2] = packed & 15, packed >> 4
    codes = codes.reshape(*lead, m_tile // 16, k_tile, 16)
    codes = np.swapaxes(codes, -1, -2).reshape(*lead, m_tile, k_tile)
    return codes, scales, mins


def q4nx_dequant_ref(
    payload,
    *,
    m_tile=Q4NX_M_TILE,
    k_tile=Q4NX_K_TILE,
    group=Q4NX_GROUP,
    ct_k=128,
    s=8,
    t=8,
):
    """Dequantize packed q4nx to the kernel's bfp16ebs8 output bytes.

    Input has shape ``(..., m_tile*k_tile//2 + 4*m_tile*k_tile//group)``,
    one q4nx block per row in the layout
    [`q4nx_unpack`][iron.kernels.quant.q4nx_unpack] reads. ``n`` below is
    the out-feature, ``m`` there.

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
    q, scales, mins = q4nx_unpack(data, m_tile=m_tile, k_tile=k_tile, group=group)
    if not (np.isfinite(scales).all() and np.isfinite(mins).all()):
        raise ValueError("q4nx_dequant_ref: scales and minima must be finite")
    # A bf16 scale times a four-bit integer is exact in float32. Float64
    # models the fused addition before its single float32 accumulator rounding.
    with np.errstate(over="ignore", invalid="ignore"):
        values = (
            np.repeat(scales, group, axis=-1).astype(np.float64) * q
            + np.repeat(mins, group, axis=-1).astype(np.float64)
        ).astype(np.float32)
        values = bf16_floor(np.swapaxes(values, -1, -2))
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
    *,
    m_tile=Q4NX_M_TILE,
    k_tile=Q4NX_K_TILE,
    group=Q4NX_GROUP,
    ct_k=128,
    s=8,
    t=8,
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
    if not _arch_traits().bfp16:
        raise NotImplementedError("q4nx_dequant() is only available on aie2p.")
    input_bytes = m_tile * k_tile // 2 + 4 * m_tile * k_tile // group
    output_bytes = m_tile * k_tile * 9 // 8
    return _make_extern(
        "q4nx_dequant_bfp",
        _kernel_source("quant/q4nx_dequant.cc"),
        [
            np.ndarray[(input_bytes,), np.dtype[np.uint8]],
            np.ndarray[(output_bytes,), np.dtype[np.uint8]],
        ],
        # The inner loop is one long latency chain, so it is allowed five
        # pipeline stages instead of the default three.
        compile_flags=[
            f"-DQ4NX_{name.upper()}={value}" for name, value in geometry.items()
        ]
        + ["-mllvm", "--aie-pipeliner-max-stagecount=5"],
        contract=KernelContract(
            trace=Trace.whole_call(),
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
