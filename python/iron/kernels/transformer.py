# transformer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Transformer building blocks: rms_norm, layer_norm (bf16, f32, affine+cast), rope, mm_activation_epilogue.

The bf16 norms and RoPE are re-exported from ``norm`` and ``datamovement``;
they support aie2 and aie2p, with ``cols`` as an alias for ``tile_size``.
The activation epilogue and the f32/affine norms take their aie2p source on
both generations. Each processes one row
(``cols`` elements) per call; the row length is a scalar ``Param`` the
factory binds to ``cols``. These are the kernels
``programming_examples/ml/{norm,rope,mm_activation_epilogue}`` build.
"""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _arch_traits,
    _by_tuned_arch,
    _kernel_source,
    _make_extern,
    _runtime_lib_include,
    _tuned_arch,
)
from .activation import _bf16, tanh_lut_ref
from .core import conv_even
from .datamovement import rope as rope
from .datamovement import rope_ref as rope_ref
from .norm import _LAYER_NORM_BF16_AIE2, _NORM_BF16
from .norm import layer_norm as layer_norm
from .norm import layer_norm_ref as layer_norm_ref
from .norm import rms_norm as rms_norm
from .norm import rms_norm_ref as rms_norm_ref

_EPS = 1e-5

# programming_examples/ml/norm judges the bf16 norms with atol 0.05 under the
# canonical bf16 rtol; the f32 LayerNorm pins rtol = 0 so a 1e-3 atol governs.
_NORM_F32 = Tolerance.relative(
    0.0, 1e-3, note="programming_examples/ml/norm layer_f32: atol 1e-3, rtol 0"
)
# The aie2 kernel multiplies in three bf16 limbs. Its error measured on npu1
# grows with cols: 9.5e-7 at 1024 and 2048, 1.4e-6 at 3072 and 1.9e-6 at
# 4096. For |y| >= 1, |got - ref| / (|got| + |ref|) reaches 3.1e-7.
_NORM_F32_AIE2 = Tolerance.relative(
    2.0**-21, 2e-6, note="aie2, measured on npu1 for cols up to 4096"
)
# The aie2p kernel multiplies in the same limbs. Its error measured on npu2
# reaches 8.6e-7 at 4096 (the kernel it replaced: 6.1e-7), and its bf16
# output stays within one ulp of the reference.
_NORM_F32_AIE2P = Tolerance.relative(
    2.0**-21, 2e-6, note="aie2p, measured on npu2 for cols up to 4096"
)
_LAYER_NORM_BF16_AIE2P = Tolerance.bf16_ulps(
    1,
    atol=1e-5,
    note="aie2p, measured on npu2: one ulp; atol covers the cancellation near 0",
)
_NORM_F32_TOLERANCE = {"aie2": _NORM_F32_AIE2, "aie2p": _NORM_F32_AIE2P}
_AFFINE_TOLERANCE = {"aie2": _LAYER_NORM_BF16_AIE2, "aie2p": _LAYER_NORM_BF16_AIE2P}


_EPILOGUE_LUT_TOLERANCE = Tolerance.exact(
    note="model of the aie2 build; measured bit-exact on npu1 over every "
    "extensive data case, 3 seeds"
)


def _cols(name: str, cols: int) -> None:
    if cols <= 0 or cols % 16:
        raise ValueError(f"{name}: cols must be a positive multiple of 16, got {cols}")


def _row_kernel(
    name: str,
    symbol: str,
    source: str,
    cols: int,
    in_dt,
    out_dt,
    ref,
    tol,
    ops,
    *,
    setup=None,
    stack_bytes=None,
) -> ExternalFunction:
    _cols(name, cols)
    in_ty = np.ndarray[(cols,), np.dtype[in_dt]]
    out_ty = np.ndarray[(cols,), np.dtype[out_dt]]
    return _make_extern(
        symbol,
        _kernel_source(source),
        [in_ty, out_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, cols),),
            reference=ref,
            tolerance=tol,
            ops_per_call=ops,
            acc_dtype=np.float32,
            reduction=cols,
            setup=setup,
            stack_bytes=stack_bytes,
        ),
    )


def layer_norm_f32(cols: int = 4096) -> ExternalFunction:
    """Row-wise LayerNorm on float32 in and out (gamma = 1, beta = 0, eps 1e-5).

    A separate factory rather than a dtype of
    [`layer_norm`][iron.kernels.norm.layer_norm]: this one is held to atol 2e-6
    (1e-3 in the portable build) instead of the bf16 tolerance, which its
    reference meets only by computing the variance two-pass in float64.
    Merging them would put that numerical difference behind a dtype switch.

    Args:
        cols: Elements per row (multiple of 16).
    """
    return _row_kernel(
        "layer_norm_f32",
        "layer_norm_f32",
        "transformer/layer_norm_f32.cc",
        cols,
        np.float32,
        np.float32,
        layer_norm_f32_ref,
        _by_tuned_arch(_NORM_F32_TOLERANCE, _NORM_F32),
        6 * cols,
        # aiecc measured_stack_size: 896 B tuned for aie2p, 160 B tuned for
        # aie2, 832 B untuned on aie2p (672 B on aie2); the 1024 B default
        # covers every build.
    )


def layer_norm_affine_cast(cols: int = 4096) -> ExternalFunction:
    """Row-wise LayerNorm, f32 in, per-column gamma/beta, bf16 out.

    The second argument holds ``gamma`` (``cols`` values) followed by ``beta``
    (``cols`` values) as float32: a tensor ``Param``, which the generic
    builder bakes into a core buffer.

    Args:
        cols: Elements per row (multiple of 16).
    """
    _cols("layer_norm_affine_cast", cols)
    in_ty = np.ndarray[(cols,), np.dtype[np.float32]]
    gb_ty = np.ndarray[(2 * cols,), np.dtype[np.float32]]
    out_ty = np.ndarray[(cols,), np.dtype[bfloat16]]
    return _make_extern(
        "layer_norm_affine_cast",
        _kernel_source("transformer/layer_norm_f32.cc"),
        [in_ty, gb_ty, out_ty, np.int32],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, Out, Param),
            parameter_bindings=((3, cols),),
            reference=layer_norm_affine_cast_ref,
            acc_dtype=np.float32,
            reduction=cols,
            tolerance=_by_tuned_arch(_AFFINE_TOLERANCE, _NORM_BF16),
            ops_per_call=8 * cols,
        ),
    )


def mm_activation_epilogue(tile_size: int = 1024) -> ExternalFunction:
    """GEMM epilogue on float32 rows: identity (0), SiLU (1), tanh-GELU (2) or ReLU (3) by ``mode``.

    One resident kernel whose ``mode`` is a runtime argument, so a design can
    switch activations without recompiling
    (programming_examples/ml/mm_activation_epilogue).

    AIE2 has no tanh instruction, so there SiLU and GELU read getTanhBf16's
    table, and its tuned build is judged against a model of that arithmetic
    instead of the true functions.

    Args:
        tile_size: Elements per call (multiple of 16).
    """
    _cols("mm_activation_epilogue", tile_size)
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    source = _kernel_source("transformer/mm_activation_epilogue.cc")
    lut = not _arch_traits().native_tanh
    lut_model = _tuned_arch() == "aie2"
    flags = None
    if lut:
        # lut_kernel.cc compiles the source next to lut_based_ops.cpp, whose
        # tables getTanhBf16 reads.
        flags = [f'-DAIE_LUT_KERNEL_SOURCE="{source}"', f"-I{_runtime_lib_include()}"]
        source = _kernel_source("common/lut_kernel.cc")
    return _make_extern(
        "mm_activation_epilogue_row",
        source,
        [tile_ty, tile_ty, np.int32, np.int32],
        compile_flags=flags,
        contract=KernelContract(
            trace=Trace.whole_call(),
            setup=conv_even,
            roles=(In, Out, Param, Param),
            parameter_bindings=((2, tile_size),),
            reference=(
                mm_activation_epilogue_lut_ref
                if lut_model
                else mm_activation_epilogue_ref
            ),
            acc_dtype=np.float32,
            reduction=1,
            tolerance=(
                _EPILOGUE_LUT_TOLERANCE
                if lut_model
                else Tolerance.relative(
                    0.128,
                    0.05,
                    note="programming_examples/ml/mm_activation_epilogue: atol 0.05 "
                    "for the bf16-internal SiLU / GELU, identity and ReLU are exact",
                )
            ),
            ops_per_call=8 * tile_size,
            uses_lut=lut,
        ),
    )


# --------------------------------------------------------------------------
# Numpy references (row-wise over the last axis)
# --------------------------------------------------------------------------


def layer_norm_f32_ref(x):
    """Numpy reference for [`layer_norm_f32`][iron.kernels.transformer.layer_norm_f32].

    Centered two-pass variance in float64 so it stays exact on the
    non-zero-mean input the f32 kernel is exercised with.
    """
    x64 = x.astype(np.float64)
    mean = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x64 - mean) / np.sqrt(var + _EPS)).astype(np.float32)


def layer_norm_affine_cast_ref(x, gamma_beta):
    """Numpy reference for [`layer_norm_affine_cast`][iron.kernels.transformer.layer_norm_affine_cast].

    ``gamma_beta`` is ``gamma`` then ``beta``, each ``cols`` float32 values.
    """
    x32 = x.astype(np.float32)
    gb = np.asarray(gamma_beta, dtype=np.float32).reshape(-1)
    cols = x32.shape[-1]
    gamma, beta = gb[:cols], gb[cols : 2 * cols]
    mean = x32.mean(axis=-1, keepdims=True)
    var = ((x32 - mean) ** 2).mean(axis=-1, keepdims=True)
    return ((x32 - mean) / np.sqrt(var + _EPS) * gamma + beta).astype(bfloat16)


def mm_activation_epilogue_ref(x, mode):
    """Numpy reference for [`mm_activation_epilogue`][iron.kernels.transformer.mm_activation_epilogue].

    ``mode`` 0 identity, 1 ``x * sigmoid(x)``, 2 tanh-approximation GELU,
    3 ``max(x, 0)``.
    """
    x32 = x.astype(np.float32)
    mode = int(mode)
    if mode == 0:
        return x32.astype(x.dtype)
    if mode == 1:
        with np.errstate(over="ignore"):
            return (x32 / (1.0 + np.exp(-x32))).astype(x.dtype)
    if mode == 2:
        # gelu's limit at -inf is 0, which the most negative float gives
        # rather than -inf * 0.
        x32 = np.maximum(x32, -np.finfo(np.float32).max)
        inner = 0.7978845608 * (x32 + 0.044715 * x32**3)
        return (0.5 * x32 * (1.0 + np.tanh(inner))).astype(x.dtype)
    if mode == 3:
        return np.maximum(x32, 0.0).astype(x.dtype)
    raise ValueError(f"mm_activation_epilogue mode must be 0, 1, 2 or 3, got {mode}")


def mm_activation_epilogue_lut_ref(x, mode):
    """Model of [`mm_activation_epilogue`][iron.kernels.transformer.mm_activation_epilogue] on aie2.

    Follows mm_activation_epilogue.cc's roundings around getTanhBf16
    ([`tanh_lut_ref`][iron.kernels.activation.tanh_lut_ref]). SiLU splits
    ``x`` into ``hi``, its top 16 bits, and ``lo``, ``bf16(x - hi)``, and
    multiplies each by the bf16 sigmoid ``(bf16(t + 1)) / 2``, where ``t`` is
    the table's tanh of ``bf16(x) / 2`` narrowed to bf16. ``hi`` is finite for
    any finite ``x``, so huge inputs give about ``x`` or 0 rather than NaN;
    +-inf still gives NaN. GELU runs in bf16: ``x``, ``x * x`` and the inner
    polynomial are each rounded before the next step, and the output is
    ``bf16(x / 2) * bf16(t + 1)``, all reading ``x`` clamped at -8 so -inf
    gives 0. Both return +0 where IEEE arithmetic gives -0, as the accumulator
    does. Identity and ReLU are exact.
    """
    x32 = np.asarray(x, np.float32)
    mode = int(mode)
    if mode not in (1, 2):
        return mm_activation_epilogue_ref(x, mode)
    with np.errstate(over="ignore", invalid="ignore"):
        if mode == 1:
            hi = (x32.view(np.uint32) & np.uint32(0xFFFF0000)).view(np.float32)
            lo = _bf16(x32 - hi)
            t = tanh_lut_ref(_bf16(x32) * np.float32(0.5))
            sig = _bf16(_bf16(t + np.float32(1.0)) * np.float32(0.5))
            out = hi * sig + lo * sig
        else:
            c0 = _bf16(np.float32(0.7978845608))
            c0c1 = _bf16(np.float32(0.7978845608) * np.float32(0.044715))
            xl = np.maximum(_bf16(x32), np.float32(-8.0))
            poly = _bf16(c0 + c0c1 * _bf16(xl * xl))
            t = tanh_lut_ref(_bf16(xl * poly))
            half_x = _bf16(np.float32(0.5) * xl)
            out = half_x * _bf16(t + np.float32(1.0))
    return (out + np.float32(0.0)).astype(x.dtype)
