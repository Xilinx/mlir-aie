# transformer.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Transformer building blocks: rms_norm, layer_norm (bf16, f32, affine+cast), rope, mm_activation_epilogue.

The bf16 norms and RoPE are re-exported from ``norm`` and ``datamovement``;
they support aie2 and aie2p, with ``cols`` as an alias for ``tile_size``.
The f32/affine norms and activation epilogue are aie2p-only. Each processes one row
(``cols`` elements) per call; the row length is a runtime argument the
design passes as the ``count`` role. These are the kernels
``programming_examples/ml/{norm,rope,mm_activation_epilogue}`` build.
"""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out, Param, Scalar
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import KernelContract, _default_source_path, _detect_arch, _make_extern
from .core import conv_even
from .datamovement import rope as rope
from .datamovement import rope_ref as rope_ref
from .norm import _NORM_BF16
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


def _aie2p_only(name: str, source: str) -> None:
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            f"{name}: aie_kernels/aie2p/{source} has no aie2 port; select an NPU2 device"
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
    _aie2p_only(name, source)
    _cols(name, cols)
    in_ty = np.ndarray[(cols,), np.dtype[in_dt]]
    out_ty = np.ndarray[(cols,), np.dtype[out_dt]]
    return _make_extern(
        symbol,
        _default_source_path(source, subdir="aie2p"),
        [in_ty, out_ty, np.int32],
        contract=KernelContract(
            roles=(In, Out, Scalar),
            scalar_bindings=((2, cols),),
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
    [`layer_norm`][iron.kernels.norm.layer_norm], though both come from
    one templated core in ``layer_norm.cc``: this one is held to atol 1e-3
    instead of the bf16 tolerance, which its reference meets only by computing
    the variance two-pass in float64. Merging them would put that numerical
    difference behind a dtype switch.

    Args:
        cols: Elements per row (multiple of 16).
    """
    return _row_kernel(
        "layer_norm_f32",
        "layer_norm_f32",
        "layer_norm.cc",
        cols,
        np.float32,
        np.float32,
        layer_norm_f32_ref,
        _NORM_F32,
        6 * cols,
        # Allow headroom above Peano 22's 1152-byte measured lower bound:
        # __divsf3 adds 64 bytes without .stack_sizes metadata.
        stack_bytes=2048,
    )


def layer_norm_affine_cast(cols: int = 4096) -> ExternalFunction:
    """Row-wise LayerNorm, f32 in, per-column gamma/beta, bf16 out.

    The second argument holds ``gamma`` (``cols`` values) followed by ``beta``
    (``cols`` values) as float32; it is a ``param`` the design holds for the
    whole run.

    Args:
        cols: Elements per row (multiple of 16).
    """
    _aie2p_only("layer_norm_affine_cast", "layer_norm.cc")
    _cols("layer_norm_affine_cast", cols)
    in_ty = np.ndarray[(cols,), np.dtype[np.float32]]
    gb_ty = np.ndarray[(2 * cols,), np.dtype[np.float32]]
    out_ty = np.ndarray[(cols,), np.dtype[bfloat16]]
    return _make_extern(
        "layer_norm_affine_cast",
        _default_source_path("layer_norm.cc", subdir="aie2p"),
        [in_ty, gb_ty, out_ty, np.int32],
        contract=KernelContract(
            roles=(In, Param, Out, Scalar),
            scalar_bindings=((3, cols),),
            reference=layer_norm_affine_cast_ref,
            acc_dtype=np.float32,
            reduction=cols,
            tolerance=_NORM_BF16,
            ops_per_call=8 * cols,
        ),
    )


def mm_activation_epilogue(tile_size: int = 1024) -> ExternalFunction:
    """GEMM epilogue on float32 rows: identity (0), SiLU (1), tanh-GELU (2) or ReLU (3) by ``mode``.

    One resident kernel whose ``mode`` is a runtime argument, so a design can
    switch activations without recompiling
    (programming_examples/ml/mm_activation_epilogue).

    Args:
        tile_size: Elements per call (multiple of 16).
    """
    _aie2p_only("mm_activation_epilogue", "mm_activation_epilogue.cc")
    _cols("mm_activation_epilogue", tile_size)
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.float32]]
    return _make_extern(
        "mm_activation_epilogue_row",
        _default_source_path("mm_activation_epilogue.cc", subdir="aie2p"),
        [tile_ty, tile_ty, np.int32, np.int32],
        contract=KernelContract(
            setup=conv_even,
            roles=(In, Out, Scalar, Scalar),
            scalar_bindings=((2, tile_size),),
            reference=mm_activation_epilogue_ref,
            acc_dtype=np.float32,
            reduction=1,
            tolerance=Tolerance.relative(
                0.128,
                0.05,
                note="programming_examples/ml/mm_activation_epilogue: atol 0.05 "
                "for the bf16-internal SiLU / GELU, identity and ReLU are exact",
            ),
            ops_per_call=8 * tile_size,
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
        inner = 0.7978845608 * (x32 + 0.044715 * x32**3)
        return (0.5 * x32 * (1.0 + np.tanh(inner))).astype(x.dtype)
    if mode == 3:
        return np.maximum(x32, 0.0).astype(x.dtype)
    raise ValueError(f"mm_activation_epilogue mode must be 0, 1, 2 or 3, got {mode}")
