# kernels/norm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Normalization kernel factories + numpy references: rms_norm, layer_norm.

Each norm has two C entries, one fixing eps at 1e-5 with ``(in, out, cols)``
and an ``_eps`` one taking ``(in, out, cols, epsilon)``; a factory names the
entry it builds, so the two keep their names rather than one ABI changing.
"""

from pathlib import Path

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    BUILDS,
    KernelContract,
    Param,
    Trace,
    _detect_arch,
    _kernel_source,
    _make_extern,
    _tuned_arch,
)
from .core import conv_even

_NORM_BF16 = Tolerance.relative(
    0.128, 0.05, note="programming_examples/ml/norm: atol 0.05 with the bf16 rtol"
)

# The aie2 kernels keep the statistics in f32, measured on npu1 within one ulp
# of the reference. layer_norm's x inv_std - mean inv_std cancels near zero,
# where f32 leaves an absolute error under 1e-6 on unit-variance rows.
_RMS_NORM_BF16_AIE2 = Tolerance.bf16_ulps(
    1,
    atol=2.0**-126,
    note="aie2, measured on npu1: one ulp; atol is the smallest normal bf16",
)
_RMS_NORM_BF16_AIE2P = Tolerance.bf16_ulps(
    1,
    atol=2.0**-126,
    note="aie2p, measured on npu2: one ulp; atol is the smallest normal bf16",
)
_LAYER_NORM_BF16_AIE2 = Tolerance.bf16_ulps(
    1,
    atol=1e-5,
    note="aie2, measured on npu1: one ulp; atol covers the cancellation near 0",
)
_LAYER_NORM_BF16_AIE2P = Tolerance.bf16_ulps(
    1,
    atol=1e-5,
    note="aie2p, measured on npu2: one ulp; atol covers the cancellation near 0",
)
_RMS_NORM_TOLERANCE = {"aie2": _RMS_NORM_BF16_AIE2, "aie2p": _RMS_NORM_BF16_AIE2P}
_LAYER_NORM_TOLERANCE = {"aie2": _LAYER_NORM_BF16_AIE2, "aie2p": _LAYER_NORM_BF16_AIE2P}


def _norm_extern(
    func_name: str,
    filename: str,
    arg_types: list,
    contract: KernelContract,
    row: int,
) -> ExternalFunction:
    """Norm kernel with aie_runtime_lib (the arch's vec_math.h) on the include path.

    ``row`` is the row length the contract binds ``cols`` to; as ``NORM_COLS``
    it folds the kernel's divide by ``cols``.
    """
    from aie.utils import config

    runtime_dir = Path(config.aie_runtime_lib_dir()) / _detect_arch().upper()
    return _make_extern(
        func_name,
        _kernel_source(f"norm/{filename}"),
        arg_types,
        compile_flags=[f"-I{runtime_dir}", f"-DNORM_COLS={row}"],
        contract=contract,
    )


def _row_size(name: str, tile_size: int, cols: int | None, multiple: int = 1) -> int:
    if cols is not None:
        if tile_size != 1024 and tile_size != cols:
            raise ValueError(f"{name}: tile_size and cols must agree")
        tile_size = cols
    if tile_size <= 0 or tile_size % multiple:
        raise ValueError(
            f"{name}: tile_size (cols) must be a positive multiple of {multiple}, "
            f"got {tile_size}"
        )
    return tile_size


def _norm(
    name: str,
    tile_size: int,
    cols: int | None,
    *,
    eps: bool,
    layer: bool,
    multiple: int = 1,
) -> ExternalFunction:
    """``name``, a layer or RMS norm row kernel; with ``eps`` the design passes epsilon last."""
    tile_size = _row_size(name, tile_size, cols, multiple)
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    base = layer_norm_ref if layer else rms_norm_ref

    def with_eps(x, epsilon):
        return base(x, eps=epsilon)

    reference = with_eps if eps else base
    tolerance = _LAYER_NORM_TOLERANCE if layer else _RMS_NORM_TOLERANCE
    return _norm_extern(
        name,
        "layer_norm.cc" if layer else "rms_norm.cc",
        [tile_ty, tile_ty, np.int32] + ([np.float32] if eps else []),
        KernelContract(
            trace=Trace.whole_call(),
            setup=None if layer or _tuned_arch() == "aie2" else conv_even,
            roles=(In, Out, Param) + ((Param,) if eps else ()),
            parameter_bindings=((2, tile_size),),
            per_build=tuple(
                (build, reference, tolerance.get(build.tuned, _NORM_BF16))
                for build in BUILDS
            ),
            acc_dtype=np.float32,
            reduction=tile_size,
            ops_per_call=(6 if layer else 4) * tile_size,
        ),
        tile_size,
    )


def rms_norm(tile_size: int = 1024, *, cols: int | None = None) -> ExternalFunction:
    """RMS-norm a bf16 row on aie2/aie2p; ``(in, out, cols)``, eps=1e-5.

    ``cols`` is a compatibility alias for ``tile_size``. Positive row lengths
    need not be vector-aligned: the kernel handles scalar tails.
    """
    return _norm("rms_norm", tile_size, cols, eps=False, layer=False)


def rms_norm_eps(tile_size: int = 1024, *, cols: int | None = None) -> ExternalFunction:
    """RMS-norm a bf16 row (gamma=1); design passes ``(in, out, cols, epsilon)``."""
    return _norm("rms_norm_eps", tile_size, cols, eps=True, layer=False)


def layer_norm(tile_size: int = 1024, *, cols: int | None = None) -> ExternalFunction:
    """Layer-norm a bf16 row; ``(in, out, cols)``, gamma=1, beta=0, eps=1e-5.

    ``cols`` aliases ``tile_size``, a positive multiple of 16 (the source
    processes whole 16-lane halves, without a scalar tail).
    """
    return _norm("layer_norm", tile_size, cols, eps=False, layer=True, multiple=16)


def layer_norm_eps(
    tile_size: int = 1024, *, cols: int | None = None
) -> ExternalFunction:
    """Layer-norm a bf16 row (gamma=1, beta=0); design passes ``(in, out, cols, epsilon)``.

    Args:
        tile_size: The row length, a positive multiple of 16.
        cols: An alias for ``tile_size``.

    Returns:
        The kernel.
    """
    return _norm("layer_norm_eps", tile_size, cols, eps=True, layer=True, multiple=16)


def rms_norm_ref(x, *, eps: float = 1e-5):
    """Numpy reference for [`rms_norm`][aie.iron.kernels.norm.rms_norm]: ``x / sqrt(mean(x**2) + eps)``."""
    xf = x.astype(np.float32)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf / np.sqrt(ms + eps)).astype(x.dtype)


def layer_norm_ref(x, *, eps: float = 1e-5):
    """Numpy reference for [`layer_norm`][aie.iron.kernels.norm.layer_norm]: ``(x - mean) / sqrt(var + eps)``."""
    xf = x.astype(np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True)
    var = np.mean((xf - mean) ** 2, axis=-1, keepdims=True)
    return ((xf - mean) / np.sqrt(var + eps)).astype(x.dtype)
