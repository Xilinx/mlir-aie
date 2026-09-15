# kernels/norm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Normalization kernel factories + numpy references: rms_norm, layer_norm."""

from pathlib import Path

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import _default_source_path, _detect_arch, _include_dirs


def _norm_extern(func_name: str, filename: str, arg_types: list) -> ExternalFunction:
    """Norm kernel with aie_runtime_lib (the arch's vec_math.h) on the include path."""
    from aie.utils import config

    include = _include_dirs()
    include.append(
        str(Path(config.root_path()) / "aie_runtime_lib" / _detect_arch().upper())
    )
    return ExternalFunction(
        func_name,
        source_file=str(_default_source_path(filename)),
        arg_types=arg_types,
        include_dirs=include,
    )


def rms_norm(tile_size: int = 1024) -> ExternalFunction:
    """RMS-norm a bf16 row (gamma=1); design passes ``(in, out, cols)``, eps=1e-5."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _norm_extern("rms_norm", "rms_norm.cc", [tile_ty, tile_ty, np.int32])


def rms_norm_eps(tile_size: int = 1024) -> ExternalFunction:
    """RMS-norm a bf16 row (gamma=1); design passes ``(in, out, cols, epsilon)``."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _norm_extern(
        "rms_norm_eps", "rms_norm.cc", [tile_ty, tile_ty, np.int32, np.float32]
    )


def layer_norm(tile_size: int = 1024) -> ExternalFunction:
    """Layer-norm a bf16 row (gamma=1, beta=0); design passes ``(in, out, cols)``, eps=1e-5."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _norm_extern("layer_norm", "layer_norm.cc", [tile_ty, tile_ty, np.int32])


def rms_norm_ref(x, *, eps: float = 1e-5):
    """Numpy reference for [`rms_norm`][iron.kernels.norm.rms_norm]: ``x / sqrt(mean(x**2) + eps)``."""
    xf = x.astype(np.float32)
    ms = np.mean(xf * xf, axis=-1, keepdims=True)
    return (xf / np.sqrt(ms + eps)).astype(x.dtype)


def layer_norm_ref(x, *, eps: float = 1e-5):
    """Numpy reference for [`layer_norm`][iron.kernels.norm.layer_norm]: ``(x - mean) / sqrt(var + eps)``."""
    xf = x.astype(np.float32)
    mean = np.mean(xf, axis=-1, keepdims=True)
    var = np.mean((xf - mean) ** 2, axis=-1, keepdims=True)
    return ((xf - mean) / np.sqrt(var + eps)).astype(x.dtype)
