# kernels/norm.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Normalization kernel factories + numpy references: rms_norm, layer_norm."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16

from ._common import _default_source_path, _make_extern


def rms_norm(tile_size: int = 1024) -> ExternalFunction:
    """RMS-norm a bf16 row (gamma=1); design passes ``(in, out, cols)``, eps=1e-5."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "rms_norm",
        _default_source_path("rms_norm.cc"),
        [tile_ty, tile_ty, np.int32],
    )


def rms_norm_eps(tile_size: int = 1024) -> ExternalFunction:
    """RMS-norm a bf16 row (gamma=1); design passes ``(in, out, cols, epsilon)``."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "rms_norm_eps",
        _default_source_path("rms_norm.cc"),
        [tile_ty, tile_ty, np.int32, np.float32],
    )


def layer_norm(tile_size: int = 1024) -> ExternalFunction:
    """Layer-norm a bf16 row (gamma=1, beta=0); design passes ``(in, out, cols)``, eps=1e-5."""
    tile_ty = np.ndarray[(tile_size,), np.dtype[bfloat16]]
    return _make_extern(
        "layer_norm",
        _default_source_path("layer_norm.cc"),
        [tile_ty, tile_ty, np.int32],
    )


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
