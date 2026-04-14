# kernels/linalg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Linear algebra kernel factories: mm, mm_zero, mv, cascade_mm."""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction

from ._common import _detect_arch, _include_dirs, _kernel_source

_CASCADE_COMBOS = {
    (np.int16, np.int16): "i16_i16",
    (np.int16, np.int32): "i16_i32",
    (bfloat16, bfloat16): "bf16_bf16",
    (bfloat16, np.float32): "bf16_f32",
}

_MM_COMBOS = {
    (np.int8, np.int8): ("i8_i8", "i8_i8_ONLY"),
    (np.int8, np.int16): ("i8_i16", "i8_i16_ONLY"),
    (np.int8, np.int32): ("i8_i32", "i8_i32_ONLY"),
    (np.int16, np.int16): ("i16_i16", "i16_i16_ONLY"),
    (np.int16, np.int32): ("i16_i32", "i16_i32_ONLY"),
    (bfloat16, bfloat16): ("bf16_bf16", "bf16_bf16_ONLY"),
    (bfloat16, np.float32): ("bf16_f32", "bf16_f32_ONLY"),
}


def mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    vectorized: bool = True,
) -> ExternalFunction:
    """Matrix-multiply kernel: C += A * B.

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type (``np.int8``, ``np.int16``, or ``bfloat16``).
        output_dtype: Output element type.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        ExternalFunction configured for the matmul kernel.

    Raises:
        ValueError: When ``(input_dtype, output_dtype)`` is not a supported combination.
    """
    key = (input_dtype, output_dtype)
    if key not in _MM_COMBOS:
        supported = ", ".join(
            f"({k[0].__name__ if hasattr(k[0], '__name__') else k[0]}, "
            f"{k[1].__name__ if hasattr(k[1], '__name__') else k[1]})"
            for k in _MM_COMBOS
        )
        raise ValueError(
            f"mm(): unsupported (input_dtype, output_dtype) = {key}. "
            f"Supported combos: {supported}"
        )

    suffix, only_flag = _MM_COMBOS[key]
    prefix = "matmul" if vectorized else "matmul_scalar"
    func_name = f"{prefix}_{suffix}"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    source = _kernel_source(arch, arch, "mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
            f"-D{only_flag}",
        ],
    )


def mm_zero(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    output_dtype=np.int16,
    vectorized: bool = True,
) -> ExternalFunction:
    """Zero-fill kernel companion for :func:`mm`.

    Args:
        dim_m: Number of rows.
        dim_k: Inner dimension (must match the paired :func:`mm` call).
        dim_n: Number of columns.
        output_dtype: Element type of the output matrix.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        ExternalFunction configured for the zero kernel.

    Raises:
        ValueError: When ``output_dtype`` is not supported.
    """
    _dtype_suffix_map = {
        np.int8: "i8",
        np.int16: "i16",
        np.int32: "i32",
        np.float32: "f32",
        bfloat16: "bf16",
    }
    if output_dtype not in _dtype_suffix_map:
        raise ValueError(
            f"mm_zero(): unsupported output_dtype {output_dtype}. "
            f"Supported: {list(_dtype_suffix_map.keys())}"
        )

    suffix = _dtype_suffix_map[output_dtype]
    prefix = "zero" if vectorized else "zero_scalar"
    func_name = f"{prefix}_{suffix}"

    arch = _detect_arch()
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    _combo_for_out = {
        np.int8: "i8_i8_ONLY",
        np.int16: "i16_i16_ONLY",
        np.int32: "i16_i32_ONLY",
        np.float32: "bf16_f32_ONLY",
        bfloat16: "bf16_bf16_ONLY",
    }
    only_flag = _combo_for_out[output_dtype]

    source = _kernel_source(arch, arch, "mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
            f"-D{only_flag}",
        ],
    )


def mv(
    dim_m: int = 32,
    dim_k: int = 32,
    input_dtype=np.int16,
    output_dtype=np.int32,
    vectorized: bool = True,
) -> ExternalFunction:
    """Matrix-vector multiply kernel: c += A * b.

    Args:
        dim_m: Number of rows of A (output vector length).
        dim_k: Number of columns of A (input vector length).
        input_dtype: Input element type. Only ``np.int16`` is supported.
        output_dtype: Output element type. Only ``np.int32`` is supported.
        vectorized: If ``True`` use the vectorized variant.

    Returns:
        ExternalFunction configured for the matvec kernel.

    Raises:
        ValueError: When the dtype combination is not supported.
    """
    if input_dtype != np.int16 or output_dtype != np.int32:
        raise ValueError(
            f"mv(): only (np.int16, np.int32) is supported, "
            f"got ({input_dtype}, {output_dtype})"
        )

    prefix = "matvec_vectorized" if vectorized else "matvec_scalar"
    func_name = f"{prefix}_i16_i32"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[np.int16]]
    b_ty = np.ndarray[(dim_k,), np.dtype[np.int16]]
    c_ty = np.ndarray[(dim_m,), np.dtype[np.int32]]

    source = _kernel_source(arch, arch, "mv.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}"],
    )


def cascade_mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    cascade_mode: str = "get_only",
) -> ExternalFunction:
    """Cascade matrix-multiply kernel for multi-core accumulation.

    Available cascade modes: ``"put_only"``, ``"get_only"``, ``"put_get"``.

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type.
        output_dtype: Output element type.
        cascade_mode: One of ``"put_only"``, ``"get_only"``, ``"put_get"``.

    Returns:
        ExternalFunction configured for the cascade matmul kernel.

    Raises:
        ValueError: When the cascade_mode or dtype combination is not supported.
    """
    valid_modes = ("put_only", "get_only", "put_get")
    if cascade_mode not in valid_modes:
        raise ValueError(
            f"cascade_mm(): cascade_mode must be one of {valid_modes}, "
            f"got '{cascade_mode}'"
        )
    key = (input_dtype, output_dtype)
    if key not in _CASCADE_COMBOS:
        raise ValueError(
            f"cascade_mm(): unsupported (input_dtype, output_dtype) = {key}. "
            f"Supported: {list(_CASCADE_COMBOS.keys())}"
        )

    suffix = _CASCADE_COMBOS[key]
    func_name = f"matmul_scalar_cascade_{cascade_mode}_{suffix}"

    arch = _detect_arch()
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]

    source = _kernel_source(arch, arch, "cascade_mm.cc")
    return ExternalFunction(
        func_name,
        source_file=str(source),
        arg_types=[a_ty, b_ty, c_ty],
        include_dirs=_include_dirs(),
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
        ],
    )
