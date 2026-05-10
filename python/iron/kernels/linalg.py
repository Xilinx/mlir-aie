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

from ._common import _default_source_path, _detect_arch, _make_extern

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

# Per-arch MMUL micro-kernel dimensions (r, s, t) used by aie_kernels/<arch>/mm.cc
# for each (input_dtype, output_dtype) combo.  These mirror the
# `combos(X) X(..., r, s, t)` macros in those files; if the C++ side
# changes geometry or adds a dtype combo, both tables here AND those macros
# must move together.  Designs use `kernels.mm(...).mac_dims` to look up
# the layout the freshly-compiled kernel actually expects.
_MM_MAC_DIMS = {
    "aie2": {
        (np.int8, np.int8): (4, 8, 8),
        (np.int8, np.int16): (4, 8, 8),
        (np.int8, np.int32): (4, 8, 8),
        (np.int16, np.int16): (4, 4, 4),
        (np.int16, np.int32): (4, 4, 4),
        (bfloat16, bfloat16): (4, 8, 4),
        (bfloat16, np.float32): (4, 8, 4),
    },
    "aie2p": {
        (np.int8, np.int8): (8, 8, 8),
        (np.int8, np.int16): (8, 8, 8),
        (np.int8, np.int32): (8, 8, 8),
        (np.int16, np.int16): (4, 4, 8),
        (np.int16, np.int32): (4, 4, 8),
        (bfloat16, bfloat16): (4, 8, 8),
        (bfloat16, np.float32): (4, 8, 8),
    },
}

# AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16 swaps the bf16 MMUL implementation
# on AIE2P from native bf16 mul-acc to BFP16-based emulation, which uses an
# 8x8x8 micro-kernel instead of the default 4x8x8.  Other (arch, dtype) combos
# are unaffected by the toggle.
_MM_EMULATED_BF16_MAC_DIMS_AIE2P = {
    (bfloat16, bfloat16): (8, 8, 8),
    (bfloat16, np.float32): (8, 8, 8),
}

# (suffix, _MM_COMBOS-style only_flag) per supported mm_zero output dtype.
_ZERO_DTYPE_INFO = {
    np.int8: ("i8", "i8_i8_ONLY"),
    np.int16: ("i16", "i16_i16_ONLY"),
    np.int32: ("i32", "i16_i32_ONLY"),
    np.float32: ("f32", "bf16_f32_ONLY"),
    bfloat16: ("bf16", "bf16_bf16_ONLY"),
}


def mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    vectorized: bool = True,
    b_col_maj: bool = False,
    c_col_maj: bool = False,
    use_chess: bool = False,
    emulate_bf16_mmul_with_bfp16: bool = False,
) -> ExternalFunction:
    """Matrix-multiply kernel: C += A * B.

    The compiled ``.o`` exports only the ``matmul_*`` symbols (``zero_*``
    suppressed via ``-DMATMUL_ONLY``) so it links cleanly alongside a ``.o``
    produced by :func:`mm_zero` for the same shape/dtype.

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type (``np.int8``, ``np.int16``, or ``bfloat16``).
        output_dtype: Output element type.
        vectorized: If ``True`` use the vectorized variant.
        b_col_maj: If ``True`` compile with ``-DB_COL_MAJ`` so the kernel
            consumes B laid out column-major.  Must agree with the
            design's B ``dims_to_stream``.
        c_col_maj: If ``True`` compile with ``-DC_COL_MAJ`` so the kernel
            writes C laid out column-major.  Must agree with the design's
            C output ``dims_to_stream``.
        use_chess: If ``True`` build with ``xchesscc_wrapper`` instead of
            Peano's ``clang++``.  All ExternalFunctions in a single
            ``@iron.jit`` design must share the same toolchain.
        emulate_bf16_mmul_with_bfp16: AIE2P only, bf16 inputs only.  When
            ``True`` compile with ``-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16``
            so the kernel uses BFP16-based emulation of the bf16 MMUL.
            Changes the micro-kernel dims to (8, 8, 8); designs reading
            ``.mac_dims`` will see the new geometry automatically.  Ignored
            for non-bf16 inputs and on AIE2.

    Returns:
        ExternalFunction configured for the matmul kernel.

    Raises:
        ValueError: When ``(input_dtype, output_dtype)`` is not a supported combination.
    """
    key = (input_dtype, output_dtype)
    if key not in _MM_COMBOS:
        raise ValueError(
            f"mm(): unsupported (input_dtype, output_dtype) = {key}. "
            f"Supported: {list(_MM_COMBOS.keys())}"
        )

    suffix, only_flag = _MM_COMBOS[key]
    prefix = "matmul" if vectorized else "matmul_scalar"
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]
    compile_flags = [
        f"-DDIM_M={dim_m}",
        f"-DDIM_K={dim_k}",
        f"-DDIM_N={dim_n}",
        f"-D{only_flag}",
        "-DMATMUL_ONLY",
    ]
    if b_col_maj:
        compile_flags.append("-DB_COL_MAJ")
    if c_col_maj:
        compile_flags.append("-DC_COL_MAJ")
    arch = _detect_arch()
    bf16_emulated = (
        emulate_bf16_mmul_with_bfp16 and arch == "aie2p" and input_dtype is bfloat16
    )
    if bf16_emulated:
        compile_flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
    extern = _make_extern(
        f"{prefix}_{suffix}",
        _default_source_path("mm.cc"),
        [a_ty, b_ty, c_ty],
        compile_flags=compile_flags,
        use_chess=use_chess,
    )
    if bf16_emulated:
        extern.mac_dims = _MM_EMULATED_BF16_MAC_DIMS_AIE2P[key]
    else:
        extern.mac_dims = _MM_MAC_DIMS[arch][key]
    return extern


def mm_zero(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    output_dtype=np.int16,
    vectorized: bool = True,
    use_chess: bool = False,
) -> ExternalFunction:
    """Zero-fill kernel companion for :func:`mm`.

    The compiled ``.o`` exports only the ``zero_*`` symbols (the companion
    ``matmul_*`` variants are suppressed via ``-DZERO_ONLY``) so it can be
    safely linked alongside a ``.o`` produced by :func:`mm` for the same
    shape/dtype without duplicate-symbol errors.

    Args:
        dim_m: Number of rows.
        dim_k: Inner dimension (must match the paired :func:`mm` call).
        dim_n: Number of columns.
        output_dtype: Element type of the output matrix.
        vectorized: If ``True`` use the vectorized variant.
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano's ``clang++``.  Must agree with the paired
            :func:`mm` call's ``use_chess``; mixed peano/chess in one
            design is rejected at compile time.

    Returns:
        ExternalFunction configured for the zero kernel.

    Raises:
        ValueError: When ``output_dtype`` is not supported.
    """
    if output_dtype not in _ZERO_DTYPE_INFO:
        raise ValueError(
            f"mm_zero(): unsupported output_dtype {output_dtype}. "
            f"Supported: {list(_ZERO_DTYPE_INFO.keys())}"
        )

    suffix, only_flag = _ZERO_DTYPE_INFO[output_dtype]
    prefix = "zero" if vectorized else "zero_scalar"
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]
    return _make_extern(
        f"{prefix}_{suffix}",
        _default_source_path("mm.cc"),
        [c_ty],
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
            f"-D{only_flag}",
            # Suppress mm.cc's matmul_* symbols so this .o doesn't collide
            # at link time with one produced by a parallel kernels.mm(...)
            # call.  See aie_kernels/aie2{,p}/mm.cc for the gating macro.
            "-DZERO_ONLY",
        ],
        use_chess=use_chess,
    )


def mv(
    dim_m: int = 32,
    dim_k: int = 32,
    input_dtype=np.int16,
    output_dtype=np.int32,
    vectorized: bool = True,
    use_chess: bool = False,
) -> ExternalFunction:
    """Matrix-vector multiply kernel: c += A * b.

    Args:
        dim_m: Number of rows of A (output vector length).
        dim_k: Number of columns of A (input vector length).
        input_dtype: Input element type. Only ``np.int16`` is supported.
        output_dtype: Output element type. Only ``np.int32`` is supported.
        vectorized: If ``True`` use the vectorized variant.
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano.  See :func:`mm` for the design-level
            constraint (all EFs in one design must agree).

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
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[np.int16]]
    b_ty = np.ndarray[(dim_k,), np.dtype[np.int16]]
    c_ty = np.ndarray[(dim_m,), np.dtype[np.int32]]
    return _make_extern(
        f"{prefix}_i16_i32",
        _default_source_path("mv.cc"),
        [a_ty, b_ty, c_ty],
        compile_flags=[f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}"],
        use_chess=use_chess,
    )


def cascade_mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    cascade_mode: str = "get_only",
    use_chess: bool = False,
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
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano.  See :func:`mm` for the design-level
            constraint (all EFs in one design must agree).

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
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[input_dtype]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[input_dtype]]
    c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[output_dtype]]
    return _make_extern(
        f"matmul_scalar_cascade_{cascade_mode}_{suffix}",
        _default_source_path("cascade_mm.cc"),
        [a_ty, b_ty, c_ty],
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
        ],
        use_chess=use_chess,
    )
