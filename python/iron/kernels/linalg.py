# kernels/linalg.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Linear algebra kernel factories: mm, mv, cascade_mm."""

import numpy as np
from ml_dtypes import bfloat16

from aie.iron.kernel import ExternalFunction, Kernel

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

# Suffix for the zero_* symbol per output_dtype (used by kernels.mm to bind
# the .zero sibling Kernel).
_ZERO_SUFFIX = {
    np.int8: "i8",
    np.int16: "i16",
    np.int32: "i32",
    np.float32: "f32",
    bfloat16: "bf16",
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

    The compiled ``.o`` exports both the ``matmul_*`` and ``zero_*`` symbols.
    Use ``kernels.mm(...).zero`` to get a sibling Kernel binding the zero
    symbol against the same .o, suitable for accumulator initialization.

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
    # mm.cc emits both matmul_* and zero_* symbols; expose the zero binding
    # as a sibling Kernel pointing at the same .o so the design does
    # `matmul = kernels.mm(...); zero = matmul.zero` instead of a separate
    # kernels.mm_zero call (which would compile mm.cc a second time).
    zero_prefix = "zero" if vectorized else "zero_scalar"
    extern.zero = Kernel(
        f"{zero_prefix}_{_ZERO_SUFFIX[output_dtype]}",
        extern.object_file_name,
        [c_ty],
    )
    return extern


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
    extern = _make_extern(
        f"{prefix}_i16_i32",
        _default_source_path("mv.cc"),
        [a_ty, b_ty, c_ty],
        compile_flags=[f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}"],
        use_chess=use_chess,
    )
    # mv.cc emits both matvec_* and zero_* symbols; expose the zero binding
    # as a sibling Kernel pointing at the same .o.
    zero_prefix = "zero_vectorized" if vectorized else "zero_scalar"
    extern.zero = Kernel(f"{zero_prefix}_i32", extern.object_file_name, [c_ty])
    return extern


def cascade_mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype=np.int16,
    output_dtype=np.int16,
    use_chess: bool = False,
) -> ExternalFunction:
    """Cascade matrix-multiply kernel for multi-core accumulation.

    cascade_mm.cc emits all three cascade variants (``get_only``,
    ``put_only``, ``put_get``) plus a ``zero`` companion in one .o.  The
    returned ExternalFunction binds the ``get_only`` symbol; the other
    three are sibling :class:`Kernel`\\s available as attributes:

    * ``.get_only`` — same as the returned EF (top of the cascade chain).
    * ``.put_only`` — bottom of the chain.
    * ``.put_get`` — middle of the chain.
    * ``.zero`` — accumulator initializer.

    Designs typically use all four together, one per row of compute cores.

    Args:
        dim_m: Number of rows of A / C.
        dim_k: Number of columns of A / rows of B.
        dim_n: Number of columns of B / C.
        input_dtype: Input element type.
        output_dtype: Output element type.
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Raises:
        ValueError: When the dtype combination is not supported.
    """
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
    extern = _make_extern(
        f"matmul_scalar_cascade_get_only_{suffix}",
        _default_source_path("cascade_mm.cc"),
        [a_ty, b_ty, c_ty],
        compile_flags=[
            f"-DDIM_M={dim_m}",
            f"-DDIM_K={dim_k}",
            f"-DDIM_N={dim_n}",
        ],
        use_chess=use_chess,
    )
    extern.get_only = extern
    extern.put_only = Kernel(
        f"matmul_scalar_cascade_put_only_{suffix}",
        extern.object_file_name,
        [a_ty, b_ty, c_ty],
    )
    extern.put_get = Kernel(
        f"matmul_scalar_cascade_put_get_{suffix}",
        extern.object_file_name,
        [a_ty, b_ty, c_ty],
    )
    extern.zero = Kernel(
        f"zero_scalar_{_ZERO_SUFFIX[output_dtype]}",
        extern.object_file_name,
        [c_ty],
    )
    return extern
