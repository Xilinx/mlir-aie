# kernels/linalg.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Linear algebra kernel factories: mm, mv, cascade_mm."""

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron.kernel import ExternalFunction
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    _declare_dtypes,
    _default_source_path,
    _detect_arch,
    _make_extern,
)

_CASCADE_COMBOS = {
    (np.int16, np.int16): "i16_i16",
    (np.int16, np.int32): "i16_i32",
    (bfloat16, bfloat16): "bf16_bf16",
    (bfloat16, np.float32): "bf16_f32",
}

# Mirror of the ``combos(X)`` macro in aie_kernels/aie2/cascade_mm.cc.
# Designs use ``kernels.cascade_mm(...).mac_dims`` to look up the
# scalar-block geometry the compiled cascade kernel expects.  cascade_mm
# only ships an aie2 .cc today; if an aie2p variant lands the table
# needs the new arch added.
#
# The cascade_mm.cc kernel is fully scalar — `a[row * colA + i]` walks A
# element-by-element with no SIMD tiling — so the L2->L1 buffer must be
# plain row-major.  mac_dims (1, 1, 1) yields the identity dim_to_stream
# pattern when designs build it as [(m//r, r*k), (k//s, s), (r, k), (s, 1)].
# Larger values would shuffle A/B into a tiled layout the scalar kernel
# does not understand, producing garbage outputs (262144-element mismatch
# observed in CI with the previous (4, 4, 4) entries).
_CASCADE_MM_SCALAR_DIMS = {
    (np.int16, np.int16): (1, 1, 1),
    (np.int16, np.int32): (1, 1, 1),
    (bfloat16, bfloat16): (1, 1, 1),
    (bfloat16, np.float32): (1, 1, 1),
}

_CASCADE_MM_MAC_DIMS = {
    # cascade_mm.cc is currently shared by AIE2 and AIE2P through
    # _kernel_source's aie2 fallback.  It is scalar on both targets, so the
    # required stream layout remains plain row-major.
    "aie2": _CASCADE_MM_SCALAR_DIMS,
    "aie2p": _CASCADE_MM_SCALAR_DIMS,
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


def mm_acc_dtype(input_dtype):
    """Return what ``mm.cc`` accumulates in for ``input_dtype``.

    ``accauto`` is acc32 for 8-bit and acc64 for 16-bit integer inputs, and
    float32 for bf16.
    """
    dt = np.dtype(input_dtype)
    if dt == np.dtype(np.int8):
        return np.int32
    if dt == np.dtype(np.int16):
        return np.int64
    return np.float32


def mm_ref(a, b):
    """Numpy reference for [`mm`][iron.kernels.linalg.mm]: ``a @ b`` accumulated in int64 or float64."""
    acc = np.int64 if np.issubdtype(a.dtype, np.integer) else np.float64
    return a.astype(acc) @ b.astype(acc)


def mv_ref(a, b):
    """Numpy reference for [`mv`][iron.kernels.linalg.mv]: ``a @ b`` accumulated in int64."""
    return a.astype(np.int64) @ b.astype(np.int64)


def mm_bfp_ref(a, b):
    """Numpy reference for [`mm_bfp`][iron.kernels.linalg.mm_bfp]: ``a @ b`` on bfp16ebs8-quantised operands.

    ``a`` is ``(M, K)`` and ``b`` ``(K, N)`` float; each is quantised the way
    the host encodes it for the kernel (blocks of 8 along ``K``, see
    :mod:`aie.utils.bfp`) and the product is accumulated in float64. The
    kernel's own output is bfp16ebs8 too, which the tolerance covers.
    """
    from aie.utils import bfp

    aq = bfp.quantize(np.asarray(a, dtype=np.float32)).astype(np.float64)
    bq = bfp.quantize(np.ascontiguousarray(np.asarray(b, dtype=np.float32).T)).T
    return aq @ bq.astype(np.float64)


def mm_bfp_mixed_ref(a, b):
    """Numpy reference for [`mm_bfp`][iron.kernels.linalg.mm_bfp] with ``mixed=True``.

    ``a`` (bf16) is used as is, as ``mixed_test.cpp`` does -- the core
    converts it to bfp16 itself, with a rounding the reference does not
    model, which is why the mixed tolerance is twice the plain one; ``b``
    is quantised as in [`mm_bfp_ref`][iron.kernels.linalg.mm_bfp_ref].
    """
    from aie.utils import bfp

    bq = bfp.quantize(np.ascontiguousarray(np.asarray(b, dtype=np.float32).T)).T
    return np.asarray(a).astype(np.float64) @ bq.astype(np.float64)


# programming_examples/ml/block_datatypes/matrix_multiplication/{bfp,mixed}_test.cpp:
# the bf16 matmul tolerances, with 3x the absolute term for a bfp16 C (its
# 8-bit mantissas share one exponent per 8 values) and 2x the relative term
# when the core converts a bf16 A itself.
_BFP_TOLERANCE = Tolerance.relative(
    0.05, 1.5, note="bfp_test.cpp: bf16 rel_tol, 3x abs_tol for the bfp16 C"
)
_BFP_MIXED_TOLERANCE = Tolerance.relative(
    0.1, 0.5, note="mixed_test.cpp: 2x bf16 rel_tol for the in-core bf16->bfp16 A"
)


def _linalg_tolerance(input_dtype) -> Tolerance:
    if np.issubdtype(np.dtype(input_dtype), np.integer):
        return Tolerance.exact(note="integer matmul")
    # bf16 inputs accumulate in fp32 in an order that differs from numpy's.
    return Tolerance.relative(
        0.05,
        0.5,
        note="C++ matmul harness get_*_tol, via verify.assert_close_with_benchmark defaults",
    )


def mm_stream_dims(
    dim_m: int,
    dim_k: int,
    dim_n: int,
    mac_dims,
    *,
    b_col_maj: bool = False,
    c_col_maj: bool = False,
) -> dict:
    """DMA ``dims_to_stream`` that feed ``mm.cc`` its (r, s, t) micro-tiles.

    ``mm.cc`` consumes A, B and produces C in the micro-tile blocking given by
    ``mac_dims``; a plain row-major stream yields wrong numbers, not an error.
    Every matmul design (single_core, whole_array, cascade, ...) derives these
    same three transforms from ``mac_dims``; ``kernels.mm(...).stream_dims``
    carries them so designs do not re-derive them. Keys ``"A"``, ``"B"``, ``"C"``.

    ``b_col_maj`` describes a B tile stored as ``(n, k)`` (the transpose) and
    ``c_col_maj`` a C tile emitted as ``(n, m)``, matching the kernel's
    ``-DB_COL_MAJ`` / ``-DC_COL_MAJ`` builds.
    """
    r, s, t = mac_dims
    m, k, n = dim_m, dim_k, dim_n
    a = [(m // r, r * k), (k // s, s), (r, k), (s, 1)]
    if b_col_maj:
        b = [(n // t, t * k), (k // s, s), (t, k), (s, 1)]
    else:
        b = [(k // s, s * n), (n // t, t), (s, n), (t, 1)]
    if c_col_maj:
        c = [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
    else:
        c = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    return {"A": a, "B": b, "C": c}


def mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype: type = np.int16,
    output_dtype: type = np.int16,
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
        contract=KernelContract(
            rounding_mode=(
                # aie2p/mm.cc sets conv_even itself and restores it; aie2/mm.cc
                # stores bf16 in whatever mode the core is in.
                "sets_own"
                if arch == "aie2p"
                else "conv_even" if output_dtype is bfloat16 else "unspecified"
            ),
            roles=("in", "in", "inout"),  # C += A * B; see the .zero sibling
            reference=mm_ref,
            acc_dtype=mm_acc_dtype(input_dtype),
            reduction=dim_k,
            overflow="undefined",  # to_vector<T_out> without set_sat
            tolerance=_linalg_tolerance(input_dtype),
            ops_per_call=2 * dim_m * dim_k * dim_n,
        ),
    )
    if bf16_emulated:
        extern.mac_dims = _MM_EMULATED_BF16_MAC_DIMS_AIE2P[key]
    else:
        extern.mac_dims = _MM_MAC_DIMS[arch][key]
    extern.dims = (dim_m, dim_k, dim_n)
    extern.stream_dims = mm_stream_dims(
        dim_m,
        dim_k,
        dim_n,
        extern.mac_dims,
        b_col_maj=b_col_maj,
        c_col_maj=c_col_maj,
    )
    # Host-side layout the streams above assume: B given as (n, k) tiles of
    # B^T, C emitted as (n, m) tiles of C^T. aie.utils.kernel_harness reads
    # these to transpose the host operands.
    extern.b_col_maj = bool(b_col_maj)
    extern.c_col_maj = bool(c_col_maj)
    # mm.cc emits both matmul_* and zero_* symbols; expose the zero binding
    # as a sibling Kernel pointing at the same .o so the design does
    # `matmul = kernels.mm(...); zero = matmul.zero` instead of a separate
    # kernels.mm_zero call (which would compile mm.cc a second time).
    zero_prefix = "zero" if vectorized else "zero_scalar"
    extern.zero = extern.sibling(f"{zero_prefix}_{_ZERO_SUFFIX[output_dtype]}", [c_ty])
    return extern


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(
    mm, tuple({"input_dtype": i, "output_dtype": o} for (i, o) in _MM_MAC_DIMS["aie2p"])
)


def mv(
    dim_m: int = 32,
    dim_k: int = 32,
    input_dtype: type = np.int16,
    output_dtype: type = np.int32,
    vectorized: bool = True,
    use_chess: bool = False,
    vec_size: int = 64,
) -> ExternalFunction:
    """Matrix-vector multiply kernel: c += A * b.

    Two kernels live behind this factory, selected by dtype:

    * ``(np.int16, np.int32)`` builds ``aie_kernels/<arch>/mv.cc``, whose
      vectorized path wants A in the word-transposed layout
      ``a_dims_from_stream`` publishes, and which also exports the
      ``zero_*`` symbol exposed as ``.zero``.
    * ``(bfloat16, bfloat16)`` builds the shared
      ``aie_kernels/generic/mv.cc``, the kernel behind IRON's ``GEMV``
      operator. Its signature leads with two runtime scalars,
      ``(m, row_offset, A, b, c)`` -- ``row_offset`` shifts the write into
      ``c`` so one core can fill several output blocks -- it takes
      ``VEC_SIZE`` as well as ``DIM_K``, reads A row-major, and exports no
      zero symbol.

    Args:
        dim_m: Number of rows of A (output vector length).
        dim_k: Number of columns of A (input vector length).
        input_dtype: Input element type: ``np.int16`` or ``bfloat16``.
        output_dtype: Output element type: ``np.int32`` for ``np.int16``
            inputs, ``bfloat16`` for ``bfloat16`` inputs.
        vectorized: If ``True`` use the vectorized variant.
        use_chess: If ``True`` build the .o with ``xchesscc_wrapper``
            instead of Peano.  See [`mm`][iron.kernels.linalg.mm] for the design-level
            constraint (all EFs in one design must agree).
        vec_size: bf16 only: the kernel's ``VEC_SIZE`` accumulation width.

    Returns:
        ExternalFunction configured for the matvec kernel.

    Raises:
        ValueError: When the dtype combination is not supported.
    """
    if (input_dtype, output_dtype) == (bfloat16, bfloat16):
        return _mv_bf16(dim_m, dim_k, vectorized, use_chess, vec_size)
    if input_dtype != np.int16 or output_dtype != np.int32:
        raise ValueError(
            f"mv(): only (np.int16, np.int32) and (bfloat16, bfloat16) are "
            f"supported, got ({input_dtype}, {output_dtype})"
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
        contract=KernelContract(
            roles=("in", "in", "inout"),  # C += A * B; see the .zero sibling
            reference=mv_ref,
            acc_dtype=np.int32,  # acc32
            reduction=dim_k,
            overflow="undefined",
            tolerance=Tolerance.exact(note="int16 x int16 accumulated in int32"),
            ops_per_call=2 * dim_m * dim_k,
        ),
    )
    # The vectorized kernel reads A in a "32-bit-word transposed" layout (see
    # aie_kernels/aie2/mv.cc): 2-byte elements are packed two per word, rows
    # of each 2-column word slowly, m rows then the next 2-col word. A design
    # applies this as dims_from_stream on the hop into the core.
    extern.a_dims_from_stream = (
        [(dim_m, 2), (dim_k // 2, 2 * dim_m), (2, 1)] if vectorized else None
    )
    # mv.cc emits both matvec_* and zero_* symbols; expose the zero binding
    # as a sibling Kernel pointing at the same .o.
    zero_prefix = "zero_vectorized" if vectorized else "zero_scalar"
    extern.zero = extern.sibling(f"{zero_prefix}_i32", [c_ty])
    return extern


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(
    mv,
    (
        {"input_dtype": np.int16, "output_dtype": np.int32},
        # dim_k: the bf16 kernel accumulates VEC_SIZE (64) elements at a
        # time, so the default dim_k of 32 is too short for it.
        {"input_dtype": bfloat16, "output_dtype": bfloat16, "dim_k": 256},
    ),
)


def _mv_bf16(dim_m, dim_k, vectorized, use_chess, vec_size) -> ExternalFunction:
    """bf16 matvec from ``aie_kernels/generic/mv.cc`` (see [`mv`][iron.kernels.linalg.mv])."""
    if vec_size <= 0 or dim_k % vec_size:
        raise ValueError(
            f"mv(): dim_k ({dim_k}) must be a positive multiple of vec_size "
            f"({vec_size})"
        )
    prefix = "matvec_vectorized" if vectorized else "matvec_scalar"
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(dim_k,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(dim_m,), np.dtype[bfloat16]]
    return _make_extern(
        f"{prefix}_bf16_bf16",
        _default_source_path("mv.cc", subdir="generic"),
        [np.int32, np.int32, a_ty, b_ty, c_ty],
        compile_flags=[f"-DDIM_K={dim_k}", f"-DVEC_SIZE={vec_size}"],
        use_chess=use_chess,
        contract=KernelContract(
            rounding_mode="sets_own",
            roles=("scalar", "scalar", "in", "in", "out"),
            reference=mv_bf16_ref,
            acc_dtype=np.float32,  # accfloat, reduced to bf16 on store
            reduction=dim_k,
            tolerance=_linalg_tolerance(bfloat16),
            ops_per_call=2 * dim_m * dim_k,
            unsupported=(
                "the harness's matvec design drives the int16 kernel's "
                "(A, b, c) signature; this one leads with the runtime m and "
                "row_offset scalars"
            ),
        ),
    )


def mv_bf16_ref(m, row_offset, a, b):
    """Numpy reference for the bf16 [`mv`][iron.kernels.linalg.mv]: ``a @ b`` over ``m`` rows, accumulated in float32.

    ``a`` is row-major ``(rows, K)``; only the first ``m`` rows are computed,
    and the kernel writes them at ``row_offset`` into ``c``.
    """
    a = np.asarray(a).reshape(-1, np.asarray(b).shape[-1])
    out = a[: int(m)].astype(np.float32) @ np.asarray(b).astype(np.float32)
    return np.concatenate([np.zeros(int(row_offset), np.float32), out])


_BFP_MAC_DIMS = (8, 8, 8)  # the bfp16ebs8 mmul is 8x8x8


def mm_bfp(
    dim_m: int = 64, dim_k: int = 64, dim_n: int = 64, mixed: bool = False
) -> ExternalFunction:
    """Block-floating-point matmul ``C += A @ B`` on bfp16ebs8 blocks (aie2p only).

    ``mixed=False`` (``aie_kernels/aie2p/mm_bfp.cc``): A, B and C are
    ``v8bfp16ebs8`` blocks, all pre-shuffled into the mmul layout, so no
    DMA transform applies (``stream_dims`` is ``None`` for every operand).
    ``mixed=True`` (``mm_bfp_mixed.cc``): A is bf16 in the (r, s, t)
    micro-tile layout, B is bfp16ebs8, C is bf16; ``stream_dims["A"]`` and
    ``["C"]`` carry the transforms and ``["B"]`` is ``None``.

    ``.zero`` is the matching zeroing kernel, compiled from the same source
    with ``-DZERO_ONLY``. The host holds B transposed (``b_col_maj``), and
    every bfp16ebs8 operand is encoded and shuffled into the mmul tile
    layout on the host with :mod:`aie.utils.bfp`, which is what the generic
    harness does; the contract's reference multiplies the quantised
    operands. These are the kernels
    programming_examples/ml/block_datatypes/matrix_multiplication build.

    Args:
        dim_m: Tile rows of A and C (multiple of 8).
        dim_k: Tile columns of A / rows of B (multiple of 8).
        dim_n: Tile columns of B and C (multiple of 8).
        mixed: bf16 A and C with bfp16 B.
    """
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            "mm_bfp: bfp16ebs8 is an AIE2P type; select an NPU2 device"
        )
    for name, v in (("dim_m", dim_m), ("dim_k", dim_k), ("dim_n", dim_n)):
        if v <= 0 or v % 8:
            raise ValueError(
                f"mm_bfp: {name} must be a positive multiple of 8, got {v}"
            )
    flags = [f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}", f"-DDIM_N={dim_n}"]
    b_ty = np.ndarray[(dim_k * dim_n // 8,), np.dtype[v8bfp16ebs8]]
    if mixed:
        source = _default_source_path("mm_bfp_mixed.cc", subdir="aie2p")
        a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]]
        c_ty = np.ndarray[(dim_m * dim_n,), np.dtype[bfloat16]]
        symbol, zero_symbol = (
            "matmul_vectorized_different_datatypes",
            "zero_kernel_bf16",
        )
    else:
        source = _default_source_path("mm_bfp.cc", subdir="aie2p")
        a_ty = np.ndarray[(dim_m * dim_k // 8,), np.dtype[v8bfp16ebs8]]
        c_ty = np.ndarray[(dim_m * dim_n // 8,), np.dtype[v8bfp16ebs8]]
        symbol, zero_symbol = "matmul_vectorized_bfp16", "zero_kernel"
    extern = _make_extern(
        symbol,
        source,
        [a_ty, b_ty, c_ty],
        compile_flags=flags + ["-DMATMUL_ONLY"],
        contract=KernelContract(
            rounding_mode="conv_even",
            roles=("in", "in", "inout"),  # C += A * B; see the .zero sibling
            reference=mm_bfp_mixed_ref if mixed else mm_bfp_ref,
            acc_dtype=np.float32,
            reduction=dim_k,
            tolerance=_BFP_MIXED_TOLERANCE if mixed else _BFP_TOLERANCE,
            ops_per_call=2 * dim_m * dim_k * dim_n,
        ),
    )
    extern.zero = _make_extern(
        zero_symbol, source, [c_ty], compile_flags=flags + ["-DZERO_ONLY"]
    )
    extern.mac_dims = _BFP_MAC_DIMS
    dims = mm_stream_dims(dim_m, dim_k, dim_n, _BFP_MAC_DIMS)
    extern.dims = (dim_m, dim_k, dim_n)
    extern.stream_dims = (
        {"A": dims["A"], "B": None, "C": dims["C"]}
        if mixed
        else {"A": None, "B": None, "C": None}
    )
    # The kernel reads B transposed (8x8 sub-tiles of B^T), so the host B
    # buffer is B^T (N, K), as the block_datatypes examples tile it.
    extern.b_col_maj, extern.c_col_maj = True, False
    return extern


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(mm_bfp, ({"mixed": False}, {"mixed": True}))


def mm_bfp_shuffle(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    *,
    in_shape: tuple | None = None,
    out_shape: tuple | None = None,
) -> ExternalFunction:
    """Scalar shuffle of a bfp16ebs8 tile into (or out of) the mmul block layout (aie2p).

    ``scalar_shuffle(in, out, tile_width, tile_height, unshuffle)`` from
    ``mm_bfp.cc``; the in-core-shuffle block-datatype examples run it before
    ``mm_bfp``. By default the input tile is sized like ``mm_bfp``'s A and
    the output like its C; ``in_shape`` / ``out_shape`` (in ``v8bfp16ebs8``
    blocks) override that, e.g. ``(dim_m, dim_k // 8)`` twice to shuffle an
    A tile in place, matching the ObjectFifo types a design already uses.

    Args:
        dim_m: Tile rows (multiple of 8).
        dim_k: A's tile columns (multiple of 8).
        dim_n: C's tile columns (multiple of 8).
        in_shape: Input tile shape in blocks; default ``(dim_m * dim_k // 8,)``.
        out_shape: Output tile shape in blocks; default ``(dim_m * dim_n // 8,)``.
    """
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            "mm_bfp_shuffle: bfp16ebs8 is an AIE2P type; select an NPU2 device"
        )
    flags = [f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}", f"-DDIM_N={dim_n}"]
    in_shape = tuple(in_shape) if in_shape else (dim_m * dim_k // 8,)
    out_shape = tuple(out_shape) if out_shape else (dim_m * dim_n // 8,)
    in_ty = np.ndarray[in_shape, np.dtype[v8bfp16ebs8]]
    out_ty = np.ndarray[out_shape, np.dtype[v8bfp16ebs8]]
    return _make_extern(
        "scalar_shuffle",
        _default_source_path("mm_bfp.cc", subdir="aie2p"),
        [in_ty, out_ty, np.int16, np.int16, np.int16],
        compile_flags=flags + ["-DSHUFFLE_ONLY"],
        contract=KernelContract(
            roles=("in", "out", "scalar", "scalar", "scalar"),
            reference=mm_bfp_shuffle_ref,
            tolerance=Tolerance.exact(note="a byte permutation"),
            ops_per_call=0,
            unsupported=(
                "streams bfp16ebs8 tiles through a plain fifo, which the "
                "harness samples only for matmul operands"
            ),
        ),
    )


def mha(dim_m: int = 64, dim_k: int = 64, dim_n: int = 64) -> ExternalFunction:
    """Flash-attention toolkit from ``aie_kernels/aie2p/mha.cc`` (aie2p only).

    Not one kernel but one *translation unit*: ``mha.cc`` ``#include``s
    ``softmax.cc`` and ``mm.cc`` and exports the symbols an attention
    dataflow composes, all sharing the ``DIM_M`` / ``DIM_K`` / ``DIM_N``
    micro-tile. This factory compiles it once and binds them together --
    the returned ExternalFunction is the ``QK^T`` matmul, and the rest hang
    off it:

    | Attribute | Symbol | Role |
    | --- | --- | --- |
    | (returned) | ``matmul_bf16_bf16_wrapper`` | ``QK^T``, skipped when the block index says so |
    | ``.matmul_scalar`` | ``matmul_bf16_bf16_wrapper_scalar`` | the scalar fallback |
    | ``.matmul_rowmaj`` | ``matmul_bf16_bf16_rowmaj`` | row-major B and C |
    | ``.zero`` | ``zero_bf16_rowmaj`` | clear an accumulator tile |
    | ``.partial_softmax`` | ``partial_softmax`` | running max / sum over a block |
    | ``.matmul_pv`` | ``matmul_PV`` | ``P @ V`` into the running output |
    | ``.rescale_o`` | ``rescale_O`` | apply the running scale to O |
    | ``.init_scale_buffer`` | ``init_scale_buffer`` | seed the scale buffer |

    ``mha.cc`` also *declares* ``passThroughLine`` without defining it; that
    line copy is its own translation unit, so take it from
    ``passthrough(dtype=np.int32)`` as a second kernel, which is what IRON's
    MHA operator builds too.

    Because the unit includes ``mm.cc``, it defines ``matmul_*`` and
    ``zero_*`` names of its own; the per-parameterisation symbol prefix is
    what keeps those from colliding with a separate ``mm`` kernel in the
    same design.

    The contract declares ``unsupported``: attention is a multi-core
    dataflow with a running softmax, not something the single-Worker
    harness can drive.

    Args:
        dim_m: Rows of the micro-tile (multiple of 16).
        dim_k: Depth of the micro-tile (multiple of 8).
        dim_n: Columns of the micro-tile (multiple of 16).
    """
    if _detect_arch() != "aie2p":
        raise NotImplementedError(
            "mha: mha.cc is an AIE2P kernel; select an NPU2 device"
        )
    for name, v, mult in (
        ("dim_m", dim_m, 16),
        ("dim_k", dim_k, 8),
        ("dim_n", dim_n, 16),
    ):
        if v <= 0 or v % mult:
            raise ValueError(
                f"mha: {name} must be a positive multiple of {mult}, got {v}"
            )
    tile = np.ndarray[(dim_m * dim_n,), np.dtype[bfloat16]]
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(dim_k * dim_n,), np.dtype[bfloat16]]
    idx = np.ndarray[(2,), np.dtype[np.int32]]
    scale = np.ndarray[(dim_m,), np.dtype[bfloat16]]
    flags = [f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}", f"-DDIM_N={dim_n}"]
    extern = _make_extern(
        "matmul_bf16_bf16_wrapper",
        _default_source_path("mha.cc", subdir="aie2p"),
        [a_ty, b_ty, tile, idx],
        compile_flags=flags,
        contract=KernelContract(
            rounding_mode="sets_own",
            roles=("in", "in", "inout", "param"),
            acc_dtype=np.float32,
            reduction=dim_k,
            tolerance=_linalg_tolerance(bfloat16),
            ops_per_call=2 * dim_m * dim_k * dim_n,
            unsupported=(
                "attention is a multi-core dataflow with a running softmax "
                "across blocks; the single-Worker harness cannot drive it"
            ),
        ),
    )
    extern.matmul_scalar = extern.sibling(
        "matmul_bf16_bf16_wrapper_scalar", [a_ty, b_ty, tile]
    )
    extern.matmul_rowmaj = extern.sibling("matmul_bf16_bf16_rowmaj", [a_ty, b_ty, tile])
    extern.zero = extern.sibling("zero_bf16_rowmaj", [tile])
    extern.partial_softmax = extern.sibling(
        "partial_softmax",
        [tile, tile, scale, idx, bfloat16, np.int32, np.int32, np.int32, np.int32],
    )
    extern.matmul_pv = extern.sibling(
        "matmul_PV", [a_ty, b_ty, tile, scale, np.int32, np.int32, idx]
    )
    extern.rescale_o = extern.sibling("rescale_O", [tile, scale, np.int32, idx])
    extern.init_scale_buffer = extern.sibling("init_scale_buffer", [scale, np.int32])
    extern.dims = (dim_m, dim_k, dim_n)
    return extern


def mm_bfp_shuffle_ref(tile, tile_width, tile_height, unshuffle):
    """Numpy reference for [`mm_bfp_shuffle`][iron.kernels.linalg.mm_bfp_shuffle]: :func:`aie.utils.bfp.shuffle` of one tile's bytes.

    ``tile`` is the encoded ``(tile_height, tile_width)`` tile as bytes
    (``tile_width`` in values); returns the reordered bytes.
    """
    from aie.utils import bfp

    w, h = int(tile_width), int(tile_height)
    return bfp.shuffle(tile, w, h, w, h, unshuffle=bool(unshuffle)).ravel()


def cascade_mm(
    dim_m: int = 64,
    dim_k: int = 64,
    dim_n: int = 64,
    input_dtype: type = np.int16,
    output_dtype: type = np.int16,
    use_chess: bool = False,
) -> ExternalFunction:
    r"""Cascade matrix-multiply kernel for multi-core accumulation.

    cascade_mm.cc emits all three cascade variants (``get_only``,
    ``put_only``, ``put_get``) plus a ``zero`` companion in one .o.  The
    returned ExternalFunction binds the ``get_only`` symbol; the other
    three are sibling [`Kernel`][iron.Kernel]\\s available as attributes:

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
        contract=KernelContract(
            roles=("in", "in", "inout"),  # C += A * B; see the .zero sibling
            reference=mm_ref,
            acc_dtype=mm_acc_dtype(input_dtype),
            reduction=dim_k,
            overflow="undefined",
            tolerance=_linalg_tolerance(input_dtype),
            ops_per_call=2 * dim_m * dim_k * dim_n,
            unsupported=(
                "a cascade design: get_only / put_only / put_get pass partial "
                "sums between cores over the cascade stream, which is not an "
                "argument; the reference is the whole product"
            ),
        ),
    )
    extern.get_only = extern
    extern.put_only = extern.sibling(
        f"matmul_scalar_cascade_put_only_{suffix}", [a_ty, b_ty, c_ty]
    )
    extern.put_get = extern.sibling(
        f"matmul_scalar_cascade_put_get_{suffix}", [a_ty, b_ty, c_ty]
    )
    extern.zero = extern.sibling(f"zero_scalar_{_ZERO_SUFFIX[output_dtype]}", [c_ty])
    arch = _detect_arch()
    if arch not in _CASCADE_MM_MAC_DIMS:
        raise ValueError(
            f"cascade_mm(): unsupported arch {arch!r}; "
            f"cascade_mm.cc only ships for {sorted(_CASCADE_MM_MAC_DIMS)}."
        )
    extern.mac_dims = _CASCADE_MM_MAC_DIMS[arch][key]
    return extern
