# kernels/linalg.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Linear algebra kernel factories: mm, mv, cascade_mm."""

from dataclasses import replace
from functools import partial
from typing import NamedTuple, get_args

import numpy as np
from aie.dialects.aiex import v8bfp16ebs8
from aie.iron.dataflow import StreamDims
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, InOut, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    TensorLayout,
    _default_source_path,
    _detect_arch,
    _make_extern,
    dtypes,
)
from .core import conv_even
from .zero import zero

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


# The references above take a whole (M, K) x (K, N) problem, which is what a
# matmul *design* computes and what programming_examples check against. A
# contract's reference is handed one tile per kernel call instead, so the
# per-call forms below sit beside them. Accumulating several products into one
# C is the design's loop, not the kernel's, so these stay one product per call.


def mm_tile_ref(a, b, *, dim_m: int, dim_k: int, dim_n: int):
    """One [`mm`][iron.kernels.linalg.mm] call: a ``(dim_m, dim_k)`` tile times a ``(dim_k, dim_n)`` one.

    Tiles arrive flattened as ``(calls, ...)``, one row per call, and one
    ``(dim_m * dim_n,)`` row comes back per call.
    """
    a = np.asarray(a)
    acc = np.int64 if np.issubdtype(a.dtype, np.integer) else np.float64
    a = a.reshape(-1, dim_m, dim_k).astype(acc)
    b = np.asarray(b).reshape(-1, dim_k, dim_n).astype(acc)
    return (a @ b).reshape(len(a), dim_m * dim_n)


def mv_tile_ref(a, b, *, dim_m: int, dim_k: int):
    """One [`mv`][iron.kernels.linalg.mv] call: a ``(dim_m, dim_k)`` tile times a ``(dim_k,)`` vector."""
    a = np.asarray(a).reshape(-1, dim_m, dim_k).astype(np.int64)
    b = np.asarray(b).reshape(-1, dim_k).astype(np.int64)
    return np.einsum("cmk,ck->cm", a, b)


def mm_bfp_tile_ref(a, b, *, dim_m: int, dim_k: int, dim_n: int, mixed: bool = False):
    """One [`mm_bfp`][iron.kernels.linalg.mm_bfp] call, on operands quantised as the host encodes them.

    Blocks of 8 run along K for both operands, so B is quantised transposed.
    With ``mixed`` the A tile stays bf16 and the core converts it itself, with
    a rounding this does not model -- which is what the wider mixed tolerance
    covers.
    """
    from aie.utils import bfp

    a = np.asarray(a).reshape(-1, dim_m, dim_k)
    b = np.asarray(b).reshape(-1, dim_k, dim_n)
    aq = (
        a.astype(np.float64)
        if mixed
        else bfp.quantize(a.astype(np.float32)).astype(np.float64)
    )
    bt = np.ascontiguousarray(b.astype(np.float32).swapaxes(-1, -2))
    bq = bfp.quantize(bt).swapaxes(-1, -2).astype(np.float64)
    return (aq @ bq).reshape(len(a), dim_m * dim_n)


def _tile_layout(shape, dims=None, *, axes=None, inverse=False):
    """Translate a DMA permutation into an equivalent per-tile host codec."""
    logical = np.arange(np.prod(shape)).reshape(shape)
    if axes is not None:
        logical = logical.transpose(axes)
    order = logical.ravel()
    if dims:
        offsets = np.zeros(1, dtype=np.int64)
        for size, stride in dims:
            offsets = (offsets[:, None] + np.arange(size) * stride).ravel()
        order = order[np.argsort(offsets) if inverse else offsets]
    undo = np.argsort(order)
    return TensorLayout(
        shape,
        pack=lambda x: x.reshape(len(x), -1)[:, order],
        unpack=lambda x: x[:, undo].reshape(len(x), *shape),
    )


def _block_layout(shape, *, axes=None):
    """Describe the block datatype kernels' 8x8 tile storage."""
    from aie.utils import bfp

    def pack(values):
        tiles = []
        for value in values:
            if axes is not None:
                value = value.transpose(axes)
            h, w = value.shape
            tiles.append(
                bfp.shuffle(bfp.encode(np.ascontiguousarray(value)), w, h, w, h).ravel()
            )
        return np.stack(tiles)

    def unpack(values):
        stored_shape = tuple(shape[i] for i in axes) if axes else shape
        h, w = stored_shape
        tiles = []
        for value in values:
            tile = bfp.decode(bfp.shuffle(value, w, h, w, h, unshuffle=True)).reshape(
                stored_shape
            )
            tiles.append(tile.transpose(np.argsort(axes)) if axes else tile)
        return np.stack(tiles)

    return TensorLayout(shape, pack=pack, unpack=unpack)


def _zero_output(fn):
    shape, dtype = get_args(fn.arg_types()[2])
    return zero(shape, get_args(dtype)[0], use_chess=fn.use_chess)


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


class StreamDimsABC(NamedTuple):
    """The three ``dims_to_stream`` a matmul design needs, one per operand.

    ``None`` for an operand a build streams untransformed.
    """

    A: StreamDims | None
    B: StreamDims | None
    C: StreamDims | None


def _blocked(rows: int, cols: int, tile_rows: int, tile_cols: int) -> list:
    """``dims_to_stream`` walking a ``(rows, cols)`` tensor in tile-sized blocks."""
    from aie.helpers.taplib import TensorTiler2D

    tiles = TensorTiler2D.group_tiler(
        (rows, cols), (tile_rows, tile_cols), (rows // tile_rows, cols // tile_cols)
    )
    return list(tiles[0].transformation_dims)


def mm_stream_dims(
    dim_m: int,
    dim_k: int,
    dim_n: int,
    mac_dims,
    *,
    b_col_maj: bool = False,
    c_col_maj: bool = False,
) -> StreamDimsABC:
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
    # Walking an operand as (r x s) blocks is what TensorTiler2D generates, so
    # A and B ask for it rather than restating it.
    a = _blocked(m, k, r, s)
    b = _blocked(n, k, t, s) if b_col_maj else _blocked(k, n, s, t)
    # C is not expressible that way. The DMA reads a core-blocked buffer and
    # writes a differently ordered stream, so the intra-tile row term comes
    # *outside* the tile index -- (r, t) before (n//t, r*t). Every
    # TensorTiler2D classmethod iterates tiles outermost and elements within
    # them, and no combination of tile_col_major / iter_col_major /
    # prune_step produces this order. Closing the gap needs an un-blocking
    # tiler in taplib, which is its own change.
    if c_col_maj:
        c = [(n // t, t * m), (t, r), (m // r, r * t), (r, 1)]
    else:
        c = [(m // r, r * n), (r, t), (n // t, r * t), (t, 1)]
    return StreamDimsABC(A=a, B=b, C=c)


@dtypes(
    tuple({"input_dtype": i, "output_dtype": o} for (i, o) in _MM_MAC_DIMS["aie2p"])
)
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

    Initialize the accumulator with ``kernels.zero(dim_m * dim_n, output_dtype)``.
    The contract declares this independent initializer for the generic harness.

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
            f"mm(): unsupported (input_dtype, output_dtype) = {key}. Supported: {list(_MM_COMBOS.keys())}"
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
            stack_bytes=0xD00,  # programming_examples/basic/matrix_multiplication
            # aie2p/mm.cc sets conv_even itself and restores it; aie2/mm.cc
            # stores bf16 in whatever mode the core is in.
            setup=(conv_even if arch != "aie2p" and output_dtype is bfloat16 else None),
            roles=(In, In, InOut),
            reference=partial(mm_tile_ref, dim_m=dim_m, dim_k=dim_k, dim_n=dim_n),
            initializers=((2, _zero_output),),
            acc_dtype=mm_acc_dtype(input_dtype),
            reduction=dim_k,
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
    # B^T, C emitted as (n, m) tiles of C^T. kernel_design reads
    # these to transpose the host operands.
    extern.b_col_maj = bool(b_col_maj)
    extern.c_col_maj = bool(c_col_maj)
    extern.contract = replace(
        extern.contract,
        layouts=(
            _tile_layout((dim_m, dim_k), extern.stream_dims.A if vectorized else None),
            _tile_layout(
                (dim_k, dim_n),
                extern.stream_dims.B if vectorized else None,
                axes=(1, 0) if b_col_maj else None,
            ),
            _tile_layout(
                (dim_m, dim_n),
                extern.stream_dims.C if vectorized else None,
                axes=(1, 0) if c_col_maj else None,
                inverse=True,
            ),
        ),
    )
    return extern


@dtypes(
    (
        {"input_dtype": np.int16, "output_dtype": np.int32},
        # dim_k: the bf16 kernel accumulates VEC_SIZE (64) elements at a
        # time, so the default dim_k of 32 is too short for it.
        {"input_dtype": bfloat16, "output_dtype": bfloat16, "dim_k": 256},
    )
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
      ``a_dims_from_stream`` publishes. Initialize C with
      ``kernels.zero(dim_m, output_dtype)``.
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
            f"mv(): only (np.int16, np.int32) and (bfloat16, bfloat16) are supported, got ({input_dtype}, {output_dtype})"
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
            roles=(In, In, InOut),
            reference=partial(mv_tile_ref, dim_m=dim_m, dim_k=dim_k),
            initializers=((2, _zero_output),),
            acc_dtype=np.int32,  # acc32
            reduction=dim_k,
            tolerance=Tolerance.exact(note="int16 x int16 accumulated in int32"),
            ops_per_call=2 * dim_m * dim_k,
        ),
    )
    # The vectorized kernel reads A in a "32-bit-word transposed" layout (see
    # aie_kernels/aie2/mv.cc): 2-byte elements are packed two per word, rows
    # of each 2-column word slowly, m rows then the next 2-col word. A design
    # applies this as dims_from_stream on the hop into the core.
    extern.dims = (dim_m, dim_k)
    extern.a_dims_from_stream = (
        [(dim_m, 2), (dim_k // 2, 2 * dim_m), (2, 1)] if vectorized else None
    )
    extern.contract = replace(
        extern.contract,
        layouts=(
            _tile_layout((dim_m, dim_k), extern.a_dims_from_stream, inverse=True),
            TensorLayout((dim_k,)),
            TensorLayout((dim_m,)),
        ),
    )
    return extern


def _mv_bf16(dim_m, dim_k, vectorized, use_chess, vec_size) -> ExternalFunction:
    """bf16 matvec from ``aie_kernels/generic/mv.cc`` (see [`mv`][iron.kernels.linalg.mv])."""
    if vec_size <= 0 or dim_k % vec_size:
        raise ValueError(
            f"mv(): dim_k ({dim_k}) must be a positive multiple of vec_size ({vec_size})"
        )
    prefix = "matvec_vectorized" if vectorized else "matvec_scalar"
    a_ty = np.ndarray[(dim_m * dim_k,), np.dtype[bfloat16]]
    b_ty = np.ndarray[(dim_k,), np.dtype[bfloat16]]
    c_ty = np.ndarray[(dim_m,), np.dtype[bfloat16]]
    extern = _make_extern(
        f"{prefix}_bf16_bf16",
        _default_source_path("mv.cc", subdir="generic"),
        [np.int32, np.int32, a_ty, b_ty, c_ty],
        compile_flags=[f"-DDIM_K={dim_k}", f"-DVEC_SIZE={vec_size}"],
        use_chess=use_chess,
        contract=KernelContract(
            roles=(Param, Param, In, In, Out),
            parameter_bindings=((0, dim_m), (1, 0)),
            layouts=(
                None,
                None,
                TensorLayout((dim_m, dim_k)),
                TensorLayout((dim_k,)),
                TensorLayout((dim_m,)),
            ),
            reference=lambda a, b: np.einsum(
                "cmk,ck->cm", a.astype(np.float32), b.astype(np.float32)
            ),
            acc_dtype=np.float32,  # accfloat, reduced to bf16 on store
            reduction=dim_k,
            tolerance=_linalg_tolerance(bfloat16),
            ops_per_call=2 * dim_m * dim_k,
        ),
    )
    extern.dims = (dim_m, dim_k)
    return extern


def mv_bf16_ref(m, row_offset, a, b):
    """Numpy reference for the bf16 [`mv`][iron.kernels.linalg.mv]: ``a @ b`` over ``m`` rows, accumulated in float32.

    ``a`` is row-major ``(rows, K)``; only the first ``m`` rows are computed,
    and the kernel writes them at ``row_offset`` into ``c``.
    """
    a = np.asarray(a).reshape(-1, np.asarray(b).shape[-1])
    out = a[: int(m)].astype(np.float32) @ np.asarray(b).astype(np.float32)
    return np.concatenate([np.zeros(int(row_offset), np.float32), out])


_BFP_MAC_DIMS = (8, 8, 8)  # the bfp16ebs8 mmul is 8x8x8


@dtypes(({"mixed": False}, {"mixed": True}))
def mm_bfp(
    dim_m: int = 64, dim_k: int = 64, dim_n: int = 64, mixed: bool = False
) -> ExternalFunction:
    """Block-floating-point matmul ``C += A @ B`` on bfp16ebs8 blocks (aie2p only).

    ``mixed=False`` (``aie_kernels/aie2p/mm_bfp.cc``): A, B and C are
    ``v8bfp16ebs8`` blocks, all pre-shuffled into the mmul layout, so no
    DMA transform applies (``stream_dims`` is ``None`` for every operand).
    ``mixed=True`` (``mm_bfp_mixed.cc``): A is bf16 in the (r, s, t)
    micro-tile layout, B is bfp16ebs8, C is bf16; ``stream_dims.A`` and
    ``.C`` carry the transforms and ``.B`` is ``None``.

    Initialize C with the independent ``kernels.zero`` factory.
    The host holds B transposed (``b_col_maj``), and
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
        symbol = "matmul_vectorized_different_datatypes"
    else:
        source = _default_source_path("mm_bfp.cc", subdir="aie2p")
        a_ty = np.ndarray[(dim_m * dim_k // 8,), np.dtype[v8bfp16ebs8]]
        c_ty = np.ndarray[(dim_m * dim_n // 8,), np.dtype[v8bfp16ebs8]]
        symbol = "matmul_vectorized_bfp16"
    extern = _make_extern(
        symbol,
        source,
        [a_ty, b_ty, c_ty],
        compile_flags=flags + ["-DMATMUL_ONLY"],
        contract=KernelContract(
            stack_bytes=0xF00,  # programming_examples/ml/block_datatypes
            setup=conv_even,
            roles=(In, In, InOut),
            reference=partial(
                mm_bfp_tile_ref, dim_m=dim_m, dim_k=dim_k, dim_n=dim_n, mixed=mixed
            ),
            initializers=((2, _zero_output),),
            acc_dtype=np.float32,
            reduction=dim_k,
            tolerance=_BFP_MIXED_TOLERANCE if mixed else _BFP_TOLERANCE,
            ops_per_call=2 * dim_m * dim_k * dim_n,
        ),
    )
    extern.mac_dims = _BFP_MAC_DIMS
    dims = mm_stream_dims(dim_m, dim_k, dim_n, _BFP_MAC_DIMS)
    extern.dims = (dim_m, dim_k, dim_n)
    extern.stream_dims = (
        StreamDimsABC(A=dims.A, B=None, C=dims.C)
        if mixed
        else StreamDimsABC(A=None, B=None, C=None)
    )
    # The kernel reads B transposed (8x8 sub-tiles of B^T), so the host B
    # buffer is B^T (N, K), as the block_datatypes examples tile it.
    extern.b_col_maj, extern.c_col_maj = True, False
    extern.contract = replace(
        extern.contract,
        layouts=(
            (
                _tile_layout((dim_m, dim_k), dims.A)
                if mixed
                else _block_layout((dim_m, dim_k))
            ),
            _block_layout((dim_k, dim_n), axes=(1, 0)),
            (
                _tile_layout((dim_m, dim_n), dims.C, inverse=True)
                if mixed
                else _block_layout((dim_m, dim_n))
            ),
        ),
    )
    return extern


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
    from aie.utils import bfp

    logical_shape = (dim_m, dim_k)
    plain = TensorLayout(
        logical_shape,
        pack=lambda x: bfp.encode(x).reshape(len(x), -1),
        unpack=lambda x: bfp.decode(x).reshape(len(x), *logical_shape),
    )
    blocked = _block_layout(logical_shape)
    extern = _make_extern(
        "scalar_shuffle",
        _default_source_path("mm_bfp.cc", subdir="aie2p"),
        [in_ty, out_ty, np.int16, np.int16, np.int16],
        compile_flags=flags + ["-DSHUFFLE_ONLY"],
        contract=KernelContract(
            roles=(In, Out, Param, Param, Param),
            parameter_bindings=((2, dim_k), (3, dim_m), (4, 0)),
            layouts=(
                plain,
                blocked,
                None,
                None,
                None,
            ),
            reference=lambda x: bfp.quantize(x),
            tolerance=Tolerance.exact(note="byte permutation preserves encoded values"),
            ops_per_call=0,
            unsupported=(
                None
                if np.prod(in_shape) == np.prod(out_shape) == dim_m * dim_k // 8
                else "independent shuffle validation requires both buffers to hold "
                "exactly the dim_m by dim_k tile; custom unequal buffers need "
                "their enclosing design's runtime dimensions"
            ),
        ),
    )
    return extern


def mha(dim_m: int = 64, dim_k: int = 64, dim_n: int = 64) -> ExternalFunction:
    """Flash-attention toolkit from ``aie_kernels/aie2p/mha.cc`` (aie2p only).

    Not one kernel but one *translation unit*: ``mha.cc`` ``#include``s
    ``softmax.cc`` and ``mm.cc`` and exports the symbols an attention
    dataflow composes, all sharing the ``DIM_M`` / ``DIM_K`` / ``DIM_N``
    micro-tile. The returned ExternalFunction is the ``QK^T`` matmul.
    Bind additional entry points explicitly with
    ``fn.object_file.bind(symbol, arg_types)``: ``matmul_bf16_bf16_wrapper_scalar``,
    ``matmul_bf16_bf16_rowmaj``, ``partial_softmax``, ``matmul_PV``,
    ``rescale_O`` and ``init_scale_buffer``. Clear tiles with ``kernels.zero``.

    ``mha.cc`` also *declares* ``passThroughLine`` without defining it; that
    line copy is its own translation unit, so take it from
    ``passthrough(dtype=np.int32)`` as a second kernel, which is what IRON's
    MHA operator builds too.

    Because the unit includes ``mm.cc``, it defines ``matmul_*``
    names of its own; the per-parameterisation symbol prefix is
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
    flags = [f"-DDIM_M={dim_m}", f"-DDIM_K={dim_k}", f"-DDIM_N={dim_n}"]
    extern = _make_extern(
        "matmul_bf16_bf16_wrapper",
        _default_source_path("mha.cc", subdir="aie2p"),
        [a_ty, b_ty, tile, idx],
        compile_flags=flags,
        contract=KernelContract(
            roles=(In, In, InOut, Param),
            acc_dtype=np.float32,
            reduction=dim_k,
            tolerance=_linalg_tolerance(bfloat16),
            ops_per_call=2 * dim_m * dim_k * dim_n,
            unsupported=(
                "attention is a multi-core dataflow with a running softmax across blocks; the single-Worker harness cannot drive it"
            ),
        ),
    )
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
    ``put_only``, ``put_get``) in one object. The returned ExternalFunction
    binds ``get_only``; bind the other entries explicitly with
    ``fn.object_file.bind("matmul_scalar_cascade_put_only_<dtype>", fn.arg_types())``
    (or ``put_get``). Initialize accumulators with ``kernels.zero``.

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
            f"cascade_mm(): unsupported (input_dtype, output_dtype) = {key}. Supported: {list(_CASCADE_COMBOS.keys())}"
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
            roles=(In, In, InOut),
            reference=mm_ref,
            initializers=((2, _zero_output),),
            acc_dtype=mm_acc_dtype(input_dtype),
            reduction=dim_k,
            tolerance=_linalg_tolerance(input_dtype),
            ops_per_call=2 * dim_m * dim_k * dim_n,
            unsupported=(
                "a cascade design: get_only / put_only / put_get pass partial "
                "sums between cores over the cascade stream, which is not an "
                "argument; the reference is the whole product"
            ),
        ),
    )
    arch = _detect_arch()
    if arch not in _CASCADE_MM_MAC_DIMS:
        raise ValueError(
            f"cascade_mm(): unsupported arch {arch!r}; cascade_mm.cc only ships for {sorted(_CASCADE_MM_MAC_DIMS)}."
        )
    extern.mac_dims = _CASCADE_MM_MAC_DIMS[arch][key]
    return extern
