# kernels/vision.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vision kernel factories: color conversion, threshold, filter2d, add_weighted,
and the filter tables of torch's antialiased bicubic resize."""

import math
from dataclasses import dataclass

import numpy as np
from aie.iron.kernel import ExternalFunction, Kernel
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance
from ml_dtypes import bfloat16

from ._common import (
    KernelContract,
    Param,
    Trace,
    _dtype_to_bit_width,
    _kernel_source,
    _make_extern,
    _require_vector_alignment,
    _runtime_lib_include,
    _tuned_arch,
    dtypes,
)


def _color_convert_kernel(
    func_name: str,
    filename: str,
    in_size: int,
    out_size: int,
    use_chess: bool = False,
    compile_flags: list[str] | None = None,
    contract: KernelContract | None = None,
) -> ExternalFunction:
    """Shared implementation for color-space conversion line kernels."""
    in_ty = np.ndarray[(in_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]
    return _make_extern(
        func_name,
        _kernel_source(f"vision/{filename}"),
        [in_ty, out_ty, np.int32],
        compile_flags=compile_flags,
        use_chess=use_chess,
        contract=contract,
    )


def _bitwise_kernel(
    op: str, line_width: int, dtype, use_chess: bool = False
) -> ExternalFunction:
    """Shared implementation for [`bitwise_or`][iron.kernels.vision.bitwise_or] and [`bitwise_and`][iron.kernels.vision.bitwise_and]."""
    bit_width = _dtype_to_bit_width(dtype, factory_name=f"bitwise{op}")
    _require_vector_alignment(
        f"bitwise{op}", line_width, 512 // bit_width, param="line_width"
    )
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        f"bitwise{op}Line",
        _kernel_source(f"vision/bitwise{op}.cc"),
        [line_ty, line_ty, line_ty, np.int32],
        compile_flags=[f"-DBIT_WIDTH={bit_width}", f"-DBITWISE_ELEMS={line_width}"],
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Param),
            parameter_bindings=((3, line_width),),
            reference={"OR": bitwise_or_ref, "AND": bitwise_and_ref}[op],
            tolerance=Tolerance.exact(note="bitwise op"),
        ),
    )


def rgba2hue(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert a line of RGBA pixels to hue values (full-range, 0..255)."""
    _require_vector_alignment("rgba2hue", line_width, 32, param="line_width")
    # lut_inv.h pins its gather pair with AIE_BANK_A/AIE_BANK_B.
    flags = [f"-I{_runtime_lib_include()}"]
    return _color_convert_kernel(
        "rgba2hueLine",
        "rgba2hue.cc",
        line_width * 4,
        line_width,
        use_chess=use_chess,
        compile_flags=flags,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, line_width),),
            reference=rgba2hue_ref,
            acc_dtype=np.int32,
            reduction=1,
            tolerance=Tolerance.exact(note="integer reciprocal, no rounding slack"),
            uses_lut=True,
        ),
    )


@dtypes(({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32}))
def threshold(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Apply a threshold operation to a line of pixels.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type, ``np.uint8`` or ``np.int16``. The source's
            32-bit branch multiplies int32 data by int16 coefficients, a MAC
            AIE2 does not have, so it does not compile and is not offered.
        use_chess: When ``True``, build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8``, ``np.int16``, or ``np.int32``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="threshold")
    _require_vector_alignment(
        "threshold", line_width, 512 // bit_width, param="line_width"
    )
    scalar_ty = np.int32 if bit_width == 32 else np.int16
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "thresholdLine",
        _kernel_source("vision/threshold.cc"),
        [line_ty, line_ty, np.int32, scalar_ty, scalar_ty, np.int8],
        compile_flags=[f"-DBIT_WIDTH={bit_width}", f"-DTHRESHOLD_ELEMS={line_width}"],
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param, Param, Param),
            parameter_bindings=((2, line_width),),
            reference=threshold_ref,
            tolerance=Tolerance.exact(note="selection"),
        ),
    )


@dtypes(({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32}))
def bitwise_or(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise OR of two lines."""
    return _bitwise_kernel("OR", line_width, dtype, use_chess=use_chess)


@dtypes(({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32}))
def bitwise_and(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise AND of two lines."""
    return _bitwise_kernel("AND", line_width, dtype, use_chess=use_chess)


def gray2rgba(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert a grayscale line to RGBA."""
    return _color_convert_kernel(
        "gray2rgbaLine",
        "gray2rgba.cc",
        line_width,
        line_width * 4,
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, line_width),),
            reference=gray2rgba_ref,
            tolerance=Tolerance.exact(note="copy with alpha = 255"),
        ),
    )


def rgba2gray(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert an RGBA line to grayscale."""
    flags = []
    if not use_chess and _tuned_arch() == "aie2p":
        # The pre-RA pipeliner's schedule of the 64-pixel loop ends up at II13
        # after register allocation; the postpipeliner finds II11.
        flags += ["-mllvm", "--aie-force-postpipeliner"]
    return _color_convert_kernel(
        "rgba2grayLine",
        "rgba2gray.cc",
        line_width * 4,
        line_width,
        use_chess=use_chess,
        compile_flags=flags,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param),
            parameter_bindings=((2, line_width),),
            reference=rgba2gray_ref,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            acc_dtype=np.int32,
            reduction=3,
        ),
    )


def filter2d(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Apply a 3x3 2D convolution filter across three input lines."""
    if line_width % 32 or line_width < 96:
        raise ValueError(
            f"filter2d: line_width must be a multiple of 32 and at least 96 "
            f"(the vector kernel handles the two borders in 32-pixel blocks), "
            f"got {line_width}"
        )
    line_ty = np.ndarray[(line_width,), np.dtype[np.uint8]]
    kernel_ty = np.ndarray[(3, 3), np.dtype[np.int16]]
    return _make_extern(
        "filter2dLine",
        _kernel_source("vision/filter2d.cc"),
        [line_ty, line_ty, line_ty, line_ty, np.int32, kernel_ty],
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, In, Out, Param, Param),
            parameter_bindings=((4, line_width),),
            reference=filter2d_ref,
            acc_dtype=np.int32,
            reduction=9,
            # The one-LSB slack these pixel kernels used to share was
            # absorbing a wrong carry across the 32-pixel boundary here (see
            # filter2d.cc); an exact contract is what would have caught it.
            tolerance=Tolerance.exact(
                note="vector path matches the reference bit-for-bit"
            ),
            ops_per_call=18 * line_width,
        ),
    )


@dtypes(({"dtype": np.uint8}, {"dtype": np.int16}))
def add_weighted(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Weighted addition of two lines with a gamma offset.

    Args:
        line_width: Number of elements per line.
        dtype: Element data type, ``np.uint8`` or ``np.int16``. The source's
            32-bit branch multiplies int32 data by int16 coefficients, a MAC
            AIE2 does not have, so it does not compile and is not offered.
        use_chess: When ``True``, build the .o with ``xchesscc_wrapper``
            instead of Peano.

    Raises:
        ValueError: When ``dtype`` is not ``np.uint8`` or ``np.int16``.
    """
    bit_width = _dtype_to_bit_width(dtype, factory_name="add_weighted")
    if bit_width == 32:
        raise ValueError(
            "add_weighted: no int32 build; addWeighted.cc has no int32 x int16 MAC. "
            "Use np.uint8 or np.int16."
        )
    gamma_ty = {8: np.int8, 16: np.int16}[bit_width]
    _require_vector_alignment(
        "add_weighted", line_width, 256 // bit_width, param="line_width"
    )
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "addWeightedLine",
        _kernel_source("vision/addWeighted.cc"),
        [line_ty, line_ty, line_ty, np.int32, np.int16, np.int16, gamma_ty],
        compile_flags=[
            f"-DBIT_WIDTH={bit_width}",
            f"-DADD_WEIGHTED_ELEMS={line_width}",
        ],
        use_chess=use_chess,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, Out, Param, Param, Param, Param),
            parameter_bindings=((3, line_width),),
            reference=add_weighted_ref,
            acc_dtype=np.int32,
            reduction=2,
            tolerance=Tolerance.exact(
                note="measured bit-exact against the reference over every data case"
            ),
            ops_per_call=3 * line_width,
        ),
    )


def _check_resample(owner: str, words: int, cores: int, slots: int | None) -> None:
    if words < _RESAMPLE_HEADER + 5 or cores < 1 or (slots is not None and slots < 1):
        raise ValueError(
            f"{owner}() needs words >= {_RESAMPLE_HEADER + 5} (a header and one "
            "slot of the smallest window), cores >= 1 and slots None or >= 1, "
            f"got words={words}, cores={cores}, slots={slots}."
        )


class _ResamplePeakKernel(ExternalFunction):
    resample_join: Kernel


class _ResizeKernel(ExternalFunction):
    resize_setup: Kernel
    resize_take: Kernel
    resize_band: Kernel
    resize_emit: Kernel
    resize_finish: Kernel


def resample_peak(
    words: int = 256, *, cores: int = 16, slots: int | None = None
) -> _ResamplePeakKernel:
    """The largest normalized weight of one core's share of a resample table.

    One axis of torch's antialiased bicubic resize (``in`` samples to
    ``out``) is a table of ``words``-int32 chunks, ``cores`` cores taking
    chunks ``core, core + cores, ...``; see
    [`resample_quantize`][iron.kernels.vision.resample_quantize].
    ``resample_peak(peak, in, out, core, chunks)`` writes the largest
    normalized weight over the outputs of the core's ``chunks`` chunks as
    float64 bits in ``peak`` (two int32), 0 when the table cannot hold the
    sizes. ``fn.resample_join(a, b, out)`` writes the larger of two peaks,
    so a chain of cores reduces them to the table's precision. The weights
    are float64 without contraction, as torch computes them on the CPU;
    amd/IRON's ``ResampleTaps`` runs both for EmbeddingGemma 2's image
    processor.

    Args:
        words: The int32 words of a chunk.
        cores: The cores the chunks are dealt to.
        slots: The outputs a chunk is fixed to, or ``None`` for as many as
            the window lets fit.

    Returns:
        ExternalFunction for ``resample_peak``, with ``resample_join`` bound
        on its object.

    Raises:
        ValueError: When ``words`` holds no slot, ``cores`` is below 1, or
            ``slots`` is below 1.
    """
    _check_resample("resample_peak", words, cores, slots)
    peak_ty = np.ndarray[(2,), np.dtype[np.int32]]
    fn = _make_extern(
        "resample_peak",
        _kernel_source("vision/resample_peak.cc"),
        [peak_ty] + [np.int32] * 4,
        compile_flags=[f"-DWORDS={words}", f"-DCORES={cores}", f"-DPER={slots or 0}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, Param, Param, Param, Param),
            reference=lambda in_size, out_size, core, chunks: resample_peak_ref(
                in_size, out_size, core, chunks, words=words, cores=cores, slots=slots
            ),
            tolerance=Tolerance.exact(note="IEEE float64, as torch on the CPU"),
            ops_per_call=0,
        ),
        cls=_ResamplePeakKernel,
    )
    fn.resample_join = fn.object_file.bind("resample_join", [peak_ty, peak_ty, peak_ty])
    return fn


def resample_quantize(
    words: int = 256, *, cores: int = 16, slots: int | None = None
) -> ExternalFunction:
    """One chunk of the int16 filter table torch resamples an axis with.

    ``resample_quantize(peak, chunk, in, out, core, k, chunks)`` writes
    chunk ``core + k * cores`` of the table of ``in`` samples to ``out``
    whose ``cores`` cores each write ``chunks`` chunks. A chunk is ``words``
    int32: a header ``[precision, window, first, outputs]``, then
    ``outputs`` slots from output ``first`` on, each ``[start, count]`` and
    the window's int16 weights two to a word, the rest zero. The precision
    is the one the largest normalized weight ``peak`` (float64 bits, from
    [`resample_peak`][iron.kernels.vision.resample_peak]) sets, as torch
    picks it. A size the table cannot hold (a size below 1, a window no
    slot of ``words`` holds, too few chunks) writes ``[-1, window, 0, 0]``.

    Args:
        words: The int32 words of a chunk.
        cores: The cores the chunks are dealt to.
        slots: The outputs a chunk is fixed to, or ``None`` for as many as
            the window lets fit.

    Returns:
        ExternalFunction for ``resample_quantize``.

    Raises:
        ValueError: When ``words`` holds no slot, ``cores`` is below 1, or
            ``slots`` is below 1.
    """
    _check_resample("resample_quantize", words, cores, slots)
    return _make_extern(
        "resample_quantize",
        _kernel_source("vision/resample_quantize.cc"),
        [
            np.ndarray[(2,), np.dtype[np.int32]],
            np.ndarray[(words,), np.dtype[np.int32]],
        ]
        + [np.int32] * 5,
        compile_flags=[f"-DWORDS={words}", f"-DCORES={cores}", f"-DPER={slots or 0}"],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Out, Param, Param, Param, Param, Param),
            parameter_bindings=((5, 0),),
            call_index=5,
            sample=_resample_peaks,
            reference=lambda peak, in_size, out_size, core, chunks: (
                resample_quantize_ref(
                    peak,
                    in_size,
                    out_size,
                    core,
                    chunks,
                    words=words,
                    cores=cores,
                    slots=slots,
                )
            ),
            tolerance=Tolerance.exact(note="IEEE float64, as torch on the CPU"),
            ops_per_call=0,
        ),
    )


def resize(
    words: int = 356, *, chunk: int = 4096, patch_columns: int = 8, cores: int = 16
) -> _ResizeKernel:
    """A uint8 RGB image resized as torch's antialiased bicubic resize, as patches.

    The image is resized across, then down, each pass rounded to uint8 as
    torch does it on the CPU, and written as 16x16 patches of ``u8 / 255``
    in bf16, channels last. The filters are
    [`resample_quantize`][iron.kernels.vision.resample_quantize] tables at
    16 slots, one chunk a patch column (width) or a band of 16 rows
    (height). ``cores`` cores receive every ``chunk``-byte image chunk, each
    image row padded to whole chunks; core ``c`` owns patch columns ``c, c +
    cores, ...``, at most ``patch_columns`` of them. Six entry points share
    one core's state, and a core runs:

    ```text
    resize_setup(counts, rows, columns, out_rows, out_columns, c)
    for k in range(counts[0]): resize_take(width chunk k, k)
    for each of counts[1] bands:
        resize_band(height chunk, counts)
        for counts[2] image chunks: resize_consume(image chunk, height chunk)
        for i in range(counts[3]): resize_emit(patch, i)
    resize_finish(counts)
    then counts[4] image chunks, unread
    ```

    ``counts`` is five int32. Band ``b``'s patch ``i`` is patch column ``c
    + i * cores``, zeros past the image: a core writes ``out_columns // 16
    // cores + 1`` patches a band, so the last patch column of every band is
    zeros. The counts are the same on every core whatever the input, so the
    broadcast streams stay in step; sizes or a table the kernel refuses
    write zeros from there on, where
    [`resize_ref`][iron.kernels.vision.resize_ref] says. amd/IRON's
    ``Resample`` runs it for EmbeddingGemma 2's image processor.

    Args:
        words: The int32 words of a table chunk; a 16-slot chunk holds a
            window of ``2 * ((words - 4) // 16 - 2) - 1`` taps.
        chunk: The bytes of an image chunk.
        patch_columns: The most patch columns one core holds.
        cores: The cores the patch columns are dealt to.

    Returns:
        ExternalFunction for ``resize_consume(image_chunk, height_chunk)``,
        with ``resize_setup``, ``resize_take``, ``resize_band``,
        ``resize_emit`` and ``resize_finish`` bound on its object.

    Raises:
        ValueError: When ``words`` holds no window of 1 to 64 taps,
            ``chunk`` is not a multiple of 4 from 4 to 98300, or
            ``patch_columns`` or ``cores`` is below 1.
    """
    window = 2 * ((words - _RESAMPLE_HEADER) // _RESIZE_SIDE - 2) - 1
    if not 1 <= window <= 64:
        raise ValueError(
            f"resize() needs words from 52 to 563 (a window of 1 to 64 taps), "
            f"got words={words}."
        )
    if chunk % 4 or not 4 <= chunk < 98304:
        raise ValueError(
            f"resize() needs chunk a multiple of 4 from 4 to 98300, got chunk={chunk}."
        )
    if patch_columns < 1 or cores < 1:
        raise ValueError(
            "resize() needs patch_columns >= 1 and cores >= 1, got "
            f"patch_columns={patch_columns}, cores={cores}."
        )
    counts_ty = np.ndarray[(5,), np.dtype[np.int32]]
    taps_ty = np.ndarray[(words,), np.dtype[np.int32]]
    fn = _make_extern(
        "resize_consume",
        _kernel_source("vision/resize.cc"),
        [np.ndarray[(chunk,), np.dtype[np.uint8]], taps_ty],
        compile_flags=[
            f"-DWORDS={words}",
            f"-DCHUNK={chunk}",
            f"-DCOLS={patch_columns}",
            f"-DCORES={cores}",
        ],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In),
            unsupported=(
                "six entry points share one core's state, driven by counts "
                "they write; test_resize_e2e.py drives and judges them"
            ),
            ops_per_call=0,
        ),
        cls=_ResizeKernel,
    )
    lib = fn.object_file
    fn.resize_setup = lib.bind(
        "resize_setup",
        [counts_ty, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    fn.resize_take = lib.bind("resize_take", [taps_ty, np.int32])
    fn.resize_band = lib.bind("resize_band", [taps_ty, counts_ty])
    fn.resize_emit = lib.bind(
        "resize_emit",
        [np.ndarray[(_RESIZE_SIDE * _RESIZE_SIDE * 3,), np.dtype[bfloat16]], np.int32],
    )
    fn.resize_finish = lib.bind("resize_finish", [counts_ty])
    return fn


# --------------------------------------------------------------------------
# Numpy references. Each follows the *vector* path of its kernel (the one the
# ``*Line`` entry points call), read off aie_kernels/vision/*.cc.
# --------------------------------------------------------------------------


def gray2rgba_ref(y):
    """Numpy reference for [`gray2rgba`][iron.kernels.vision.gray2rgba]: ``(y, y, y, 255)`` per pixel."""
    y = np.asarray(y, dtype=np.uint8)
    out = np.empty(y.shape[:-1] + (y.shape[-1] * 4,), dtype=np.uint8)
    px = out.reshape(*y.shape, 4)
    px[..., 0] = px[..., 1] = px[..., 2] = y
    px[..., 3] = 255
    return out


# Q0.15 weights of rgba2gray.cc's vector path (BT.470 luma), plus its 2**14
# rounding term before the shift by 15.
_GRAY_WEIGHTS = (round(0.299 * 2**15), round(0.587 * 2**15), round(0.114 * 2**15))


def rgba2gray_ref(rgba):
    """Numpy reference for [`rgba2gray`][iron.kernels.vision.rgba2gray]: fixed-point BT.470 luma.

    ``Y = (9798 R + 19235 G + 3736 B + 2**14) >> 15``, saturated to ``uint8``;
    alpha is ignored. Within one LSB of the kernel (rounding of the final
    shift).
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    px = rgba.reshape(*rgba.shape[:-1], -1, 4).astype(np.int64)
    wr, wg, wb = _GRAY_WEIGHTS
    acc = px[..., 0] * wr + px[..., 1] * wg + px[..., 2] * wb + (1 << 14)
    return np.clip(acc >> 15, 0, 255).astype(np.uint8)


def rgba2hue_ref(rgba):
    """Numpy reference for [`rgba2hue`][iron.kernels.vision.rgba2hue]: full-range hue.

    ``rgba2hue.cc`` multiplies by a Q7.9 reciprocal rather than dividing, so
    with ``d = max - min`` of R, G, B and ``inv = 85 * 512 / d`` the hue is
    ``(offset * 512 + c * inv) >> 10`` for whichever channel holds the max:
    ``c = G - B`` at offset 1, ``B - R`` at 171, ``R - G`` at 341.
    Each offset carries the ``+ 1`` that rounds the final halving, so there is
    one rounding step rather than two. The cast to ``uint8`` wraps, so a
    negative hue (R max, G < B) comes out as ``256 + h`` -- the right circular
    value. Gray pixels (``d == 0``) are hue 0, and a max held by both G and R
    goes to G, as the kernel's select order does. ``inv`` truncates, which
    leaves hue up to one LSB below the exact value.
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    px = rgba.reshape(*rgba.shape[:-1], -1, 4).astype(np.int64)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    d = mx - np.minimum(np.minimum(r, g), b)
    inv = (85 * 512) // np.where(d == 0, 1, d)  # avoid /0; masked to 0 below
    h = np.where(
        mx == g,
        (171 * 512 + (b - r) * inv) >> 10,
        np.where(
            mx == r,
            (1 * 512 + (g - b) * inv) >> 10,
            (341 * 512 + (r - g) * inv) >> 10,
        ),
    )
    return (np.where(d == 0, 0, h) & 0xFF).astype(np.uint8)


def threshold_ref(x, thresh, maxval, ttype):
    """Numpy reference for [`threshold`][iron.kernels.vision.threshold] (OpenCV semantics).

    ``ttype``: 0 binary (``x > thresh ? maxval : 0``), 1 binary inverted,
    2 truncate (``min(x, thresh)``), 3 to-zero (``x > thresh ? x : 0``),
    4 to-zero inverted. Exact.
    """
    x = np.asarray(x)
    t, mv, zero = x.dtype.type(thresh), x.dtype.type(maxval), x.dtype.type(0)
    above = x > t
    ttype = int(ttype)
    if ttype == 0:
        return np.where(above, mv, zero)
    if ttype == 1:
        return np.where(above, zero, mv)
    if ttype == 2:
        return np.minimum(x, t)
    if ttype == 3:
        return np.where(above, x, zero)
    if ttype == 4:
        return np.where(above, zero, x)
    raise ValueError(f"threshold type must be 0..4, got {ttype}")


def bitwise_or_ref(a, b):
    """Numpy reference for [`bitwise_or`][iron.kernels.vision.bitwise_or]. Exact."""
    return np.bitwise_or(np.asarray(a), np.asarray(b))


def bitwise_and_ref(a, b):
    """Numpy reference for [`bitwise_and`][iron.kernels.vision.bitwise_and]. Exact."""
    return np.bitwise_and(np.asarray(a), np.asarray(b))


def add_weighted_ref(a, b, alpha, beta, gamma):
    """Numpy reference for [`add_weighted`][iron.kernels.vision.add_weighted]: Q2.14 blend.

    ``out = sat(((alpha * a + beta * b) >> 14) + gamma)`` with ``alpha`` and
    ``beta`` as Q2.14 fixed point (``8192`` is 0.5) and ``gamma`` in output
    units, as OpenCV's ``addWeighted`` has it; the kernel reads ``gamma`` as
    the data type, so for ``uint8`` data ``-56`` is ``200``. The vector path in
    ``addWeighted.cc`` rounds the shift down; its scalar path rounds to
    nearest, so the two differ by at most one LSB.
    """
    a = np.asarray(a)
    info = np.iinfo(a.dtype)
    acc = a.astype(np.int64) * int(alpha) + np.asarray(b).astype(np.int64) * int(beta)
    acc = (acc >> 14) + int(np.array(gamma).astype(a.dtype))
    return np.clip(acc, info.min, info.max).astype(a.dtype)


def filter2d_ref(line0, line1, line2, kernel):
    """Numpy reference for [`filter2d`][iron.kernels.vision.filter2d]: 3x3 correlation of three lines.

    Produces the middle line. The vector path keeps only the top byte of
    each ``int16`` coefficient (``k >> 8`` as ``int8``) and shifts the sum by
    4, so a Q4.12 kernel such as ``4096 * [[0, 1, 0], [1, -4, 1], [0, 1, 0]]``
    lands on integer taps. Borders replicate the edge pixel; the result is
    saturated to ``uint8``. Within one LSB of the kernel.
    """
    k8 = (np.asarray(kernel, dtype=np.int16) >> 8).astype(np.int8).astype(np.int64)
    k8 = k8.reshape(3, 3)
    terms = []
    for r, line in enumerate((line0, line1, line2)):
        v = np.asarray(line, dtype=np.uint8).astype(np.int64)
        left = np.concatenate([v[..., :1], v[..., :-1]], axis=-1)
        right = np.concatenate([v[..., 1:], v[..., -1:]], axis=-1)
        terms.append(left * k8[r, 0] + v * k8[r, 1] + right * k8[r, 2])
    acc = terms[0] + terms[1] + terms[2]
    return np.clip(acc >> 4, 0, 255).astype(np.uint8)


# int32 words of a resample chunk's header (resample.h).
_RESAMPLE_HEADER = 4


def _bicubic(x: float) -> float:
    a = -0.5
    x = abs(x)
    if x < 1.0:
        return ((a + 2) * x - (a + 3)) * x * x + 1
    if x < 2.0:
        return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return 0.0


@dataclass(frozen=True)
class _ResampleAxis:
    """One axis as ``resample.h``'s ``axis()`` sees it, every step in float64."""

    in_size: int
    out_size: int
    scale: float = 0.0
    support: float = 0.0
    invscale: float = 0.0
    window: int = 0
    slot: int = 0
    per: int = 0
    ok: bool = False

    @classmethod
    def of(cls, in_size, out_size, chunks, *, words, slots):
        if in_size < 1 or out_size < 1:
            return cls(in_size, out_size)
        scale = in_size / out_size
        support = 2.0 * scale if scale >= 1.0 else 2.0
        window = 2 * math.ceil(support) + 1
        slot = 2 + (window + 1) // 2
        per = (words - _RESAMPLE_HEADER) // slot
        if slots:
            per = slots if slots <= per else 0
        return cls(
            in_size,
            out_size,
            scale,
            support,
            1.0 / scale if scale >= 1.0 else 1.0,
            window,
            slot,
            per,
            per >= 1 and per * chunks >= out_size,
        )

    def taps(self, i: int) -> tuple[int, list[float], float]:
        """Output ``i``'s first sample, unnormalized weights and their sum."""
        center = self.scale * (i + 0.5)
        xmin = max(int(center - self.support + 0.5), 0)
        hi = min(int(center + self.support + 0.5), self.in_size)
        xsize = min(max(hi - xmin, 0), self.window)
        weights = [
            _bicubic((float(j + xmin) - center + 0.5) * self.invscale)
            for j in range(xsize)
        ]
        # In order: Python's sum() of floats compensates.
        total = 0.0
        for v in weights:
            total += v
        return xmin, weights, total


def _resample_peaks(rng, calls):
    # Mostly a real peak's range; first no peak, then each side of the
    # precision step at 2**-15 * 32767.5.
    peaks = 2.0 ** rng.uniform(-6.0, 1.0, calls)
    edges = [0.0, 32767.5 / 2**15, np.nextafter(32767.5 / 2**15, 0.0)]
    peaks[: min(calls, len(edges))] = edges[:calls]
    return [peaks.astype(np.float64).view(np.int32).reshape(calls, 2)]


def resample_peak_ref(
    in_size: int,
    out_size: int,
    core: int,
    chunks: int,
    *,
    words: int = 256,
    cores: int = 16,
    slots: int | None = None,
) -> np.ndarray:
    """Numpy reference for [`resample_peak`][iron.kernels.vision.resample_peak].

    Args:
        in_size: The input samples on the axis.
        out_size: The output samples on the axis.
        core: The core whose chunks ``core, core + cores, ...`` are read.
        chunks: The chunks each core writes.
        words: The int32 words of a chunk.
        cores: The cores the chunks are dealt to.
        slots: The outputs a chunk is fixed to, or ``None``.

    Returns:
        The peak's float64 bits as two int32.
    """
    axis = _ResampleAxis.of(in_size, out_size, chunks * cores, words=words, slots=slots)
    m = 0.0
    for k in range(chunks if axis.ok else 0):
        first = (core + k * cores) * axis.per
        for i in range(first, min(first + axis.per, out_size)):
            _, weights, total = axis.taps(i)
            if total == 0.0:
                continue
            e = weights[0]
            for v in weights[1:]:
                if (e < v) if total > 0.0 else (v < e):
                    e = v
            if m < e / total:
                m = e / total
    return np.array([m], np.float64).view(np.int32)


def resample_quantize_ref(
    peak: np.ndarray,
    in_size: int,
    out_size: int,
    core: int,
    chunks: int,
    *,
    words: int = 256,
    cores: int = 16,
    slots: int | None = None,
) -> np.ndarray:
    """Numpy reference for [`resample_quantize`][iron.kernels.vision.resample_quantize].

    Args:
        peak: ``(calls, 2)`` int32, each call's peak as float64 bits; call
            ``k`` writes chunk ``core + k * cores``.
        in_size: The input samples on the axis.
        out_size: The output samples on the axis.
        core: The core whose chunks are written.
        chunks: The chunks each core writes.
        words: The int32 words of a chunk.
        cores: The cores the chunks are dealt to.
        slots: The outputs a chunk is fixed to, or ``None``.

    Returns:
        ``(calls, words)`` int32, a chunk per call. A weight past int16
        wraps, as the kernel's store does.
    """
    peaks = np.ascontiguousarray(peak, np.int32).reshape(-1, 2).view(np.float64)
    axis = _ResampleAxis.of(in_size, out_size, chunks * cores, words=words, slots=slots)
    table = np.zeros((len(peaks), words), np.int32)
    for k, chunk in enumerate(table):
        if not axis.ok:
            chunk[:2] = -1, axis.window
            continue
        wt_max = float(peaks[k, 0])
        p = next(
            (p for p in range(22) if int(0.5 + wt_max * (1 << (p + 1))) >= 1 << 15),
            22,
        )
        first = (core + k * cores) * axis.per
        n = min(max(out_size - first, 0), axis.per)
        chunk[:_RESAMPLE_HEADER] = p, axis.window, first, n
        for s in range(n):
            xmin, weights, total = axis.taps(first + s)
            at = _RESAMPLE_HEADER + s * axis.slot
            chunk[at : at + 2] = xmin, len(weights)
            q = np.zeros(2 * (axis.slot - 2), np.int64)
            for j, v in enumerate(weights):
                v = (v / total if total != 0.0 else v) * (1 << p)
                q[j] = int(-0.5 + v) if v < 0 else int(0.5 + v)
            chunk[at + 2 : at + axis.slot] = q.astype(np.int16).view(np.int32)
    return table


# A patch's pixels on each axis, and the outputs of a resize table chunk.
_RESIZE_SIDE = 16

# The bf16 of u8 / 255, computed in float32 and rounded to bf16 once.
_RESIZE_PIXELS = (np.arange(256, dtype=np.float32) * np.float32(1 / 255)).astype(
    bfloat16
)


def _resize_slots(chunk, win):
    """A 16-slot table chunk's starts, counts and int16 weights (each slot's from its third word on)."""
    slot = 2 + (win + 1) // 2
    words = np.asarray(chunk, np.int32)
    at = _RESAMPLE_HEADER + slot * np.arange(_RESIZE_SIDE)
    return words[at], words[at + 1], [words[a + 2 :].view(np.int16) for a in at]


def resize_ref(
    image: np.ndarray,
    taps_w: np.ndarray,
    taps_h: np.ndarray,
    rows: int,
    columns: int,
    out_rows: int,
    out_columns: int,
    *,
    words: int = 356,
    chunk: int = 4096,
    patch_columns: int = 8,
    cores: int = 16,
) -> np.ndarray:
    """Numpy reference for [`resize`][iron.kernels.vision.resize].

    Args:
        image: The image chunks, ``(n, chunk)`` uint8: row ``r``'s ``3 *
            columns`` bytes start chunk ``r * ceil(3 * columns / chunk)``.
        taps_w: ``(n, words)`` int32, the width table, chunk ``k`` patch
            column ``k``.
        taps_h: ``(n, words)`` int32, the height table, chunk ``b`` band
            ``b``.
        rows: The image's rows.
        columns: The image's columns.
        out_rows: The resized rows.
        out_columns: The resized columns.
        words: The int32 words of a table chunk.
        chunk: The bytes of an image chunk.
        patch_columns: The most patch columns one core holds.
        cores: The cores the patch columns are dealt to.

    Returns:
        ``(bands * pad, 768)`` bf16: band ``b``'s patch column ``j`` is row
        ``b * pad + j``, ``pad = (out_columns // 16 // cores + 1) * cores``,
        and patch columns past the image are zeros. Sizes the kernel
        refuses zero every patch, a width chunk a core refuses zeros that
        core's patch columns, and a height chunk refused, or a band whose
        rows read past the image's, zeros every patch from that band on.
    """
    side = _RESIZE_SIDE
    win_cap = 2 * ((words - _RESAMPLE_HEADER) // side - 2) - 1
    strips = out_columns // side if out_columns > 0 else 0
    bands = out_rows // side if out_rows > 0 else 0
    pad = (strips // cores + 1) * cores
    out = np.zeros((bands, pad, side, side, 3), np.uint8)
    sized = (
        rows >= 1
        and columns >= 1
        and out_rows >= side
        and out_columns >= side
        and out_rows % side == 0
        and out_columns % side == 0
        and pad // cores <= patch_columns
    )
    if not sized:
        return _RESIZE_PIXELS[out.reshape(bands * pad, -1)]

    cpr = -(-3 * columns // chunk)
    lines = np.asarray(image, np.uint8).reshape(-1)[: rows * cpr * chunk]
    pixels = lines.reshape(rows, cpr * chunk)[:, : 3 * columns]
    pixels = pixels.reshape(rows, columns, 3).astype(np.int64)

    # Across: each core's patch columns at the precision and window of the
    # last width chunk it takes.
    ok = [True] * cores
    mid = np.zeros((rows, strips, side, 3), np.int64)
    for c in range(cores):
        filters = []
        for k in range(c, strips, cores):
            t = np.asarray(taps_w[k], np.int32)
            p, win = int(t[0]), int(t[1])
            if not (1 <= p <= 22 and 1 <= win <= win_cap) or (
                t[2] != k * side or t[3] != side
            ):
                ok[c] = False
                break
            start, count, weights = _resize_slots(t, win)
            first, end = int(start[0]), int(start[-1] + count[-1])
            if (
                (start < first).any()
                or (count < 0).any()
                or (count > win).any()
                or (start + count > end).any()
                or first < 0
                or end > columns
                or end - first > 5 * win_cap + 8
            ):
                ok[c] = False
                break
            hweight = np.zeros((side, 64), np.int64)
            for o in range(side):
                hweight[o, : count[o]] = weights[o][: count[o]]
            filters.append((k, start, hweight, p, win))
        if not ok[c] or not filters:
            continue
        _, _, _, hp, win = filters[-1]
        taps = 64 if win > 32 else 32
        for k, start, hweight, _, _ in filters:
            index = np.minimum(start[:, None] + np.arange(taps), columns - 1)
            acc = np.einsum("rotc,ot->roc", pixels[:, index], hweight[:, :taps])
            mid[:, k] = np.clip((acc + (1 << (hp - 1))) >> hp, 0, 255)

    # Down: a band's output rows as their input rows arrive, every core alike.
    arrived = 0
    for b in range(bands):
        vt = np.asarray(taps_h[b], np.int32)
        p, win = int(vt[0]), int(vt[1])
        if not (1 <= p <= 22 and 1 <= win <= win_cap) or (
            vt[2] != b * side or vt[3] != side
        ):
            break
        start, count, weights = _resize_slots(vt, win)
        e = min(max(int(start[-1] + count[-1]), 0), rows)
        hold = np.zeros((side, strips, side, 3), np.int64)
        done, failed = 0, False
        for now in range(arrived, max(arrived, e) + 1):
            for o in range(done, side):
                s, n = int(start[o]), int(count[o])
                if s + n > now:
                    break
                if s < 0 or s < now - win_cap or n < 0 or n > win_cap:
                    failed = True
                    break
                w = weights[o][:n].astype(np.int64)
                acc = np.einsum("k,kjxc->jxc", w, mid[s : s + n])
                hold[o] = np.clip((acc + (1 << (p - 1))) >> p, 0, 255)
                done = o + 1
            if failed:
                break
        arrived = max(arrived, e)
        if failed or done < side:
            break
        for c in range(cores):
            for k in range(c, strips, cores) if ok[c] else ():
                out[b, k] = hold[:, k]
    return _RESIZE_PIXELS[out.reshape(bands * pad, -1)]
