# kernels/vision.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vision kernel factories: color conversion, threshold, filter2d, add_weighted."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.verify import Tolerance

from ._common import (
    KernelContract,
    _declare_dtypes,
    _default_source_path,
    _dtype_to_bit_width,
    _make_extern,
)

# Fixed-point pixel kernels end in a saturating accumulator shift whose
# rounding mode is a core setting the kernels do not fix (they call set_sat but
# not set_rounding), so the references below use floor and allow one LSB.
_LSB = Tolerance.lsb(
    1, note="fixed-point srs shift: rounding mode not modelled, floor +/- 1"
)


def _color_convert_kernel(
    func_name: str,
    filename: str,
    in_size: int,
    out_size: int,
    use_chess: bool = False,
    contract: KernelContract | None = None,
) -> ExternalFunction:
    """Shared implementation for color-space conversion line kernels."""
    in_ty = np.ndarray[(in_size,), np.dtype[np.uint8]]
    out_ty = np.ndarray[(out_size,), np.dtype[np.uint8]]
    return _make_extern(
        func_name,
        _default_source_path(filename),
        [in_ty, out_ty, np.int32],
        use_chess=use_chess,
        contract=contract,
    )


def _bitwise_kernel(
    op: str, line_width: int, dtype, use_chess: bool = False
) -> ExternalFunction:
    """Shared implementation for [`bitwise_or`][iron.kernels.vision.bitwise_or] and [`bitwise_and`][iron.kernels.vision.bitwise_and]."""
    bit_width = _dtype_to_bit_width(dtype, factory_name=f"bitwise{op}")
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        f"bitwise{op}Line",
        _default_source_path(f"bitwise{op}.cc"),
        [line_ty, line_ty, line_ty, np.int32],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "in", "out", "count"),
            reference={"OR": bitwise_or_ref, "AND": bitwise_and_ref}[op],
            tolerance=Tolerance.exact(note="bitwise op"),
        ),
    )


# The vector path in rgba2hue.cc divides through a 16-bit reciprocal LUT and
# rounds a Q7.9 accumulator; the scalar path (the reference) divides exactly.
# Two LSB and a 2 % budget for the LUT's worst divisors, until a device run
# says otherwise. See rgba2hue_ref for the wrap-around at hue 0.
_HUE_TOLERANCE = Tolerance.lsb(
    2,
    max_mismatch_frac=0.02,
    note="reference is the exact-division scalar path; vector path uses a 16-bit "
    "reciprocal LUT (unmeasured on device)",
)


def rgba2hue(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert a line of RGBA pixels to hue values (full-range, 0..255)."""
    return _color_convert_kernel(
        "rgba2hueLine",
        "rgba2hue.cc",
        line_width * 4,
        line_width,
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "out", "count"),
            reference=rgba2hue_ref,
            acc_dtype=np.int32,
            reduction=1,
            overflow="wrap",  # the uint8 store of a negative hue
            rounding="unspecified",
            tolerance=_HUE_TOLERANCE,
        ),
    )


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
    scalar_ty = np.int32 if bit_width == 32 else np.int16
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "thresholdLine",
        _default_source_path("threshold.cc"),
        [line_ty, line_ty, np.int32, scalar_ty, scalar_ty, np.int8],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "out", "count", "scalar", "scalar", "scalar"),
            reference=threshold_ref,
            tolerance=Tolerance.exact(note="selection"),
        ),
    )


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(
    threshold, ({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32})
)


def bitwise_or(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise OR of two lines."""
    return _bitwise_kernel("OR", line_width, dtype, use_chess=use_chess)


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(
    bitwise_or, ({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32})
)


def bitwise_and(
    line_width: int = 1920, dtype: type = np.uint8, use_chess: bool = False
) -> ExternalFunction:
    """Element-wise bitwise AND of two lines."""
    return _bitwise_kernel("AND", line_width, dtype, use_chess=use_chess)


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(
    bitwise_and, ({"dtype": np.uint8}, {"dtype": np.int16}, {"dtype": np.int32})
)


def gray2rgba(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert a grayscale line to RGBA."""
    return _color_convert_kernel(
        "gray2rgbaLine",
        "gray2rgba.cc",
        line_width,
        line_width * 4,
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "out", "count"),
            reference=gray2rgba_ref,
            tolerance=Tolerance.exact(note="copy with alpha = 255"),
        ),
    )


def rgba2gray(line_width: int = 1920, use_chess: bool = False) -> ExternalFunction:
    """Convert an RGBA line to grayscale."""
    return _color_convert_kernel(
        "rgba2grayLine",
        "rgba2gray.cc",
        line_width * 4,
        line_width,
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "out", "count"),
            reference=rgba2gray_ref,
            tolerance=_LSB,
            acc_dtype=np.int32,
            reduction=3,
            overflow="undefined",  # cannot overflow: weights sum to 1
            rounding="unspecified",
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
        _default_source_path("filter2d.cc"),
        [line_ty, line_ty, line_ty, line_ty, np.int32, kernel_ty],
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "in", "in", "out", "count", "param"),
            reference=filter2d_ref,
            acc_dtype=np.int32,
            reduction=9,
            overflow="saturate",  # set_sat before the shift
            rounding="unspecified",
            tolerance=_LSB,
            ops_per_call=18 * line_width,
        ),
    )


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
    gamma_ty = {8: np.int8, 16: np.int16, 32: np.int32}[bit_width]
    line_ty = np.ndarray[(line_width,), np.dtype[dtype]]
    return _make_extern(
        "addWeightedLine",
        _default_source_path("addWeighted.cc"),
        [line_ty, line_ty, line_ty, np.int32, np.int16, np.int16, gamma_ty],
        compile_flags=[f"-DBIT_WIDTH={bit_width}"],
        use_chess=use_chess,
        contract=KernelContract(
            roles=("in", "in", "out", "count", "scalar", "scalar", "scalar"),
            reference=add_weighted_ref,
            acc_dtype=np.int32,
            reduction=2,
            overflow="saturate",  # set_saturation(saturate)
            rounding="unspecified",
            tolerance=_LSB,
            ops_per_call=3 * line_width,
        ),
    )


# Supported dtype combinations, as data: the registry and the contract test
# enumerate these instead of restating them.
_declare_dtypes(add_weighted, ({"dtype": np.uint8}, {"dtype": np.int16}))


# --------------------------------------------------------------------------
# Numpy references. Each follows the *vector* path of its kernel (the one the
# ``*Line`` entry points call), read off aie_kernels/aie2/*.cc.
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


def _c_div(num, den):
    """C integer division (truncates toward zero) for int64 arrays; den > 0."""
    return np.sign(num) * (np.abs(num) // den)


def rgba2hue_ref(rgba):
    """Numpy reference for [`rgba2hue`][iron.kernels.vision.rgba2hue]: full-range hue.

    The scalar path of ``rgba2hue.cc``: with ``d = max - min`` of R, G, B,
    ``h = 85 (G - B) / d`` when R is the max, ``170 + 85 (B - R) / d`` when G
    is, ``340 + 85 (R - G) / d`` otherwise (C integer division), then
    ``(h + 1) >> 1`` and a cast to ``uint8`` that wraps: a negative hue
    (R max, G < B) comes out as ``256 + h``, which is the right circular
    value. Grey pixels (``d == 0``) are hue 0. Ties go to R, then G, as the
    kernel's ``max == r`` / ``max == g`` tests do.
    """
    rgba = np.asarray(rgba, dtype=np.uint8)
    px = rgba.reshape(*rgba.shape[:-1], -1, 4).astype(np.int64)
    r, g, b = px[..., 0], px[..., 1], px[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = np.where(mx == mn, 1, mx - mn)  # avoid /0; masked to 0 below
    h_r = _c_div(85 * (g - b), d)
    h_g = 170 + _c_div(85 * (b - r), d)
    h_b = 340 + _c_div(85 * (r - g), d)
    h = np.where(mx == r, h_r, np.where(mx == g, h_g, h_b))
    h = np.where((mx == 0) | (mx == mn), 0, h)
    h = np.right_shift(h + 1, 1)  # arithmetic shift, as in C
    return (h & 0xFF).astype(np.uint8)


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

    ``out = sat((alpha * a + beta * b + gamma) >> 14)`` with ``alpha`` and
    ``beta`` as Q2.14 fixed point (``8192`` is 0.5). This is what the vector
    path in ``addWeighted.cc`` computes: it seeds the accumulator with
    ``gamma`` *before* the shift, so ``gamma`` contributes ``gamma / 2**14``
    and is effectively ignored. The scalar path in the same file adds
    ``gamma`` *after* the shift, as OpenCV does; the two disagree for any
    non-zero ``gamma``. Within one LSB otherwise.
    """
    a = np.asarray(a)
    info = np.iinfo(a.dtype)
    acc = a.astype(np.int64) * int(alpha) + np.asarray(b).astype(np.int64) * int(beta)
    acc = acc + int(gamma)
    return np.clip(acc >> 14, info.min, info.max).astype(a.dtype)


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
