# kernels/vision.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Vision kernel factories: color conversion, threshold, filter2d, add_weighted."""

import numpy as np
from aie.iron.kernel import ExternalFunction
from aie.utils.compile.jit.markers import In, Out
from aie.utils.verify import Tolerance

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
    if not use_chess and _tuned_arch() == "aie2p":
        # LICM hoists the three accumulator constants out of the loop, where
        # they spill. Capping its MemorySSA walk at zero keeps them in the
        # loop.
        flags += ["-mllvm", "--licm-mssa-optimization-cap=0"]
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
