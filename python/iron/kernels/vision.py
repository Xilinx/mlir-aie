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


def resample_peak(
    words: int = 256, *, cores: int = 16, slots: int | None = None
) -> ExternalFunction:
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
    )
    fn.resample_join = fn.object_file.bind("resample_join", [peak_ty] * 3)
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
    in_size, out_size, core, chunks, *, words=256, cores=16, slots=None
):
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
    peak, in_size, out_size, core, chunks, *, words=256, cores=16, slots=None
):
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
