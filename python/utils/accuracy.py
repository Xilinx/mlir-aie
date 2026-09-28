# accuracy.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""How far a kernel's results are from the mathematics, in the output's terms.

A tolerance says whether a kernel passes; this says by how much. Given what a
kernel produced and a high-precision (float64) reference, ``error_stats``
reports the largest absolute and relative error, the error in units in the
last place (ulp) of the output dtype, and how many results are not the
correctly rounded value -- the float64 reference rounded to nearest, ties to
even, which is the most accurate answer the output dtype can hold.

Rounding is done here rather than by a cast: ml_dtypes converts float64 to
bfloat16 through float32, so ``1 + 2**-8 + 2**-30`` becomes 1.0 where the
correctly rounded bfloat16 is 1.0078125. ``round_to`` corrects that.

``all_bf16`` enumerates every bfloat16 bit pattern, so a unary bf16 kernel
can be checked on its whole domain in one 65536-element run.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import ml_dtypes
import numpy as np
import numpy.typing as npt
from ml_dtypes import bfloat16

__all__ = [
    "ErrorStats",
    "all_bf16",
    "error_stats",
    "round_to",
    "ulp_distance",
    "ulp_size",
]


def _word(dtype) -> np.dtype:
    return np.dtype(f"u{np.dtype(dtype).itemsize}")


def _ordinal(x, dtype) -> np.ndarray:
    """Map each value's bits onto a monotonic integer scale; -0 and +0 meet."""
    dt = np.dtype(dtype)
    sign = 1 << (8 * dt.itemsize - 1)
    bits = np.asarray(x).astype(dt).view(_word(dt)).astype(np.int64)
    return np.where(bits & sign, sign - bits, bits)


def _from_ordinal(o, dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    sign = 1 << (8 * dt.itemsize - 1)
    return np.where(o < 0, sign - o, o).astype(_word(dt)).view(dt)


def ulp_distance(a, b, dtype: npt.DTypeLike = bfloat16) -> np.ndarray:
    """Element-wise distance between two arrays in ulps of ``dtype``.

    The values are cast to ``dtype`` and the distance is the number of
    representable values between them, so it is exact across binades and
    subnormals; infinity is one step past the largest finite value. -0 and
    +0 are the same point. NaN has no place on the scale; mask it first.
    """
    return np.abs(_ordinal(a, dtype) - _ordinal(b, dtype))


def _finfo(dtype):
    return ml_dtypes.finfo(np.dtype(dtype))


def ulp_size(x, dtype) -> np.ndarray:
    """Return the spacing of ``dtype`` at each value of ``x``, in float64.

    The gap between the two representable values around ``|x|`` (below the
    smallest normal, the subnormal spacing). Infinity and NaN give NaN.
    """
    fi = _finfo(dtype)
    x = np.abs(np.asarray(x, np.float64))
    _, e = np.frexp(x)
    exponent = np.where(x == 0, fi.minexp, np.maximum(e - 1, fi.minexp))
    size = np.ldexp(1.0, exponent - fi.nmant)
    return np.where(np.isfinite(x), size, np.nan)


def round_to(x, dtype) -> np.ndarray:
    """Round float64 values to ``dtype``: to nearest, ties to even.

    A plain cast is used as a first guess and corrected by comparing it with
    its two neighbours in float64, so it is right even when the cast is off
    by one ulp (ml_dtypes' bfloat16 cast rounds twice, through float32).
    Values past the largest finite one round to infinity where IEEE says so.
    """
    dt = np.dtype(dtype)
    x = np.asarray(x, np.float64)
    if dt == np.float64:
        return x.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        return _round_to(x, dt)


def _round_to(x, dt) -> np.ndarray:
    guess = x.astype(dt)
    fi = _finfo(dt)
    # Where infinity sits on the value line for rounding: one ulp past max.
    beyond = float(fi.max) + float(np.ldexp(1.0, fi.maxexp - 1 - fi.nmant))

    def error(c):
        v = c.astype(np.float64)
        v = np.where(np.isinf(v), np.copysign(beyond, v), v)
        return np.where(np.isnan(v), np.inf, np.abs(v - x))

    # A tie is exact in float32, where the cast already broke it to even; only
    # a strictly nearer neighbour replaces the guess.
    best, best_err = guess, error(guess)
    o = _ordinal(guess, dt)
    for step in (-1, 1):
        c = _from_ordinal(o + step, dt)
        err = error(c)
        better = err < best_err
        best = np.where(better, c, best)
        best_err = np.where(better, err, best_err)
    # The ordinal scale merges the zeros; a zero keeps the reference's sign.
    zero = best.astype(np.float64) == 0
    best = np.where(zero, np.copysign(0.0, x).astype(dt), best)
    return np.where(np.isfinite(x), best, guess).astype(dt)


@dataclass(frozen=True)
class ErrorStats:
    """Error of ``n`` results against a float64 reference, in ``dtype``'s terms.

    Indices are into the flattened arrays. ``max_ulp`` is the integer ulp
    distance from the correctly rounded reference (0 means correctly
    rounded); ``max_ulp_error`` is the libm-style error against the
    unrounded reference, |got - ref| / ulp_size(ref), where a correctly
    rounded result scores at most 0.5. Results where either side is NaN are
    left out of the ulp and error figures and counted in ``nan_mismatch``
    when only one side is NaN.
    """

    dtype: str
    n: int
    not_correctly_rounded: int
    max_ulp: int
    worst_index: int | None
    worst_got: float | None
    worst_ref: float | None
    mean_ulp: float
    ulp_histogram: dict[int, int]
    max_ulp_error: float
    max_abs: float
    max_abs_index: int | None
    max_rel: float
    max_rel_index: int | None
    nan_mismatch: int

    def as_dict(self) -> dict:
        """Return the fields, with the histogram keyed by string for JSON."""
        d = asdict(self)
        d["ulp_histogram"] = {str(k): v for k, v in self.ulp_histogram.items()}
        return d

    def summary(self) -> str:
        """One line: the verdict first, then the worst case and the spread."""
        parts = [
            f"{self.dtype} n={self.n}",
            f"{self.not_correctly_rounded} not correctly rounded",
            f"max {self.max_ulp} ulp ({self.max_ulp_error:.3g} exact)",
        ]
        if self.worst_index is not None and self.max_ulp:
            parts.append(
                f"worst [{self.worst_index}] got {self.worst_got:.9g} "
                f"ref {self.worst_ref:.9g}"
            )
        parts += [
            f"mean {self.mean_ulp:.3g} ulp",
            f"max|d| {self.max_abs:.3g}",
            f"max rel {self.max_rel:.3g}",
        ]
        if self.nan_mismatch:
            parts.append(f"{self.nan_mismatch} NaN mismatch(es)")
        return "; ".join(parts)


def _argmax(a) -> int | None:
    return int(np.argmax(a)) if a.size else None


def error_stats(got, ref, dtype=None) -> ErrorStats:
    """Measure ``got`` against the high-precision reference ``ref``.

    Args:
        got: What the kernel produced, in (or exactly convertible to) the
            output dtype.
        ref: The mathematical answer for the same inputs, as float64 (or any
            float array; it is widened, never rounded).
        dtype: The output dtype; defaults to ``got``'s.
    """
    got = np.asarray(got)
    dt = np.dtype(dtype if dtype is not None else got.dtype)
    got = got.astype(dt).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    if got.shape != ref.shape:
        raise ValueError(f"got {got.size} results for {ref.size} reference values")
    rounded = round_to(ref, dt)
    with np.errstate(invalid="ignore"):
        g64 = got.astype(np.float64)
    g_nan, r_nan = np.isnan(g64), np.isnan(ref)
    ordered = ~(g_nan | r_nan)

    ulp = np.zeros(got.size, np.int64)
    ulp[ordered] = ulp_distance(got[ordered], rounded[ordered], dt)
    wrong = int(np.count_nonzero(ulp)) + int(np.count_nonzero(g_nan != r_nan))

    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        # An infinity where the correctly rounded result is that infinity
        # has no error in the output's terms, whatever the finite reference.
        same = (g64 == ref) | (np.isinf(g64) & (got == rounded))
        diff = np.where(same, 0.0, np.abs(g64 - ref))
        rel = np.where(same, 0.0, diff / np.abs(ref))
        exact = np.where(same, 0.0, diff / ulp_size(ref, dt))
    diff, rel = np.where(ordered, diff, 0.0), np.where(ordered, rel, 0.0)
    exact = np.where(ordered & np.isfinite(ref), exact, 0.0)

    counts = np.unique(ulp[ordered], return_counts=True)
    worst = None
    if ordered.any():
        # Most ulps first, then the larger exact error among ties.
        worst = int(np.lexsort((exact, ulp))[-1])
    return ErrorStats(
        dtype=dt.name,
        n=int(got.size),
        not_correctly_rounded=wrong,
        max_ulp=int(ulp.max()) if ulp.size else 0,
        worst_index=worst,
        worst_got=None if worst is None else float(g64[worst]),
        worst_ref=None if worst is None else float(ref[worst]),
        mean_ulp=float(ulp[ordered].mean()) if ordered.any() else 0.0,
        ulp_histogram={int(k): int(v) for k, v in zip(*counts)},
        max_ulp_error=float(exact.max()) if exact.size else 0.0,
        max_abs=float(diff.max()) if diff.size else 0.0,
        max_abs_index=_argmax(diff),
        max_rel=float(rel.max()) if rel.size else 0.0,
        max_rel_index=_argmax(rel),
        nan_mismatch=int(np.count_nonzero(g_nan != r_nan)),
    )


def all_bf16(*, nan: bool = True, inf: bool = True, subnormal: bool = True):
    """Every bfloat16 value, in bit-pattern order (65536 with all options).

    ``nan``, ``inf`` and ``subnormal`` keep those classes (subnormals include
    neither zero). The full set is 64 tiles of 1024, so ``.reshape(64, 1024)``
    feeds a 1024-element unary kernel on its whole domain in one run.
    """
    x = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(bfloat16)
    with np.errstate(invalid="ignore"):
        x64 = x.astype(np.float64)
    keep = np.ones(x.size, bool)
    if not nan:
        keep &= ~np.isnan(x64)
    if not inf:
        keep &= ~np.isinf(x64)
    if not subnormal:
        tiny = float(_finfo(bfloat16).smallest_normal)
        keep &= ~((x64 != 0) & (np.abs(x64) < tiny))
    return x[keep]
