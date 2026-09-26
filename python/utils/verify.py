# verify.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Tolerance-based output verification helpers for examples and tests.

Mirrors the canonical ``test_utils::nearly_equal`` semantics used across the
C++ testbenches so Python migrations of those examples behave identically:

    |a - b|  <  max(atol, rtol * (|a| + |b|))

Defaults match the C++ default of ``rtol=0.128``, which is the widely-used
relative tolerance for bfloat16 / LUT-approximated kernels (exp, softmax,
gelu, silu, swiglu, ...).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace

import numpy as np
from aie.utils.benchmark import print_benchmark
from ml_dtypes import bfloat16

_DEFAULT_RTOL = 0.128


def nearly_equal(
    a, b, *, rtol: float = _DEFAULT_RTOL, atol: float | None = None
) -> np.ndarray:
    """Element-wise nearly-equal comparison.

    Returns a boolean ndarray of the broadcast shape; ``True`` where
    ``|a - b| < max(atol, rtol * (|a| + |b|))``.  Inputs are coerced to
    ``float32`` (sufficient headroom for bfloat16 work).  NaN inputs
    produce ``False`` (matching IEEE and the C++ semantics).

    Args:
        a: First array-like to compare.
        b: Second array-like to compare.
        rtol: Relative tolerance (default 0.128 — matches C++ test_utils).
        atol: Absolute floor.  Defaults to ``np.finfo(np.float32).tiny``.
    """
    a32 = np.asarray(a, dtype=np.float32)
    b32 = np.asarray(b, dtype=np.float32)
    if atol is None:
        atol = float(np.finfo(np.float32).tiny)
    with np.errstate(over="ignore", invalid="ignore"):
        diff = np.abs(a32 - b32)
        norm = np.minimum(np.abs(a32) + np.abs(b32), np.finfo(np.float32).max)
        thresh = np.maximum(atol, rtol * norm)
    return (a32 == b32) | (diff < thresh)


def count_mismatches(
    actual,
    ref,
    *,
    rtol: float = _DEFAULT_RTOL,
    atol: float | None = None,
    stop_at_nonfinite: bool = True,
) -> tuple[int, int]:
    r"""Count tolerance violations between ``actual`` and ``ref``.

    Returns ``(errors, n_checked)`` where ``n_checked`` is the number of
    samples that were actually compared (less than ``len(ref)`` when
    ``stop_at_nonfinite`` halts on the first inf/nan from either side).

    With ``stop_at_nonfinite=True`` (default), this matches the canonical
    C++ verify pattern that ``break``\\s on the first inf/nan rather than
    treating the LUT's behaviour outside its defined input range as part
    of the contract.
    """
    a32 = np.asarray(actual, dtype=np.float32).ravel()
    r32 = np.asarray(ref, dtype=np.float32).ravel()
    if a32.shape != r32.shape:
        raise ValueError(
            f"actual and ref must have the same shape, got {a32.shape} vs {r32.shape}"
        )
    if stop_at_nonfinite:
        bad = ~(np.isfinite(a32) & np.isfinite(r32))
        stop = int(np.argmax(bad)) if bad.any() else len(a32)
    else:
        stop = len(a32)
    ok = nearly_equal(a32[:stop], r32[:stop], rtol=rtol, atol=atol)
    return int(np.size(ok) - np.count_nonzero(ok)), stop


def assert_pass(
    actual,
    expected,
    *,
    rtol: float | None = None,
    atol: float | None = None,
    fail_msg: str | None = None,
    print_pass: bool = True,
) -> None:
    """Verify ``actual`` matches ``expected``; print ``PASS!`` on success.

    Args:
        actual: Array-like produced by the kernel under test.
        expected: Reference array-like (numpy arrays, scalars, lists).
        rtol: Relative tolerance for the bf16/LUT-style comparator
            (see `count_mismatches`).  When both ``rtol`` and ``atol``
            are ``None`` (the default), use ``np.array_equal`` for an exact
            compare — the right choice for integer and bit-exact pipelines.
            Pass ``rtol=`` (and/or ``atol=``) to opt into the
            tolerance comparator.
        atol: Absolute tolerance floor for the tolerance comparator.
            See ``rtol`` for the default-exact-compare behaviour.
        fail_msg: Optional context appended to the ``FAIL!`` line that
            ``sys.exit()`` raises on mismatch.
        print_pass: When ``True`` (default), print ``PASS!`` on success.
            Set to ``False`` to do the verify check but defer the
            success banner — useful when you want to print benchmark
            stats first and then the ``PASS!`` line.

    Raises:
        SystemExit: On mismatch (via ``sys.exit``) — exits with the
            ``"FAIL!"`` message as the status string.
    """
    if rtol is None and atol is None:
        ok = bool(np.array_equal(actual, expected))
    else:
        errors, _ = count_mismatches(
            actual,
            expected,
            rtol=rtol if rtol is not None else _DEFAULT_RTOL,
            atol=atol,
        )
        ok = errors == 0
    if not ok:
        sys.exit("FAIL!" if fail_msg is None else f"FAIL! {fail_msg}")
    if print_pass:
        print("PASS!")


def assert_close_with_benchmark(
    actual,
    expected,
    *,
    bench,
    ops: float | None = None,
    gflops_fmt: str = ".2f",
    float_rtol: float = 0.05,
    float_atol: float = 0.5,
    tolerance: "Tolerance | None" = None,
    fail_msg: str | None = None,
    mismatch_indices: bool = False,
) -> None:
    """Verify, print benchmark stats, optionally print GFLOPS, then ``PASS!``.

    Wraps the standard matmul/vector_scalar_mul tail in one call.  Picks
    the comparator based on ``expected``'s dtype: integer dtypes use the
    exact compare (``np.array_equal``), float dtypes use the tolerance
    compare with ``rtol=float_rtol`` / ``atol=float_atol``. A kernel's own
    ``Tolerance`` (``fn.contract.tolerance``) can be passed instead and
    is judged by ``compare``, whatever the dtype.

    Args:
        actual: Array-like produced by the kernel under test.
        expected: Reference array-like; ``expected.dtype`` selects the
            comparator branch.
        bench: A `aie.utils.benchmark.BenchmarkResult` (typically
            from `aie.utils.benchmark.run_iters`).
        ops: Total scalar ops for the kernel (e.g. ``2 * M * K * N`` for
            matmul, ``2 * M * K`` for matvec).  When set and
            ``bench.npu`` is available, prints ``NPU GFLOPS`` using
            ``ops / (1000 * avg_us)``.
        gflops_fmt: Format spec for the GFLOPS number (default ``".2f"``;
            matrix_vector uses ``".4f"`` for finer resolution at low
            GFLOPS).
        float_rtol: Relative tolerance for the float branch.
            Defaults match the C++ matmul harness's get_*_tol.
        float_atol: Absolute tolerance for the float branch.
            Defaults match the C++ matmul harness's get_*_tol.
        fail_msg: Optional context appended to the ``FAIL!`` line on
            mismatch.
        tolerance: The kernel's declared ``Tolerance``; when given it
            replaces the dtype-selected comparator above.
        mismatch_indices: When True (and the integer branch detects a
            mismatch), append the first five mismatch ``np.argwhere``
            indices to the ``FAIL!`` line — useful for matmul-style
            debugging.  No-op for the float branch.

    Raises:
        SystemExit: On mismatch (via `assert_pass`).
    """
    if tolerance is not None:
        verdict = compare(np.asarray(actual), np.asarray(expected), tolerance)
        if not verdict:
            base = "output mismatch" if fail_msg is None else fail_msg
            sys.exit(f"FAIL! {base}: {verdict.detail}")
    elif np.issubdtype(np.asarray(expected).dtype, np.integer):
        if mismatch_indices and not bool(np.array_equal(actual, expected)):
            diffs = np.argwhere(np.asarray(actual) != np.asarray(expected))[:5]
            base = "output mismatch" if fail_msg is None else fail_msg
            sys.exit(f"FAIL! {base} (first mismatches: {diffs.tolist()})")
        assert_pass(actual, expected, fail_msg=fail_msg, print_pass=False)
    else:
        assert_pass(
            actual,
            expected,
            rtol=float_rtol,
            atol=float_atol,
            fail_msg=fail_msg,
            print_pass=False,
        )

    print()
    print_benchmark(bench)
    if ops is not None and bench.npu is not None:
        gflops = ops / (1000 * bench.npu.avg_us)
        print(f"NPU GFLOPS                    : {gflops:{gflops_fmt}}")
    print("PASS!")


# ---------------------------------------------------------------------------
# Tolerance contracts and a dtype-aware comparator
# ---------------------------------------------------------------------------
#
# ``nearly_equal`` / ``count_mismatches`` above are the canonical loose
# comparators the examples use. Kernel regression testing needs the contract
# to be an object a kernel can own -- "bit-exact", "within 1 bf16 ULP",
# "rtol 0.128 as the LUT documents" -- so the same tolerance drives the
# correctness suite, the benchmark gate and the e2e tests without each of
# them choosing a number. ``Tolerance`` is that object; ``compare`` applies
# it and reports what went wrong, not just whether.


@dataclass(frozen=True)
class Tolerance:
    """How close a kernel's output must be to its reference.

    Exactly one of three kinds, chosen by which fields are set:

    * **exact** -- no field set: bit-equal after casting the reference to
      the output dtype. Integers, selections (relu, max), lossless copies.
    * **ulps** -- ``ulps`` set: bf16 outputs within ``ulps`` units in the
      last place of the correctly rounded reference. ``atol`` may be set
      alongside as a floor, admitting an element that meets *either* -- what a
      kernel needs when the device flushes subnormals to zero, since a flushed
      value is a full 100% relative and dozens of ulps from the reference but
      absolutely negligible. ``rtol`` stays unset, or the kind is relative.
    * **relative** -- ``rtol``/``atol`` set: the canonical
      ``|a - b| < max(atol, rtol * (|a| + |b|))`` of ``nearly_equal``.
      Integer outputs are compared with the same formula in exact integer
      arithmetic, so ``lsb`` (``atol = n + 0.5``) admits an ``n``-LSB
      slack for fixed-point pixel kernels whose rounding shift is not
      modeled; under **exact** and **ulps** integers stay bit-equal.

    Non-finite values are never skipped: NaN must meet NaN, and an infinity
    must meet an infinity of the same sign, under every kind.

    ``range_frac`` adds a floor scaled to the reference's own range: an element
    also passes at ``|a - b| <= range_frac * max|b|``. It is for a kernel whose
    error is set by the magnitudes it worked from rather than by the magnitude
    it produced -- a dot product whose terms cancel to near zero is no less
    accurate than its neighbours, but an elementwise relative bound reads it as
    100% wrong. The scale comes from ``expected``, never from ``actual``, so a
    kernel cannot widen its own tolerance by returning something large. Unlike
    ``atol`` it follows the data: the same fraction holds whether the outputs
    run to 34 or to 3.4e9, where a fixed floor would be either dead or
    permissive. Set it from a measured worst case, and say so in ``note``.

    ``max_mismatch_frac`` allows that fraction of elements to miss (LUT tails,
    saturation edges). ``note`` records where the number came from -- a
    docstring, a device run, a testbench default -- so a reviewer can tell an
    evidenced tolerance from a guessed one.
    """

    rtol: float | None = None
    atol: float | None = None
    ulps: int | None = None
    range_frac: float | None = None
    max_mismatch_frac: float = 0.0
    note: str = ""

    def __post_init__(self):
        if self.range_frac is None:
            return
        if self.range_frac <= 0:
            raise ValueError(f"range_frac must be positive, got {self.range_frac}")
        if self.kind == "exact":
            raise ValueError(
                "range_frac needs a tolerance to be a floor under; set rtol, atol "
                "or ulps as well, or drop it -- an exact comparison admits nothing"
            )

    @property
    def kind(self) -> str:
        if self.ulps is not None:
            return "ulps"
        if self.rtol is not None or self.atol is not None:
            return "relative"
        return "exact"

    @classmethod
    def exact(cls, *, note: str = "") -> "Tolerance":
        return cls(note=note)

    @classmethod
    def bf16_ulps(
        cls,
        n: int = 1,
        *,
        atol: float | None = None,
        max_mismatch_frac: float = 0.0,
        note: str = "",
    ) -> "Tolerance":
        """bf16 outputs within ``n`` ulps of the correctly rounded reference.

        ``atol`` is an optional strict bound (``error < atol``) an element may
        meet instead of the ulp bound. Use it for the device's subnormal flush
        to zero, set to the smallest normal bf16 so it admits the flushed
        values and nothing above them.
        """
        return cls(ulps=n, atol=atol, max_mismatch_frac=max_mismatch_frac, note=note)

    @classmethod
    def relative(
        cls,
        rtol: float = _DEFAULT_RTOL,
        atol: float | None = None,
        *,
        range_frac: float | None = None,
        max_mismatch_frac: float = 0.0,
        note: str = "",
    ) -> "Tolerance":
        return cls(
            rtol=rtol,
            atol=atol,
            range_frac=range_frac,
            max_mismatch_frac=max_mismatch_frac,
            note=note,
        )

    @classmethod
    def lsb(
        cls, n: int = 1, *, max_mismatch_frac: float = 0.0, note: str = ""
    ) -> "Tolerance":
        """Integer outputs within ``n`` least-significant bits of the reference.

        For fixed-point kernels whose final saturating shift may round or
        truncate (the AIE ``srs`` rounding mode is a core setting the kernel
        does not fix). ``rtol`` is zero: the slack is absolute.
        """
        return cls(
            rtol=0.0, atol=n + 0.5, max_mismatch_frac=max_mismatch_frac, note=note
        )

    @classmethod
    def default_for(cls, dtype) -> "Tolerance":
        """Return the contract a kernel gets when it declares none.

        Integer and boolean outputs are bit-exact. bfloat16 outputs get the
        repository's canonical ``rtol=0.128``, the C++ testbench default that
        the LUT-approximated kernels document; float32 outputs are held to
        ``rtol=1e-4`` (a few float32 ULPs of accumulation-order slack, far
        inside what a bf16 tolerance would hide) and float16 to ``1e-2``.
        """
        dt = np.dtype(dtype)
        if np.issubdtype(dt, np.integer) or dt == np.bool_:
            return cls.exact(note="default: integer output")
        if dt == np.dtype(np.float32) or dt == np.dtype(np.float64):
            return cls.relative(1e-4, note="default: float32 output")
        if dt == np.dtype(np.float16):
            return cls.relative(1e-2, note="default: float16 output")
        return cls.relative(_DEFAULT_RTOL, note="default: canonical bf16/LUT rtol")


@dataclass
class Verdict:
    """Outcome of ``compare``. Truthy when the comparison passed."""

    ok: bool
    n_checked: int
    n_mismatch: int
    max_abs_err: float
    max_ulp_err: int | None
    first_bad_index: int | None
    detail: str

    def __bool__(self) -> bool:
        return self.ok


def bf16_ulp_distance(a, b) -> np.ndarray:
    """Element-wise distance between two bf16 arrays in units in the last place.

    Bit patterns are mapped to a monotonic integer scale (sign-magnitude to
    two's-complement style) so the distance is a plain subtraction. -0 and +0
    map to the same point, so a kernel that produces the other zero is not
    penalized.
    """

    def ordinal(x):
        bits = np.asarray(x).astype(bfloat16).view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - bits, bits)

    return np.abs(ordinal(a) - ordinal(b))


def poisoned(n: int, dtype) -> np.ndarray:
    """Return an ``n``-element buffer filled with a value no kernel would write.

    An output buffer left as zeros lets a kernel that never writes pass a
    comparison against a reference that happens to be zeros. Filling it with
    0x55 bytes first means silence fails.
    """
    return np.full(n * np.dtype(dtype).itemsize, 0x55, dtype=np.uint8).view(dtype)


def compare(
    actual, expected, tol: Tolerance | None = None, *, range_axis: int | None = None
) -> Verdict:
    """Compare a kernel's ``actual`` output with a reference under ``tol``.

    ``expected`` may be higher precision than ``actual`` (a float64 sum, an
    int64 product); it is cast to ``actual.dtype``, so the kernel is held to
    what a correctly rounded implementation would produce. With ``tol=None``
    the output dtype's ``Tolerance.default_for`` applies.

    ``range_axis`` selects the axis reduced to compute ``range_frac``'s
    reference scale. For ``(calls, tile)`` arrays, use 1 to scale each call
    independently. The default uses the whole reference array.

    This measures; it does not model. What a kernel does on overflow, on a
    narrowing store, or with subnormal inputs belongs in the reference that
    produced ``expected`` -- a saturating kernel's reference clips, a
    denormal-flushing kernel's reference flushes. A reference that leaves the
    output range is reported as such when the comparison fails, since it
    means the reference is under-specified rather than the kernel wrong.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if tol is None:
        tol = Tolerance.default_for(actual.dtype)
    if actual.shape != expected.shape:
        return Verdict(
            False,
            0,
            0,
            float("inf"),
            None,
            None,
            f"shape mismatch {actual.shape} vs {expected.shape}",
        )
    a, e, n = actual.ravel(), expected.ravel(), actual.size

    def ref_scale(ref) -> float | np.ndarray:
        """Return max|ref| over finite entries.

        This is what ``range_frac`` is a fraction of, or zero when no range
        floor is in play.
        """
        if tol.range_frac is None or not n:
            return 0.0
        mag = np.abs(np.asarray(ref, dtype=np.float64).reshape(expected.shape))
        mag = np.where(np.isfinite(mag), mag, 0.0)
        scale = mag.max(axis=range_axis, keepdims=True, initial=0.0)
        if range_axis is None:
            return float(scale.item())
        return np.broadcast_to(scale, expected.shape).ravel()

    # Integers: bit-exact under exact / ulps; under a relative tolerance the
    # nearly_equal formula in int64 (Tolerance.lsb sets atol = n + 0.5).
    if np.issubdtype(actual.dtype, np.integer) or actual.dtype == np.bool_:
        n_over = 0
        if actual.dtype != np.bool_ and np.issubdtype(e.dtype, np.integer):
            info = np.iinfo(actual.dtype)
            e_wide = e.astype(np.int64)
            n_over = int(np.count_nonzero((e_wide < info.min) | (e_wide > info.max)))
        e_cast = e.astype(actual.dtype)
        a64, e64 = a.astype(np.int64), e_cast.astype(np.int64)
        err = np.abs(a64 - e64)
        scale = ref_scale(e64)
        if tol.kind == "relative":
            bound = np.maximum(
                tol.atol or 0.0,
                (tol.rtol or 0.0) * (np.abs(a64) + np.abs(e64)),
            )
            close = err < bound
            if tol.range_frac is not None:
                close |= err <= tol.range_frac * scale
            bad = ~close
        else:
            bad = a != e_cast
        v = _verdict(bad, err, None, tol, n, ref_range=scale)
        if n_over and not v.ok:
            # Not a policy, a diagnostic: the reference left the output range,
            # so what the device did there says nothing about the kernel.
            v = replace(
                v,
                detail=f"{v.detail}; the reference overflows "
                f"{np.dtype(actual.dtype).name} in {n_over} of {n} elements, so "
                "it does not model what the kernel does there (clip for a "
                "saturating kernel, cast for a wrapping one)",
            )
        return v

    a32, e32 = a.astype(np.float32), e.astype(np.float32)
    a_nan, e_nan = np.isnan(a32), np.isnan(e32)
    a_inf, e_inf = np.isinf(a32), np.isinf(e32)
    nonfinite_bad = (a_nan != e_nan) | (a_inf != e_inf) | (a_inf & e_inf & (a32 != e32))
    finite = ~(a_nan | e_nan | a_inf | e_inf)
    err = np.zeros(n, np.float64)

    if tol.kind == "ulps":
        if actual.dtype != bfloat16:
            raise ValueError(
                f"Tolerance in ULPs is defined for bfloat16 outputs, got {actual.dtype}"
            )
        e_bf = e32.astype(bfloat16)
        ulp = np.zeros(n, np.int64)
        ulp[finite] = bf16_ulp_distance(a[finite], e_bf[finite])
        err[finite] = np.abs(a32[finite] - e_bf[finite].astype(np.float32))
        max_ulps = tol.ulps if tol.ulps is not None else 0
        within = ulp <= max_ulps
        scale = ref_scale(e_bf.astype(np.float32))
        if tol.atol is not None:
            within |= err < tol.atol
        if tol.range_frac is not None:
            within |= err <= tol.range_frac * scale
        bad = nonfinite_bad | (finite & ~within)
        return _verdict(bad, err, ulp, tol, n, nonfinite_bad, ref_range=scale)

    if tol.kind == "exact":
        e_cast = e32.astype(actual.dtype).astype(np.float32)
        err[finite] = np.abs(a32[finite] - e_cast[finite])
        bad = nonfinite_bad | (finite & (a32 != e_cast))
        return _verdict(bad, err, None, tol, n, nonfinite_bad)

    err[finite] = np.abs(a32[finite].astype(np.float64) - e32[finite])
    scale = ref_scale(e32)
    close = nearly_equal(a32, e32, rtol=tol.rtol or 0.0, atol=tol.atol)
    if tol.range_frac is not None:
        close |= err <= tol.range_frac * scale
    bad = nonfinite_bad | (finite & ~close)
    return _verdict(bad, err, None, tol, n, nonfinite_bad, ref_range=scale)


def _verdict(
    bad,
    err,
    ulp,
    tol: Tolerance,
    n: int,
    nonfinite_bad=None,
    ref_range: float | np.ndarray = 0.0,
) -> Verdict:
    n_bad = int(np.count_nonzero(bad))
    # A non-finite mismatch must fail regardless of max_mismatch_frac: that
    # budget is for how close a finite value came, not for whether NaN/Inf
    # values were reproduced at all.
    n_nonfinite_bad = (
        int(np.count_nonzero(nonfinite_bad)) if nonfinite_bad is not None else 0
    )
    ok = n_nonfinite_bad == 0 and n_bad <= int(np.floor(tol.max_mismatch_frac * n))
    first = int(np.argmax(bad)) if n_bad else None
    max_ulp = int(ulp.max()) if ulp is not None and n else None
    max_err = float(err.max()) if n else 0.0
    if ok:
        detail = "ok"
    else:
        detail = f"{n_bad}/{n} mismatches; first at flat index {first}; max_abs_err={max_err:.4g}"
        if max_ulp is not None:
            detail += f"; max_ulp={max_ulp}"
        if np.any(np.asarray(ref_range) > 0):
            # The quantity range_frac is stated in, so a failure says directly
            # whether the bound wants raising or the kernel is wrong.
            fractions = np.divide(
                err,
                ref_range,
                out=np.where(err == 0, 0.0, np.inf),
                where=np.asarray(ref_range) > 0,
            )
            detail += (
                f"; max_abs_err/max|expected|={float(fractions.max()):.4g}"
                f" vs range_frac={tol.range_frac:.4g}"
            )
        if tol.note:
            detail += f" [{tol.kind}: {tol.note}]"
    return Verdict(ok, n, n_bad, max_err, max_ulp, first, detail)
