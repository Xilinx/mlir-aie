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
from dataclasses import dataclass

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
            (see :func:`count_mismatches`).  When both ``rtol`` and ``atol``
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
    :class:`Tolerance` (``fn.contract.tolerance``) can be passed instead and
    is judged by :func:`compare`, whatever the dtype.

    Args:
        actual: Array-like produced by the kernel under test.
        expected: Reference array-like; ``expected.dtype`` selects the
            comparator branch.
        bench: A :class:`~aie.utils.benchmark.BenchmarkResult` (typically
            from :func:`~aie.utils.benchmark.run_iters`).
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
        tolerance: The kernel's declared :class:`Tolerance`; when given it
            replaces the dtype-selected comparator above.
        mismatch_indices: When True (and the integer branch detects a
            mismatch), append the first five mismatch ``np.argwhere``
            indices to the ``FAIL!`` line — useful for matmul-style
            debugging.  No-op for the float branch.

    Raises:
        SystemExit: On mismatch (via :func:`assert_pass`).
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
      last place of the correctly rounded reference.
    * **relative** -- ``rtol``/``atol`` set: the canonical
      ``|a - b| < max(atol, rtol * (|a| + |b|))`` of :func:`nearly_equal`.
      Integer outputs are compared with the same formula in exact integer
      arithmetic, so :meth:`lsb` (``atol = n + 0.5``) admits an ``n``-LSB
      slack for fixed-point pixel kernels whose rounding shift is not
      modelled; under **exact** and **ulps** integers stay bit-equal.

    Non-finite values are never skipped: NaN must meet NaN, and an infinity
    must meet an infinity of the same sign, under every kind.

    ``max_mismatch_frac`` allows that fraction of elements to miss (LUT tails,
    saturation edges). ``note`` records where the number came from -- a
    docstring, a device run, a testbench default -- so a reviewer can tell an
    evidenced tolerance from a guessed one.
    """

    rtol: float | None = None
    atol: float | None = None
    ulps: int | None = None
    max_mismatch_frac: float = 0.0
    note: str = ""

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
        cls, n: int = 1, *, max_mismatch_frac: float = 0.0, note: str = ""
    ) -> "Tolerance":
        return cls(ulps=n, max_mismatch_frac=max_mismatch_frac, note=note)

    @classmethod
    def relative(
        cls,
        rtol: float = _DEFAULT_RTOL,
        atol: float | None = None,
        *,
        max_mismatch_frac: float = 0.0,
        note: str = "",
    ) -> "Tolerance":
        return cls(rtol=rtol, atol=atol, max_mismatch_frac=max_mismatch_frac, note=note)

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
    """Outcome of :func:`compare`. Truthy when the comparison passed."""

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
    penalised.
    """

    def ordinal(x):
        bits = np.asarray(x).astype(bfloat16).view(np.uint16).astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - bits, bits)

    return np.abs(ordinal(a) - ordinal(b))


def compare(
    actual,
    expected,
    tol: Tolerance | None = None,
    *,
    overflow: str = "wrap",
    subnormals: str = "preserve",
) -> Verdict:
    """Compare a kernel's ``actual`` output with a reference under ``tol``.

    ``expected`` may be higher precision than ``actual`` (a float64 sum, an
    int64 product); it is cast to ``actual.dtype`` for the exact and ULP
    kinds, so the kernel is held to what a correctly rounded implementation
    would produce. With ``tol=None`` the output dtype's
    :meth:`Tolerance.default_for` applies.

    ``subnormals="flush"`` compares subnormal values on either side as zero,
    for a kernel whose core flushes denormals.

    ``overflow`` says how an integer reference that leaves the output range
    is brought into it, matching the kernel's declared behaviour:
    ``"wrap"`` (two's complement, the plain cast), ``"saturate"`` (clip) or
    ``"undefined"`` -- in which case an overflowing reference is not graded
    at all: the verdict fails and says which elements overflowed, since
    whatever the device produced there proves nothing.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if tol is None:
        tol = Tolerance.default_for(actual.dtype)
    if subnormals not in ("preserve", "flush", "unspecified"):
        raise ValueError(
            f"subnormals must be preserve, flush or unspecified, got {subnormals!r}"
        )
    if overflow not in ("wrap", "saturate", "undefined"):
        raise ValueError(
            f"overflow must be wrap, saturate or undefined, got {overflow!r}"
        )
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

    # Integers: bit-exact under exact / ulps; under a relative tolerance the
    # nearly_equal formula in int64 (Tolerance.lsb sets atol = n + 0.5).
    if np.issubdtype(actual.dtype, np.integer) or actual.dtype == np.bool_:
        if actual.dtype != np.bool_ and np.issubdtype(e.dtype, np.integer):
            info = np.iinfo(actual.dtype)
            e_wide = e.astype(np.int64)
            over = (e_wide < info.min) | (e_wide > info.max)
            if over.any():
                if overflow == "saturate":
                    e = np.clip(e_wide, info.min, info.max)
                elif overflow == "undefined":
                    n_over = int(np.count_nonzero(over))
                    return Verdict(
                        False,
                        n,
                        n_over,
                        float("inf"),
                        None,
                        int(np.argmax(over)),
                        f"reference overflows {np.dtype(actual.dtype).name} in "
                        f"{n_over} of {n} elements and the kernel declares "
                        "overflow='undefined': choose inputs that fit its "
                        "accumulator (kernel_harness.input_limit) or declare "
                        "'wrap' / 'saturate' in the contract",
                    )
        e_cast = e.astype(actual.dtype)
        a64, e64 = a.astype(np.int64), e_cast.astype(np.int64)
        err = np.abs(a64 - e64)
        if tol.kind == "relative":
            bound = np.maximum(
                tol.atol or 0.0, (tol.rtol or 0.0) * (np.abs(a64) + np.abs(e64))
            )
            bad = ~(err < bound)
        else:
            bad = a != e_cast
        return _verdict(bad, err, None, tol, n)

    a32, e32 = a.astype(np.float32), e.astype(np.float32)
    if subnormals == "flush":
        # The kernel treats subnormal inputs as zero, so a subnormal on either
        # side is compared as zero.
        tiny = np.float32(np.finfo(np.dtype(actual.dtype)).tiny)
        a32 = np.where(np.abs(a32) < tiny, np.float32(0), a32)
        e32 = np.where(np.abs(e32) < tiny, np.float32(0), e32)
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
        bad = nonfinite_bad | (finite & (ulp > max_ulps))
        return _verdict(bad, err, ulp, tol, n)

    if tol.kind == "exact":
        e_cast = e32.astype(actual.dtype).astype(np.float32)
        err[finite] = np.abs(a32[finite] - e_cast[finite])
        bad = nonfinite_bad | (finite & (a32 != e_cast))
        return _verdict(bad, err, None, tol, n)

    err[finite] = np.abs(a32[finite].astype(np.float64) - e32[finite])
    close = nearly_equal(a32, e32, rtol=tol.rtol or 0.0, atol=tol.atol)
    bad = nonfinite_bad | (finite & ~close)
    return _verdict(bad, err, None, tol, n)


def _verdict(bad, err, ulp, tol: Tolerance, n: int) -> Verdict:
    n_bad = int(np.count_nonzero(bad))
    ok = n_bad <= int(np.floor(tol.max_mismatch_frac * n))
    first = int(np.argmax(bad)) if n_bad else None
    max_ulp = int(ulp.max()) if ulp is not None and n else None
    max_err = float(err.max()) if n else 0.0
    if ok:
        detail = "ok"
    else:
        detail = f"{n_bad}/{n} mismatches; first at flat index {first}; max_abs_err={max_err:.4g}"
        if max_ulp is not None:
            detail += f"; max_ulp={max_ulp}"
        if tol.note:
            detail += f" [{tol.kind}: {tol.note}]"
    return Verdict(ok, n, n_bad, max_err, max_ulp, first, detail)
