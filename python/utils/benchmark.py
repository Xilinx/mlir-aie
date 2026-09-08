# benchmark.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Benchmarking helpers for NPU kernel callables.

``Stats`` keeps the raw sample list and robust statistics (median, MAD,
p95, coefficient of variation) beside avg/min/max, so callers can report
jitter, not just central tendency.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Stats:
    avg_us: float
    min_us: float
    max_us: float
    median_us: float = 0.0
    mad_us: float = 0.0  # median absolute deviation (robust spread)
    p95_us: float = 0.0
    cov: float = 0.0  # stdev / mean; 0 for a single sample
    n: int = 0
    samples_us: list[float] = field(default_factory=list, repr=False)

    @classmethod
    def from_samples(cls, samples_us: list[float]) -> "Stats":
        if not samples_us:
            raise ValueError("Stats.from_samples needs at least one sample")
        s = sorted(samples_us)
        med = statistics.median(s)
        mad = statistics.median(abs(x - med) for x in s)
        p95 = s[min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)]
        mean = sum(s) / len(s)
        cov = (statistics.stdev(s) / mean) if len(s) > 1 and mean > 0 else 0.0
        return cls(
            avg_us=mean,
            min_us=s[0],
            max_us=s[-1],
            median_us=med,
            mad_us=mad,
            p95_us=p95,
            cov=cov,
            n=len(s),
            samples_us=list(samples_us),
        )

    def as_dict(self, prefix: str = "") -> dict:
        """Flat dict for JSON emission (samples omitted)."""
        return {
            f"{prefix}median_us": self.median_us,
            f"{prefix}mad_us": self.mad_us,
            f"{prefix}p95_us": self.p95_us,
            f"{prefix}min_us": self.min_us,
            f"{prefix}max_us": self.max_us,
            f"{prefix}avg_us": self.avg_us,
            f"{prefix}cov": self.cov,
            f"{prefix}n": self.n,
        }


@dataclass
class BenchmarkResult:
    e2e: Stats
    npu: Stats | None  # None when the callable does not expose NPU time


def _stats(samples_us: list[float]) -> Stats:
    return Stats.from_samples(samples_us)


def _extract_npu_time_ns(ret) -> int | None:
    """Pull npu_time (ns) from a kernel callable's return value.

    Supports the ``(handle, KernelResult)`` tuple returned by
    ``HostRuntime.load_and_run`` (and therefore by ``NPUKernel.__call__`` and
    ``@iron.jit`` callables). Returns ``None`` if the shape doesn't match,
    so the helper degrades to e2e-only timing for non-kernel callables.
    """
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def run_iters(
    fn: Callable,
    *args,
    warmup: int = 0,
    iters: int = 1,
    arg_sets: list[tuple] | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Invoke ``fn`` ``warmup + iters`` times, reporting timings.

    End-to-end latency is measured around the Python call. If the return
    value carries an ``npu_time`` (nanoseconds, captured by the runtime
    around ``kernel.wait()``), it is reported separately so callers can see
    the host-side overhead delta.

    ``arg_sets``: an optional list of positional-argument tuples to
    rotate through, one per iteration, so consecutive runs do not hit the
    same host buffers (buffer rotation; see the CUTLASS measurement
    guidelines). When given, ``*args`` must be empty.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1 (got {iters})")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0 (got {warmup})")
    if arg_sets is not None and args:
        raise ValueError("pass either *args or arg_sets, not both")
    if arg_sets is not None and not arg_sets:
        raise ValueError("arg_sets must hold at least one argument tuple")
    rotation = arg_sets if arg_sets is not None else [args]

    e2e_samples: list[float] = []
    npu_samples: list[float] = []
    for i in range(warmup + iters):
        call_args = rotation[i % len(rotation)]
        start = time.perf_counter()
        ret = fn(*call_args, **kwargs)
        e2e_us = (time.perf_counter() - start) * 1_000_000
        if i < warmup:
            continue
        e2e_samples.append(e2e_us)
        npu_ns = _extract_npu_time_ns(ret)
        if npu_ns is not None:
            npu_samples.append(npu_ns / 1_000.0)

    return BenchmarkResult(
        e2e=_stats(e2e_samples),
        npu=_stats(npu_samples) if npu_samples else None,
    )


def print_benchmark(result: BenchmarkResult) -> None:
    """Print a BenchmarkResult in the canonical 'avg/min/max us' format."""
    if result.npu is not None:
        s = result.npu
        print(
            f"NPU time     (avg/min/max us): {s.avg_us:.1f} / {s.min_us:.1f} / {s.max_us:.1f}"
            f"   [median {s.median_us:.1f}, MAD {s.mad_us:.1f}, p95 {s.p95_us:.1f}]"
        )
    s = result.e2e
    print(
        f"End-to-end   (avg/min/max us): {s.avg_us:.1f} / {s.min_us:.1f} / {s.max_us:.1f}"
        f"   [median {s.median_us:.1f}, MAD {s.mad_us:.1f}, p95 {s.p95_us:.1f}]"
    )
