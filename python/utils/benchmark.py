# benchmark.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Benchmarking helpers for NPU kernel callables."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass
class Stats:
    avg_us: float
    min_us: float
    max_us: float


@dataclass
class BenchmarkResult:
    e2e: Stats
    npu: Stats | None  # None when the callable does not expose NPU time


def _stats(samples_us: list[float]) -> Stats:
    return Stats(
        avg_us=sum(samples_us) / len(samples_us),
        min_us=min(samples_us),
        max_us=max(samples_us),
    )


def _extract_npu_time_ns(ret) -> int | None:
    """Pull npu_time (ns) from a kernel callable's return value.

    Supports the ``(handle, KernelResult)`` tuple returned by
    ``HostRuntime.load_and_run`` (and therefore by ``NPUKernel.__call__`` and
    ``@iron.jit`` callables).  Returns ``None`` if the shape doesn't match,
    so the helper degrades to e2e-only timing for non-kernel callables.
    """
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def run_iters(
    fn: Callable,
    *args,
    warmup: int = 0,
    iters: int = 1,
    **kwargs,
) -> BenchmarkResult:
    """Invoke ``fn(*args, **kwargs)`` ``warmup + iters`` times, reporting timings.

    End-to-end latency is measured around the Python call.  If the return
    value carries an ``npu_time`` (nanoseconds, captured by the runtime
    around ``kernel.wait()``), it is reported separately so callers can see
    the host-side overhead delta.
    """
    if iters < 1:
        raise ValueError(f"iters must be >= 1 (got {iters})")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0 (got {warmup})")

    e2e_samples: list[float] = []
    npu_samples: list[float] = []
    for i in range(warmup + iters):
        start = time.perf_counter()
        ret = fn(*args, **kwargs)
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
        )
    s = result.e2e
    print(
        f"End-to-end   (avg/min/max us): {s.avg_us:.1f} / {s.min_us:.1f} / {s.max_us:.1f}"
    )
