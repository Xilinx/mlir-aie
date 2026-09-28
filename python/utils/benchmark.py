# benchmark.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Benchmarking helpers for NPU kernel callables.

``Stats`` keeps the raw sample list and robust statistics (median, MAD,
p95, coefficient of variation) beside avg/min/max, so callers can report
jitter, not just central tendency. The numbers are numpy's.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Callable

import numpy as np
from aie.utils import config


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
        s = np.asarray(samples_us, dtype=np.float64)
        med = float(np.median(s))
        mean = float(s.mean())
        # p95 is a sample, not an interpolation between two: at benchmark
        # sample counts the interpolating percentile methods report a duration
        # no run took. inverted_cdf is numpy's name for that (nearest-rank)
        # definition, and every published chart was recorded with it.
        return cls(
            avg_us=mean,
            min_us=float(s.min()),
            max_us=float(s.max()),
            median_us=med,
            mad_us=float(np.median(np.abs(s - med))),
            p95_us=float(np.percentile(s, 95, method="inverted_cdf")),
            cov=float(s.std(ddof=1) / mean) if len(s) > 1 and mean > 0 else 0.0,
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


def peano_version() -> str | None:
    """Return the Peano that compiles the kernels, as ``21.0.0+c9c5ecb7``.

    Asked of the compiler itself rather than read from the ``llvm-aie``
    distribution: ``PEANO_INSTALL_DIR`` can select any build, and the
    installed wheel then names a compiler that never ran. ``None`` when no
    compiler is found.
    """
    try:
        out = subprocess.run(
            [config.peano_cxx_path(), "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, RuntimeError, subprocess.CalledProcessError):
        return None
    m = re.match(r"clang version (\S+) \(\S+ ([0-9a-f]{8})", out)
    if m:
        return f"{m[1]}+{m[2]}"
    return out.splitlines()[0] if out else None


def kernel_tree_digest() -> str | None:
    """Return a 12-hex digest of the kernel sources the library factories compile.

    Every file under ``aie_kernels_dir()`` and ``aie_runtime_lib_dir()``, by
    relative path and content. The commit alone cannot say which kernels
    ran: ``MLIR_AIE_KERNEL_SOURCES`` can name another tree, and a checkout
    can carry uncommitted edits. ``None`` when neither directory exists.
    """
    h = hashlib.sha256()
    found = False
    for top in (config.aie_kernels_dir(), config.aie_runtime_lib_dir()):
        root = Path(top)
        if not root.is_dir():
            continue
        found = True
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            h.update(f"{root.name}/{path.relative_to(root)}\0".encode())
            h.update(path.read_bytes())
    return h.hexdigest()[:12] if found else None


def provenance(**extra: str | None) -> str:
    """Return a one-line description of what produced a measurement.

    The git commit (``GITHUB_SHA`` or ``git rev-parse HEAD``), the Peano that
    compiles the kernels (``peano_version()``), the installed ``mlir_aie``
    version, the kernel tree
    (``MLIR_AIE_KERNEL_SOURCES`` when set, and ``kernel_tree_digest()``),
    and any ``extra`` fields (``device="NPU Strix"``, ``pmode="performance"``)
    as ``key value`` pairs. A benchmark row records it so a number can be
    traced to a toolchain and to the kernel sources.

    A package that is not installed is left out rather than recorded as
    unknown. CI builds ``mlir_aie`` from source and puts it on ``PYTHONPATH``,
    so there is no distribution to read a version from, and every published
    row would otherwise carry a word that reads like a lookup failure. The
    commit already identifies that build.
    """

    def pkg(name: str) -> str | None:
        try:
            return _pkg_version(name)
        except PackageNotFoundError:
            return None

    commit = os.environ.get("GITHUB_SHA")
    if not commit:
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            commit = "unknown"
    fields = {
        "commit": commit[:10],
        "peano": peano_version(),
        "mlir_aie": pkg("mlir_aie"),
        "kernel_sources": os.environ.get("MLIR_AIE_KERNEL_SOURCES"),
        "kernels": kernel_tree_digest(),
        **extra,
    }
    return " | ".join(f"{k} {v}" for k, v in fields.items() if v)


@dataclass(frozen=True)
class Preflight:
    """What the runtime says about the device a run is about to use."""

    npu: str  # "npu1" / "npu2"
    arch: str  # "aie2" / "aie2p"
    device: str  # the device as the runtime describes it
    pmode: str | None  # power mode, or None if it could not be read


def preflight() -> Preflight:
    """Describe the device through the host runtime (any backend)."""
    import aie.utils as aie_utils
    from aie.utils.compile.utils import resolve_target_arch

    runtime = aie_utils.DefaultNPURuntime
    if runtime is None:
        raise RuntimeError("no NPU runtime is available (XRT, HRX or HSA)")
    device = runtime.device()
    arch = resolve_target_arch(device)
    return Preflight(
        npu="npu2" if arch == "aie2p" else "npu1",
        arch=arch,
        device=runtime.device_name() or type(device).__name__,
        pmode=runtime.power_mode(),
    )
