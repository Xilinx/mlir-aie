# benchmarks/kernels/harness.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Measure one registry case: correctness first, then wall time, cycles, size.

The design, inputs, reference and verdict all come from
``aie.utils.kernel_harness`` and the kernel's contract; this module only adds
what a benchmark needs on top -- timing loops, a traced run for cycles, and a
forced rebuild for compile time and artifact sizes.

  wall      aie.utils.benchmark.run_iters       -> NPU time + e2e (median)
  cycles    kernel_harness.cycles_per_call      -> per-invocation core cycles
  compile   CallableDesign.compile(explicit paths) -> seconds, xclbin/insts/ELF bytes
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from aie.iron import kernels
from aie.utils import kernel_harness as kh
from aie.utils.benchmark import BenchmarkResult, run_iters

from .registry import Case, inputs_for

TRACE_SIZE = 16384


@dataclass
class Measurement:
    case: str
    device: str
    correct: bool
    verdict: str
    wall: BenchmarkResult | None = None
    cycles_median: int | None = None
    cycles_per_call: list[int] = field(default_factory=list)
    compile_s: float | None = None
    xclbin_bytes: int | None = None
    insts_bytes: int | None = None
    elf_bytes: int | None = None
    work_ops: int = 0
    error: str | None = None


def measure_compile(design, workdir: Path) -> tuple[float, int, int, int]:
    """Force a rebuild into ``workdir``, time it, and size the artifacts.

    ``CallableDesign.compile`` with explicit ``xclbin_path`` and
    ``inst_path`` bypasses the on-disk cache by contract and keeps its
    intermediates in ``<stem>.prj/`` next to the xclbin, so nothing about
    the cache layout has to be guessed or deleted. The cached build the
    timed runs use is untouched.

    Returns (seconds, xclbin_bytes, insts_bytes, sum_core_elf_bytes).
    """
    build = workdir / "compile"
    build.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    xclbin, insts = design.compile(
        xclbin_path=build / "final.xclbin", inst_path=build / "insts.bin"
    )
    secs = time.perf_counter() - t0
    # aiecc writes one "elfs_<core>.elf" per core (tools/aiecc/CommandLineOptions.h);
    # the harness designs request them with --get-core-elfs.
    prj = build / "final.prj"
    elf = sum(p.stat().st_size for p in prj.glob("elfs_*.elf")) if prj.is_dir() else 0
    insts_bytes = Path(insts).stat().st_size if insts else 0
    return secs, Path(xclbin).stat().st_size, insts_bytes, elf


def measure(
    case: Case,
    *,
    device: str,
    warmup: int,
    iters: int,
    do_cycles: bool,
    do_compile: bool,
    workdir: Path,
    seed: int = 0,
) -> Measurement:
    m = Measurement(case=case.name, device=device, correct=False, verdict="not run")
    try:
        m.work_ops = case.work()
        fn = case.fn()
        factory = getattr(kernels, case.factory)
        rng = np.random.default_rng(seed)
        inputs = inputs_for(case, "random", rng)
        # aiecc only writes the per-core ELFs when asked; measure_compile sizes them.
        design = kh.design(
            factory,
            **case.harness_opts(),
            params=kh.param_values(fn, inputs),
            aiecc_flags=["--get-core-elfs"],
            **case.kwargs,
        )

        # ---- correctness first (random data) -------------------------
        ref = kh.expected(fn, inputs, scalars=case.scalars)
        out_n = kh.output_size(fn, calls=case.calls, shape=case.shape)
        out_dt = kh.output_dtype(fn, ref.dtype)
        ins, out = kh.upload(inputs, out_n, out_dt, poison=True, fn=fn)
        if do_compile:
            m.compile_s, m.xclbin_bytes, m.insts_bytes, m.elf_bytes = measure_compile(
                design, workdir
            )
        design(*ins, out)
        v = kh.judge(fn, out.numpy(), ref, calls=case.calls)
        m.correct, m.verdict = bool(v), v.detail
        if not v:
            return m  # never time a wrong kernel

        # ---- wall time (same buffers) ---------------------------------
        m.wall = run_iters(design, *ins, out, warmup=warmup, iters=iters)

        # ---- cycles (separate traced run; trace perturbs timing) -------
        if do_cycles:
            m.cycles_per_call = kh.cycles_per_call(
                design,
                inputs,
                out_n,
                out_dt,
                trace_size=TRACE_SIZE,
                workdir=workdir,
                fn=fn,
            )
            if m.cycles_per_call:
                m.cycles_median = int(np.median(m.cycles_per_call))
    except Exception as ex:  # noqa: BLE001 - recorded, gated by run.py
        m.error = f"{type(ex).__name__}: {ex}"
    return m
