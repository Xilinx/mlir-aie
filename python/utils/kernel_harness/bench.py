# bench.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
r"""Measure kernels: correctness first, then wall time, core cycles and size.

    python -m aie.utils.kernel_harness --cases test/python/npu/kernel_cases.py \\
        --out bench.json --only '^mm/'
    python -m aie.utils.kernel_harness add mul --calls 16 --out bench.json

Every measurement goes through the same harness the tests use, so a number is
only recorded for a kernel that just produced a correct result:

| metric            | how                                                      |
| ----------------- | -------------------------------------------------------- |
| ``cycles``        | trace, core ``INSTR_EVENT_0`` to ``INSTR_EVENT_1`` per kernel call, median |
| ``cycles_per_kop``| cycles per thousand arithmetic ops (the contract's ``ops_per_call``) |
| ``npu_us``, ``e2e_us`` | :func:`aie.utils.benchmark.run_iters`: kernel time reported by the runtime, and the Python call, median with min/max |
| ``compile_s``, ``xclbin_bytes``, ``insts_bytes``, ``core_elf_bytes`` | a forced rebuild through ``CallableDesign.compile`` into a scratch directory |

Order of a run: preflight (device, power mode) -> canary (a passthrough that
must be bit-exact and inside a cycle band, else the machine is not trusted)
-> every case -> gate -> JSON rows for benchmark-action. Exit codes: 0 wrote
results; 2 preflight or canary failed; 3 a kernel produced wrong output or
too many failed to run. On 2 and 3 nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from aie.iron import kernels
from aie.utils import kernel_harness as kh
from aie.utils.benchmark import BenchmarkResult, provenance, run_iters
from aie.utils.compile.utils import resolve_target_arch

from .cases import Case, inputs_for, load_cases

TRACE_SIZE = 16384

# A trivially correct kernel that must be bit-exact and inside a wide cycle
# band before anything else is measured; a machine that fails it is not
# trusted and nothing is recorded. The band is deliberately wide until nightly
# data has shown the real spread.
CANARY = Case("passthrough", dict(tile_size=2048), calls=16)
CANARY_CYCLE_BAND = (1_000, 2_000_000)


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

    ``CallableDesign.compile`` with explicit ``xclbin_path`` and ``inst_path``
    bypasses the on-disk cache by contract and keeps its intermediates in
    ``<stem>.prj/`` next to the xclbin, so nothing about the cache layout has
    to be guessed or deleted; the cached build the timed runs use is untouched.

    Returns (seconds, xclbin_bytes, insts_bytes, sum_core_elf_bytes).
    """
    build = workdir / "compile"
    build.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    xclbin, insts = design.compile(
        xclbin_path=build / "final.xclbin", inst_path=build / "insts.bin"
    )
    secs = time.perf_counter() - t0
    # aiecc writes one "elfs_<core>.elf" per core when asked (--get-core-elfs).
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
    """Run one case on the current device: check it, then time it."""
    m = Measurement(case=case.name, device=device, correct=False, verdict="not run")
    try:
        m.work_ops = case.work()
        fn = case.fn()
        factory = getattr(kernels, case.factory)
        rng = np.random.default_rng(seed)
        inputs = inputs_for(case, "random", rng)
        design = kh.design(
            factory,
            **case.harness_opts(),
            params=kh.param_values(fn, inputs),
            aiecc_flags=["--get-core-elfs"],
            **case.kwargs,
        )

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

        m.wall = run_iters(design, *ins, out, warmup=warmup, iters=iters)

        if do_cycles:  # a separate traced run: tracing perturbs the timing
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
    except Exception as ex:  # noqa: BLE001 - recorded, gated by main()
        m.error = f"{type(ex).__name__}: {ex}"
    return m


# --------------------------------------------------------------------------
# benchmark-action rows
# --------------------------------------------------------------------------


def row(name: str, unit: str, value, extra: str, rng: str | None = None) -> dict:
    r = {"name": name, "unit": unit, "value": value, "extra": extra}
    if rng:
        r["range"] = rng
    return r


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty benchmark file")
    Path(path).write_text(json.dumps(rows, indent=1))


def rows_for(m: Measurement, extra: str) -> list[dict]:
    """Return the benchmark-action rows one measurement contributes."""
    base = m.case
    out = []
    if m.cycles_median is not None:
        out.append(row(f"{base}/cycles", "cycles", m.cycles_median, extra))
        if m.work_ops:
            out.append(
                row(
                    f"{base}/cycles_per_kop",
                    "cycles/1k-ops",
                    round(1000.0 * m.cycles_median / m.work_ops, 3),
                    extra,
                )
            )
    if m.wall and m.wall.npu:
        s = m.wall.npu
        out.append(
            row(
                f"{base}/npu_us",
                "us",
                round(s.median_us, 2),
                extra,
                f"min {s.min_us:.1f} max {s.max_us:.1f} n={s.n}",
            )
        )
    if m.wall:
        s = m.wall.e2e
        out.append(
            row(
                f"{base}/e2e_us",
                "us",
                round(s.median_us, 2),
                extra,
                f"min {s.min_us:.1f} max {s.max_us:.1f} n={s.n}",
            )
        )
    if m.compile_s is not None:
        out.append(row(f"{base}/compile_s", "s", round(m.compile_s, 2), extra))
        out.append(row(f"{base}/xclbin_bytes", "bytes", m.xclbin_bytes, extra))
        out.append(row(f"{base}/insts_bytes", "bytes", m.insts_bytes, extra))
        out.append(row(f"{base}/core_elf_bytes", "bytes", m.elf_bytes, extra))
    return out


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Preflight:
    """What the runtime says about the device a run is about to use."""

    npu: str  # "npu1" / "npu2"
    arch: str  # "aie2" / "aie2p"
    device: str  # the device as the runtime describes it
    pmode: str  # power mode, or "unknown"


def preflight() -> Preflight:
    """Describe the device through the host runtime (any backend)."""
    import aie.utils as aie_utils

    runtime = aie_utils.DefaultNPURuntime
    if runtime is None:
        raise RuntimeError("no NPU runtime is available (XRT, HRX or HSA)")
    device = runtime.device()
    arch = resolve_target_arch(device)
    npu = "npu2" if arch == "aie2p" else "npu1"
    return Preflight(npu=npu, arch=arch, device=str(device), pmode=runtime.power_mode())


def _default_cases(names: list[str], calls: int) -> list[Case]:
    return [Case(n, calls=calls, smoke=True) for n in names]


def main(
    argv: list[str] | None = None,
    *,
    measure_fn: Callable[..., Measurement] = measure,
    preflight_fn: Callable[[], Preflight] = preflight,
) -> int:
    """Command-line entry point; ``measure_fn`` / ``preflight_fn`` are injectable for tests."""
    p = argparse.ArgumentParser(
        prog="python -m aie.utils.kernel_harness",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "kernels",
        nargs="*",
        help="factory names to time at their default shape (ignored with --cases)",
    )
    p.add_argument("--cases", help="Python file defining CASES (a list of Case)")
    p.add_argument("--only", help="regex on case names")
    p.add_argument(
        "--calls", type=int, default=16, help="calls per run for bare kernel names"
    )
    p.add_argument("--out", required=True, help="benchmark-action JSON to write")
    p.add_argument("--meta", help="JSON with preflight, canary and per-case detail")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument(
        "--pmode",
        default="performance",
        help="required power mode as the runtime reports it; 'any' to skip the check",
    )
    p.add_argument("--no-cycles", action="store_true", help="skip the traced run")
    p.add_argument("--no-compile", action="store_true", help="skip the rebuild")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    meta: dict = {}

    try:
        pre = preflight_fn()
        if a.pmode != "any" and pre.pmode != a.pmode:
            raise RuntimeError(
                f"power mode is '{pre.pmode}', required '{a.pmode}' "
                "(set it with xrt-smi configure --pmode, or pass --pmode any)"
            )
    except Exception as ex:  # noqa: BLE001 - reported, nothing written
        meta["preflight_error"] = str(ex)
        _write(a.meta, meta)
        print(f"PREFLIGHT FAILED: {ex}", file=sys.stderr)
        return 2
    meta["preflight"] = pre.__dict__
    extra = provenance(device=pre.device, pmode=pre.pmode)
    common = dict(
        device=pre.device,
        warmup=a.warmup,
        iters=a.iters,
        do_cycles=not a.no_cycles,
        do_compile=not a.no_compile,
        workdir=Path(tempfile.mkdtemp(prefix="aie-bench-")),
        seed=a.seed,
    )

    if a.cases:
        cases = [c for c in load_cases(a.cases) if c.perf]
    else:
        if not a.kernels:
            p.error("give kernel names or --cases FILE")
        cases = _default_cases(a.kernels, a.calls)
    if a.only:
        import re

        cases = [c for c in cases if re.search(a.only, c.name)]

    can = measure_fn(CANARY, **common)
    lo, hi = CANARY_CYCLE_BAND
    in_band = can.cycles_median is None or lo <= can.cycles_median <= hi
    if not can.correct or can.error or not in_band:
        meta["canary"] = _m2d(can)
        _write(a.meta, meta)
        print(
            f"PREFLIGHT FAILED: canary {can.verdict} {can.error or ''} "
            f"cycles={can.cycles_median} band={CANARY_CYCLE_BAND}",
            file=sys.stderr,
        )
        return 2

    results = [can]
    for case in cases:
        if case.name == CANARY.name:
            continue
        if not case.supported_on(pre.npu):
            continue  # the kernel's source exists only for the other generation
        m = measure_fn(case, **common)
        results.append(m)
        status = "OK" if m.correct and not m.error else f"FAIL {m.error or m.verdict}"
        print(f"[{status}] {m.case} cycles={m.cycles_median}")

    wrong = [m for m in results if not m.error and not m.correct]
    errored = [m for m in results if m.error]
    meta["summary"] = {
        "n": len(results),
        "n_wrong": len(wrong),
        "n_error": len(errored),
        "failed": [f"{m.case}: {m.error or m.verdict}" for m in wrong + errored],
    }
    _write(a.meta, meta)
    if wrong:
        print(
            f"RESULTS INVALID: {len(wrong)} kernel(s) produced wrong output",
            file=sys.stderr,
        )
        return 3
    if len(errored) > 0.1 * len(results):
        print(
            f"RESULTS INVALID: {len(errored)}/{len(results)} kernels failed to run",
            file=sys.stderr,
        )
        return 3

    rows = [r for m in results if m.correct and not m.error for r in rows_for(m, extra)]
    write_rows(a.out, rows)
    print(f"wrote {len(rows)} metrics to {a.out}")
    return 0


def _m2d(m: Measurement) -> dict:
    d = {k: v for k, v in m.__dict__.items() if k != "wall"}
    if m.wall:
        d["npu"] = m.wall.npu.as_dict() if m.wall.npu else None
        d["e2e"] = m.wall.e2e.as_dict()
    return d


def _write(path, obj):
    if path:
        Path(path).write_text(json.dumps(obj, indent=1, default=str))


__all__ = [
    "CANARY",
    "CANARY_CYCLE_BAND",
    "Measurement",
    "Preflight",
    "main",
    "measure",
    "measure_compile",
    "preflight",
    "rows_for",
    "write_rows",
]
