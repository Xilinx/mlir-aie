#!/usr/bin/env python3
# benchmarks/kernels/run.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Nightly kernel benchmark driver.

    python -m benchmarks.kernels.run --out bench.json --meta meta.json

Order: preflight (device + pmode) -> canary (must be correct and inside its
cycle band) -> every perf case (correctness first; wrong kernels are never
timed) -> write bench.json only if every kernel that ran was correct.

Exit codes: 0 wrote results; 2 preflight/canary failed; 3 results invalid.
On 2/3 nothing is written, so downstream steps have nothing to publish.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import registry
from ._util import PreflightError, npu_arch, preflight, provenance, row, write_rows
from .harness import Measurement, measure


def rows_for(m: Measurement, extra: str) -> list[dict]:
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


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--meta")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--pmode", default="performance")
    p.add_argument("--no-cycles", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--only", help="regex on kernel name")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    meta: dict = {}

    try:
        pre = preflight(a.pmode)
    except (PreflightError, Exception) as ex:  # noqa: BLE001
        meta["preflight_error"] = str(ex)
        _write(a.meta, meta)
        print(f"PREFLIGHT FAILED: {ex}", file=sys.stderr)
        return 2
    meta["preflight"] = pre
    extra = provenance(pre["pmode"])
    common = dict(
        device=pre["npu_name"],
        warmup=a.warmup,
        iters=a.iters,
        do_cycles=not a.no_cycles,
        do_compile=not a.no_compile,
        workdir=Path(tempfile.mkdtemp(prefix="aie-bench-")),
        seed=a.seed,
    )

    # ---- canary: bad machine detector ------------------------------------
    can = measure(registry.CANARY, **common)
    lo, hi = registry.CANARY_CYCLE_BAND
    in_band = can.cycles_median is None or lo <= can.cycles_median <= hi
    if not can.correct or can.error or not in_band:
        meta["canary"] = _m2d(can)
        _write(a.meta, meta)
        print(
            f"PREFLIGHT FAILED: canary {can.verdict} {can.error or ''} "
            f"cycles={can.cycles_median} band={registry.CANARY_CYCLE_BAND}",
            file=sys.stderr,
        )
        return 2

    # ---- all perf cases ----------------------------------------------------
    results = [can]
    for case in registry.perf_cases(a.only):
        if case.name == registry.CANARY.name:
            continue
        if case.arch and case.arch != _arch(pre):
            continue  # source exists only for the other architecture
        m = measure(case, **common)
        results.append(m)
        status = "OK" if m.correct and not m.error else f"FAIL {m.error or m.verdict}"
        print(f"[{status}] {m.case} cycles={m.cycles_median}")

    # ---- gate ---------------------------------------------------------------
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


def _arch(pre: dict) -> str:
    """The arch a case's ``arch`` field names, from the preflight's device name."""
    return npu_arch(pre["npu_name"])


def _m2d(m: Measurement) -> dict:
    d = {k: v for k, v in m.__dict__.items() if k != "wall"}
    if m.wall:
        d["npu"] = m.wall.npu.as_dict() if m.wall.npu else None
        d["e2e"] = m.wall.e2e.as_dict()
    return d


def _write(path, obj):
    if path:
        Path(path).write_text(json.dumps(obj, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
