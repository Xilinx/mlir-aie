#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Sweep the SA tile placer's seed at full effort, without any hardware.

Runs ``aie-opt --aie-place-tiles`` over a fixed 4-core fixture (the same
design test/place-tiles/sa_placer/test_sa_effort.mlir already exercises) for
seeds 0..N-1, and records whether each seed placed, how long it took, its
peak RSS, and the SA placer's final cost (from --mlir-pass-statistics).
nightlyComponentChecks.yml runs this nightly and publishes the aggregate to
gh-pages; a real regression is still debuggable from the per-seed table this
prints to stdout.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time

_COST_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-final-cost")


def run_seed(aie_opt: str, fixture: str, seed: int, effort: float) -> dict:
    """Run one seed; return its pass/fail, wall time, peak RSS, and cost.

    Uses os.wait4 (not resource.getrusage(RUSAGE_CHILDREN), which is a
    running max across every reaped child) so each seed's peak RSS is its
    own, not a stale max carried over from an earlier, bigger seed.
    """
    args = [
        aie_opt,
        f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={effort}",
        "--mlir-pass-statistics",
        fixture,
        "-o",
        "/dev/null",
    ]
    start = time.perf_counter()
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    stderr = proc.stderr.read()
    _, status, ru = os.wait4(proc.pid, 0)
    wall_ms = (time.perf_counter() - start) * 1000.0
    passed = os.waitstatus_to_exitcode(status) == 0
    cost_match = _COST_RE.search(stderr)
    return {
        "seed": seed,
        "passed": passed,
        "wall_ms": wall_ms,
        "peak_rss_mb": ru.ru_maxrss / 1024.0,
        "final_cost": int(cost_match.group(1)) if cost_match else None,
        "stderr": stderr,
    }


def sweep(aie_opt: str, fixture: str, seeds: int, effort: float) -> list[dict]:
    return [run_seed(aie_opt, fixture, seed, effort) for seed in range(seeds)]


def aggregate(rows: list[dict]) -> list[dict]:
    failed = [r for r in rows if not r["passed"]]
    ok = [r for r in rows if r["passed"]]
    wall_ms = [r["wall_ms"] for r in rows]
    rss_mb = [r["peak_rss_mb"] for r in rows]
    costs = [r["final_cost"] for r in ok if r["final_cost"] is not None]

    def row(name, value, unit):
        return {"name": f"sa_placer/{name}", "unit": unit, "value": value}

    out = [
        row("fail_count", len(failed), "seeds"),
        row("mean_wall_time_ms", statistics.mean(wall_ms), "ms"),
        row("max_wall_time_ms", max(wall_ms), "ms"),
        row("mean_peak_rss_mb", statistics.mean(rss_mb), "MB"),
        row("max_peak_rss_mb", max(rss_mb), "MB"),
    ]
    if costs:
        out.append(row("mean_final_cost", statistics.mean(costs), "cost"))
        out.append(row("max_final_cost", max(costs), "cost"))
    return out


def print_table(rows: list[dict]) -> None:
    print(f"{'seed':>5}  {'pass':>4}  {'wall_ms':>9}  {'rss_mb':>8}  {'cost':>8}")
    for r in rows:
        cost = r["final_cost"] if r["final_cost"] is not None else "-"
        print(
            f"{r['seed']:>5}  {'yes' if r['passed'] else 'NO':>4}  "
            f"{r['wall_ms']:>9.1f}  {r['peak_rss_mb']:>8.1f}  {cost:>8}"
        )
        if not r["passed"]:
            print(f"        {r['stderr'].strip()}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--aie-opt", required=True, help="path to the aie-opt binary")
    parser.add_argument(
        "--fixture",
        default=os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "test",
            "place-tiles",
            "sa_placer",
            "test_sa_effort.mlir",
        ),
        help="MLIR module to place (default: the sa-effort lit fixture)",
    )
    parser.add_argument("--seeds", type=int, default=20, help="sweep seeds 0..N-1")
    parser.add_argument("--effort", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = sweep(args.aie_opt, args.fixture, args.seeds, args.effort)
    print_table(rows)
    with open(args.out, "w") as f:
        json.dump(aggregate(rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
