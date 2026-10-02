#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Sweep the SA tile placer's seed at full effort, without any hardware.

Runs ``aie-opt --aie-place-tiles`` over each fixture for seeds 1..N, and
records whether each seed placed, the CPU time it took, its peak RSS, and the
SA placer's final cost (from --mlir-pass-statistics). The fixtures are a small
4-core design (the one test/place-tiles/sa_placer/test_sa_effort.mlir already
exercises) and mobilenet before placement (fixtures/mobilenet.mlir), where the
seed moves the cost. Seed 0 is left out: the placer reads it as "seed from the
clock", so its row could not be reproduced.

Writes ``<fixture>/<metric>`` rows and a meta file in the kernel checks'
format (utils/kernel_checks/publish.py), and exits 1 when any seed failed,
after writing both. nightlyComponentChecks.yml runs this nightly and
publishes the results; a regression is debuggable from the per-seed table
this prints to stdout, and one seed is reproduced with

    aie-opt '--aie-place-tiles=placer=sa_placer sa-seed=N sa-effort=1.0' \
        --mlir-pass-statistics <fixture>
"""

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys

_COST_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-final-cost")
_HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = [
    os.path.join(
        _HERE, "..", "..", "test", "place-tiles", "sa_placer", "test_sa_effort.mlir"
    ),
    os.path.join(_HERE, "fixtures", "mobilenet.mlir"),
]


def run_seed(aie_opt: str, fixture: str, seed: int, effort: float) -> dict:
    """Run one seed; return its pass/fail, CPU time, peak RSS, and cost.

    Uses os.wait4 (not resource.getrusage(RUSAGE_CHILDREN), which is a
    running max across every reaped child) so each seed's peak RSS is its
    own, not a stale max carried over from an earlier, bigger seed. CPU time,
    not wall time: a hosted runner shares its cores with other jobs.
    """
    args = [
        aie_opt,
        f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={effort}",
        "--mlir-pass-statistics",
        fixture,
        "-o",
        "/dev/null",
    ]
    proc = subprocess.Popen(
        args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    stderr = proc.stderr.read() if proc.stderr else ""
    _, status, ru = os.wait4(proc.pid, 0)
    cost_match = _COST_RE.search(stderr)
    return {
        "fixture": case_of(fixture),
        "seed": seed,
        "passed": os.waitstatus_to_exitcode(status) == 0,
        "cpu_ms": (ru.ru_utime + ru.ru_stime) * 1000.0,
        "peak_rss_mb": ru.ru_maxrss / 1024.0,
        "final_cost": int(cost_match.group(1)) if cost_match else None,
        "stderr": stderr,
    }


def case_of(fixture: str) -> str:
    return os.path.splitext(os.path.basename(fixture))[0]


def fixtures_digest(fixtures: list[str]) -> str:
    """Digest the fixtures, by name and content, into 12 hex digits."""
    h = hashlib.sha256()
    for path in fixtures:
        h.update(f"{case_of(path)}\0".encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:12]


def aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for case in dict.fromkeys(r["fixture"] for r in rows):
        mine = [r for r in rows if r["fixture"] == case]
        cpu_ms = [r["cpu_ms"] for r in mine]
        costs = [
            r["final_cost"] for r in mine if r["passed"] and r["final_cost"] is not None
        ]

        def row(metric, value, unit):
            return {"name": f"{case}/{metric}", "unit": unit, "value": value}

        out += [
            row("failed_seeds", sum(not r["passed"] for r in mine), "seeds"),
            row("cpu_ms_mean", round(statistics.mean(cpu_ms), 1), "ms"),
            row("cpu_ms_max", round(max(cpu_ms), 1), "ms"),
            row("peak_rss_mb_max", round(max(r["peak_rss_mb"] for r in mine), 1), "MB"),
        ]
        if costs:
            out.append(row("final_cost_mean", statistics.mean(costs), "cost"))
            out.append(row("final_cost_max", max(costs), "cost"))
    return out


def commit() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                cwd=_HERE,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            sha = "unknown"
    return sha[:10]


def meta(rows: list[dict], fixtures: list[str], seeds: int, effort: float) -> dict:
    failed = [f"{r['fixture']}/seed{r['seed']}" for r in rows if not r["passed"]]
    return {
        "preflight": {},
        "provenance": f"commit {commit()} | fixtures {fixtures_digest(fixtures)}",
        "measurement_sane": True,
        "seeds": seeds,
        "effort": effort,
        "failed": failed,
        "exitstatus": 1 if failed else 0,
    }


def print_table(rows: list[dict]) -> None:
    print(
        f"{'fixture':<16} {'seed':>5}  {'pass':>4}  {'cpu_ms':>9}  "
        f"{'rss_mb':>8}  {'cost':>8}"
    )
    for r in rows:
        cost = r["final_cost"] if r["final_cost"] is not None else "-"
        print(
            f"{r['fixture']:<16} {r['seed']:>5}  {'yes' if r['passed'] else 'NO':>4}  "
            f"{r['cpu_ms']:>9.1f}  {r['peak_rss_mb']:>8.1f}  {cost:>8}"
        )
        if not r["passed"]:
            print(f"        {r['stderr'].strip()}")


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--aie-opt", required=True, help="path to the aie-opt binary")
    parser.add_argument(
        "--fixture",
        action="append",
        help="MLIR module to place, repeatable (default: the sa-effort lit "
        "fixture and mobilenet)",
    )
    parser.add_argument(
        "--seeds", type=_positive_int, default=20, help="sweep seeds 1..N"
    )
    parser.add_argument("--effort", type=float, default=1.0)
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)

    fixtures = args.fixture or FIXTURES
    rows = [
        run_seed(args.aie_opt, fixture, seed, args.effort)
        for fixture in fixtures
        for seed in range(1, args.seeds + 1)
    ]
    print_table(rows)
    with open(args.out, "w") as f:
        json.dump(aggregate(rows), f, indent=1)
    info = meta(rows, fixtures, args.seeds, args.effort)
    with open(args.meta, "w") as f:
        json.dump(info, f, indent=1)
    return info["exitstatus"]


if __name__ == "__main__":
    sys.exit(main())
