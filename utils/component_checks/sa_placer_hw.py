#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile + run full mobilenet on NPU2 for a few fixed SA placer seeds.

Complements sa_placer.py's CPU-only seed sweep (placement pass alone, no
hardware) with the thing that actually matters: does the placed design still
compile, verify, and run at the expected latency on real hardware for a
handful of seeds. Shells out to the existing aie2_mobilenet_iron.py CLI (no
new Python API) from programming_examples/ml, one fresh NPU_CACHE_HOME per
seed so runs don't share a JIT cache (test/python/npu's isolation
convention).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

_PASS_RE = re.compile(r"^PASS!$", re.MULTILINE)
_NPU_TIME_RE = re.compile(
    r"NPU time\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)"
)

_ML_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml"
)


def run_seed(seed: int, warmup: int, iters: int) -> dict:
    """Run mobilenet for one SA seed; return pass/fail and min NPU latency."""
    cache_dir = tempfile.mkdtemp(prefix=f"component-checks-sa{seed}-")
    env = dict(os.environ, NPU_CACHE_HOME=cache_dir)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "mobilenet.aie2_mobilenet_iron",
                "--sa-seed",
                str(seed),
                "--sa-effort",
                "1.0",
                "-w",
                str(warmup),
                "-i",
                str(iters),
            ],
            cwd=_ML_DIR,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    out = proc.stdout + proc.stderr
    passed = proc.returncode == 0 and _PASS_RE.search(out) is not None
    npu_match = _NPU_TIME_RE.search(out)
    # min (group 2), not avg: a single busy neighbor call inflates the mean
    # far more than it moves the min (feedback_npu_latency_use_min_not_avg).
    min_latency_us = float(npu_match.group(2)) if npu_match else None
    return {
        "seed": seed,
        "passed": passed,
        "min_latency_us": min_latency_us,
        "log": out,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    failed = [r for r in rows if not r["passed"]]
    out = [{"name": "sa_placer/hw_fail_count", "unit": "seeds", "value": len(failed)}]
    for r in rows:
        if r["min_latency_us"] is not None:
            out.append(
                {
                    "name": f"sa_placer/hw_seed{r['seed']}_latency_us",
                    "unit": "us",
                    "value": r["min_latency_us"],
                }
            )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 2, 7],
        help="SA seeds to check on hardware (default: 3 2 7, the checked-in "
        "default plus the two other seeds already verified in "
        "project_mobilenet_placement_ablation)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    rows = [run_seed(seed, args.warmup, args.iters) for seed in args.seeds]
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        lat = (
            f"{r['min_latency_us']:.1f} us" if r["min_latency_us"] is not None else "-"
        )
        print(f"seed {r['seed']:>3}: {status}  min_latency={lat}")
        if not r["passed"]:
            print(r["log"])

    with open(args.out, "w") as f:
        json.dump(aggregate(rows), f, indent=1)
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
