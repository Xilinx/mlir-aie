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

Per seed it records the compile's wall time (the CLI's compile-only mode),
the minimum NPU latency, and the placement: aiecc's placed tiles as a digest,
and its SA cost, from replaying the placement with aie-opt on the design's
MLIR (aiecc does not report it). Writes rows and a meta file in the kernel
checks' format (utils/kernel_checks/publish.py), with the device's power mode
and the toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1
when a seed fails or the check refuses, after writing both. One seed is
reproduced, from programming_examples/ml, with

    python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort 1.0
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

_PASS_RE = re.compile(r"^PASS!$", re.MULTILINE)
_NPU_TIME_RE = re.compile(
    r"NPU time\s+\(avg/min/max us\):\s*([\d.]+)\s*/\s*([\d.]+)\s*/\s*([\d.]+)"
)
_COST_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-final-cost")
_TILE_RE = re.compile(r"aie\.tile\(\d+, \d+\)")
_EFFORT = 1.0

_ML_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml"
)


def _mobilenet(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "mobilenet.aie2_mobilenet_iron", *args],
        cwd=_ML_DIR,
        env=env,
        capture_output=True,
        text=True,
    )


def placement_of(mlir: str) -> str | None:
    """Digest the placed tiles, in the order the design names them."""
    tiles = _TILE_RE.findall(mlir)
    return hashlib.sha256("\n".join(tiles).encode()).hexdigest()[:12] if tiles else None


def replay_placement(
    aie_opt: str, design: str, seed: int
) -> tuple[str | None, int | None]:
    """Place ``design`` as aiecc does for ``seed``: (placement digest, SA cost)."""
    proc = subprocess.run(
        [
            aie_opt,
            f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={_EFFORT}",
            "--mlir-pass-statistics",
            design,
        ],
        capture_output=True,
        text=True,
    )
    cost = _COST_RE.search(proc.stderr)
    if proc.returncode or not cost:
        return None, None
    return placement_of(proc.stdout), int(cost.group(1))


def parse_run(out: str, returncode: int) -> dict:
    """Read a run of the mobilenet CLI: whether it passed, and its NPU latencies."""
    npu = _NPU_TIME_RE.search(out)
    # Latency is what this check tracks, so a run that stops printing it (the
    # output format drifted, say) fails rather than publishing nothing.
    return {
        "passed": returncode == 0 and _PASS_RE.search(out) is not None and bool(npu),
        # min, not avg: a single busy neighbor call inflates the mean far
        # more than it moves the min.
        "latency_us": float(npu.group(2)) if npu else None,
        "latency_range": f"avg {npu.group(1)}, max {npu.group(3)}" if npu else None,
    }


def run_seed(seed: int, warmup: int, iters: int, aie_opt: str, design: str) -> dict:
    """Compile and run mobilenet for one SA seed."""
    cache_dir = tempfile.mkdtemp(prefix=f"component-checks-sa{seed}-")
    env = dict(os.environ, NPU_CACHE_HOME=cache_dir)
    flags = ["--sa-seed", str(seed), "--sa-effort", str(_EFFORT)]
    try:
        start = time.perf_counter()
        built = _mobilenet(
            *flags,
            "--xclbin-path",
            os.path.join(cache_dir, "final.xclbin"),
            "--insts-path",
            os.path.join(cache_dir, "insts.bin"),
            env=env,
        )
        compile_s = time.perf_counter() - start
        placed = sorted(
            glob.glob(os.path.join(cache_dir, "devices", "*", "placed.mlir"))
        )
        compiled = None
        if placed:
            with open(placed[0]) as f:
                compiled = placement_of(f.read())
        ran = _mobilenet(*flags, "-w", str(warmup), "-i", str(iters), env=env)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    log = built.stdout + built.stderr + ran.stdout + ran.stderr
    result = parse_run(ran.stdout + ran.stderr, ran.returncode)
    result["passed"] = result["passed"] and built.returncode == 0
    replayed, cost = replay_placement(aie_opt, design, seed)
    # The cost is only the compile's if the replay placed the design the same.
    same = compiled is None or replayed == compiled
    return {
        "seed": seed,
        **result,
        "compile_s": round(compile_s, 1) if built.returncode == 0 else None,
        "placement": compiled or replayed,
        "placement_cost": cost if same else None,
        "replay_matches": same,
        "log": log,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    failed = [r for r in rows if not r["passed"]]
    out = [{"name": "mobilenet/failed_seeds", "unit": "seeds", "value": len(failed)}]
    # A failed seed's latency (say, a run that printed timing but produced the
    # wrong answer) is not a measurement of the design, so only passing seeds
    # contribute a latency point; the failure shows up in the count.
    for r in rows:
        case = f"mobilenet/seed{r['seed']}"
        if r["passed"] and r["latency_us"] is not None:
            out.append(
                {
                    "name": f"{case}/latency_us",
                    "unit": "us",
                    "value": r["latency_us"],
                    "range": r["latency_range"],
                }
            )
        if r["compile_s"] is not None:
            out.append(
                {"name": f"{case}/compile_s", "unit": "s", "value": r["compile_s"]}
            )
        if r["placement_cost"] is not None:
            out.append(
                {
                    "name": f"{case}/placement_cost",
                    "unit": "cost",
                    "value": r["placement_cost"],
                    "range": f"placement {r['placement']}",
                }
            )
    return out


def emit_design(path: str) -> None:
    """Write mobilenet's MLIR, before placement, to ``path``."""
    proc = _mobilenet("--emit-mlir")
    if proc.returncode:
        sys.exit(f"mobilenet --emit-mlir failed:\n{proc.stdout}{proc.stderr}")
    with open(path, "w") as f:
        f.write(proc.stdout)


def write(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[3, 2, 7],
        help="SA seeds to check on hardware (default: 3 2 7, the design's "
        "default seed plus two others known to place and verify)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--pmode",
        default="any",
        help="refuse to measure unless the NPU is in this power mode " "(default: any)",
    )
    parser.add_argument(
        "--aie-opt",
        default=shutil.which("aie-opt"),
        help="aie-opt to replay the placement with (default: the one on PATH)",
    )
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)
    if not args.aie_opt:
        parser.error("no aie-opt on PATH; pass --aie-opt")

    from aie.utils.benchmark import preflight, provenance

    pre = preflight()
    meta = {
        "preflight": dict(vars(pre)),
        "provenance": provenance(device=pre.device, pmode=pre.pmode),
        "effort": _EFFORT,
        "iters": args.iters,
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        meta["refused"] = (
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'"
        )
        print(f"refusing to measure: {meta['refused']}")
        write(args.out, [])
        write(args.meta, {**meta, "failed": [], "exitstatus": 1})
        return 1

    with tempfile.TemporaryDirectory(prefix="component-checks-design-") as d:
        design = os.path.join(d, "mobilenet.mlir")
        emit_design(design)
        rows = [
            run_seed(seed, args.warmup, args.iters, args.aie_opt, design)
            for seed in args.seeds
        ]
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        lat = f"{r['latency_us']:.1f} us" if r["latency_us"] is not None else "-"
        print(
            f"seed {r['seed']:>3}: {status}  min_latency={lat}  "
            f"compile={r['compile_s']} s  placement={r['placement']} "
            f"cost={r['placement_cost']}"
        )
        if not r["replay_matches"]:
            print("aie-opt placed the design differently from aiecc; no cost recorded")
        if not r["passed"]:
            if r["latency_us"] is None:
                print("no 'NPU time (avg/min/max us)' line in the output")
            print(r["log"])

    failed = [f"mobilenet/seed{r['seed']}" for r in rows if not r["passed"]]
    write(args.out, aggregate(rows))
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": True,
            "seeds": [
                {k: r[k] for k in r if k not in ("log", "latency_range")} for r in rows
            ],
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
