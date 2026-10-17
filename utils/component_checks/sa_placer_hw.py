#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Compile + run full mobilenet on NPU2 for a few SA placer seeds and batch sizes.

Complements sa_placer.py's CPU-only seed sweep (placement pass alone, no
hardware) with the thing that actually matters: does the placed design still
compile, verify, and run at the expected speed on real hardware for a
handful of seeds. Shells out to the existing aie2_mobilenet_iron.py CLI (no
new Python API) from programming_examples/ml, one fresh NPU_CACHE_HOME per
run so runs don't share a JIT cache (test/python/npu's isolation
convention).

Each seed runs at each batch size at both of _EFFORTS (the design's real
default, and full effort as a placement-quality baseline -- see
mobilenet.bottleneck._common.sa_placer_flags, which the default must keep
matching). Per run it compiles twice in the same fresh NPU_CACHE_HOME (the
CLI's compile-only mode): once cold (kernel objects built from scratch, same
as any first-time compile) and immediately again, warm (the same cache_dir,
so it reuses the cold compile's kernel objects) -- kernel compilation, not
placement, is almost all of a cold compile's time (measured: ~19.5s of
MobileNet's ~20-27s), so the warm number is what an incremental compile
actually costs. Both record wall time and peak RSS. Per run it also records,
per image, the minimum NPU time of a launch and of the host's end-to-end
call.
From a seed's batch 1 and batch N it records the streaming time: what one
more image in a launch costs, (T(N) - T(1)) / (N - 1), with the launch's
fixed cost taken out. Per seed, per effort, it records the placement:
aiecc's placed tiles as a digest, and its SA cost, from replaying the
placement with aie-opt on the design's MLIR (aiecc does not report it).
Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when a
run fails or the check refuses, after writing both. One run is reproduced,
from programming_examples/ml, with

    python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort E --batch B
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
_TIME_RE = (
    r"\s+\(avg/min/max us\):\s*[\d.]+\s*/\s*(?P<min>[\d.]+)\s*/\s*[\d.]+\s*"
    r"\[median (?P<median>[\d.]+), MAD (?P<mad>[\d.]+), p95 (?P<p95>[\d.]+)\]"
)
_NPU_TIME_RE = re.compile(r"NPU time" + _TIME_RE)
_E2E_TIME_RE = re.compile(r"End-to-end" + _TIME_RE)
_COST_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-final-cost")
_TILE_RE = re.compile(r"aie\.tile\(\d+, \d+\)")
# "default" must keep matching mobilenet.bottleneck._common.sa_placer_flags'
# own default -- not imported from there since this check otherwise only
# shells out to the CLI (see the module docstring); a change to either is a
# prompt to check the other.
_EFFORTS = {"default": 0.05, "full": 1.0}

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


def _mobilenet_rss(
    *args: str, env: dict | None = None
) -> tuple[subprocess.CompletedProcess, float]:
    """Like ``_mobilenet``, plus the process's own peak RSS in MB.

    Uses os.wait4 (not resource.getrusage(RUSAGE_CHILDREN), which is a running
    max across every reaped child -- see sa_placer.py's run_seed), so this
    call's peak RSS is its own, not a stale max from an earlier, bigger case.
    stdout and stderr are merged (rather than piped separately and read with
    two blocking .read() calls, which can deadlock once either pipe fills).
    """
    proc = subprocess.Popen(
        [sys.executable, "-m", "mobilenet.aie2_mobilenet_iron", *args],
        cwd=_ML_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout.read() if proc.stdout else ""
    _, status, ru = os.wait4(proc.pid, 0)
    result = subprocess.CompletedProcess(
        proc.args, os.waitstatus_to_exitcode(status), out, ""
    )
    return result, ru.ru_maxrss / 1024.0


def placement_of(mlir: str) -> str | None:
    """Digest the placed tiles, in the order the design names them."""
    tiles = _TILE_RE.findall(mlir)
    return hashlib.sha256("\n".join(tiles).encode()).hexdigest()[:12] if tiles else None


def replay_placement(
    aie_opt: str, design: str, seed: int, effort: float
) -> tuple[str | None, int | None]:
    """Place ``design`` as aiecc does for ``seed``/``effort``: (digest, SA cost)."""
    proc = subprocess.run(
        [
            aie_opt,
            f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={effort}",
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


def _per_image(match: re.Match | None, batch: int) -> tuple[float, str] | None:
    if not match:
        return None
    t = {k: float(v) / batch for k, v in match.groupdict().items()}
    return (
        round(t["min"], 2),
        f"median {t['median']:.2f}, MAD {t['mad']:.2f}, p95 {t['p95']:.2f}",
    )


def parse_run(out: str, returncode: int, batch: int) -> dict:
    """Read a run of the mobilenet CLI: whether it passed, and its times per image."""
    npu = _NPU_TIME_RE.search(out)
    # Time is what this check tracks, so a run that stops printing it (the
    # output format drifted, say) fails rather than publishing nothing.
    per_image = _per_image(npu, batch)
    e2e = _per_image(_E2E_TIME_RE.search(out), batch)
    return {
        "passed": returncode == 0 and _PASS_RE.search(out) is not None and bool(npu),
        # min, not avg: a single busy neighbor call inflates the mean far
        # more than it moves the min.
        "launch_us": float(npu.group("min")) if npu else None,
        "us_per_image": per_image[0] if per_image else None,
        "us_per_image_range": per_image[1] if per_image else None,
        "e2e_us_per_image": e2e[0] if e2e else None,
        "e2e_range": e2e[1] if e2e else None,
    }


def run_one(seed: int, batch: int, warmup: int, iters: int, effort_label: str) -> dict:
    """Compile and run mobilenet for one SA seed at one batch size and effort."""
    effort = _EFFORTS[effort_label]
    cache_dir = tempfile.mkdtemp(
        prefix=f"component-checks-sa{seed}-b{batch}-{effort_label}-"
    )
    env = dict(os.environ, NPU_CACHE_HOME=cache_dir)
    flags = ["--sa-seed", str(seed), "--sa-effort", str(effort)]
    flags += ["--batch", str(batch)]
    compile_args = (
        *flags,
        "--xclbin-path",
        os.path.join(cache_dir, "final.xclbin"),
        "--insts-path",
        os.path.join(cache_dir, "insts.bin"),
    )
    try:
        # Cold: a fresh cache_dir, so kernel objects are compiled from
        # scratch, same as any first-time compile.
        start = time.perf_counter()
        built, compile_peak_rss_mb = _mobilenet_rss(*compile_args, env=env)
        compile_s = time.perf_counter() - start
        # Warm: immediately after, same cache_dir, so this one reuses the
        # kernel objects the cold compile just produced. Kernel compilation
        # -- not placement -- is almost all of a cold compile's time
        # (measured: ~19.5s of MobileNet's ~20-27s), so this is the number
        # that actually reflects repeat/incremental compiles.
        start = time.perf_counter()
        built_warm, compile_peak_rss_mb_warm = _mobilenet_rss(*compile_args, env=env)
        compile_s_warm = time.perf_counter() - start
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

    result = parse_run(ran.stdout + ran.stderr, ran.returncode, batch)
    result["passed"] = (
        result["passed"] and built.returncode == 0 and built_warm.returncode == 0
    )
    return {
        "seed": seed,
        "batch": batch,
        "effort_label": effort_label,
        **result,
        "compile_s": round(compile_s, 1) if built.returncode == 0 else None,
        "compile_peak_rss_mb": (
            round(compile_peak_rss_mb, 1) if built.returncode == 0 else None
        ),
        "compile_s_warm": (
            round(compile_s_warm, 1) if built_warm.returncode == 0 else None
        ),
        "compile_peak_rss_mb_warm": (
            round(compile_peak_rss_mb_warm, 1) if built_warm.returncode == 0 else None
        ),
        "placement": compiled,
        "log": built.stdout
        + built.stderr
        + built_warm.stdout
        + built_warm.stderr
        + ran.stdout
        + ran.stderr,
    }


def place_seed(runs: list[dict], aie_opt: str, design: str) -> list[dict]:
    """Replay one seed's placement at each effort: digest, SA cost, batches placed apart."""
    seed = runs[0]["seed"]
    out = []
    for effort_label in _EFFORTS:
        mine = [r for r in runs if r["effort_label"] == effort_label]
        replayed, cost = replay_placement(aie_opt, design, seed, _EFFORTS[effort_label])
        compiled = [r["placement"] for r in mine if r["placement"]]
        first = compiled[0] if compiled else replayed
        out.append(
            {
                "seed": seed,
                "effort_label": effort_label,
                "placement": first,
                # The cost is only the compile's if the replay placed the
                # design the same.
                "placement_cost": cost if first == replayed else None,
                "replay_matches": first == replayed,
                # The runtime sequence is all a batch changes, and the placer
                # should not see it; a batch placed apart from the first says
                # it did. Compared only within this effort: different efforts
                # are expected to place differently.
                "placed_apart": [
                    r["batch"] for r in mine if r["placement"] not in (None, first)
                ],
            }
        )
    return out


def streaming_us(one: dict, many: dict) -> float:
    """Return what one more image in a launch costs, from runs at batch 1 and N."""
    return round((many["launch_us"] - one["launch_us"]) / (many["batch"] - 1), 2)


def aggregate(runs: list[dict], placements: list[dict]) -> list[dict]:
    failed_seeds = {r["seed"] for r in runs if not r["passed"]}
    out = [
        {"name": "mobilenet/failed_seeds", "unit": "seeds", "value": len(failed_seeds)}
    ]
    for p in placements:
        if p["placement_cost"] is not None:
            out.append(
                {
                    "name": f"mobilenet/seed={p['seed']}/effort={p['effort_label']}"
                    "/placement_cost",
                    "unit": "cost",
                    "value": p["placement_cost"],
                    "range": f"placement {p['placement']}",
                }
            )
    # A failed run's times (say, a run that printed timing but produced the
    # wrong answer) are not a measurement of the design, so only passing runs
    # contribute a time point; the failure shows up in the count. Keyed by
    # effort too: streaming_us must compare within the same effort, since
    # different efforts are expected to place (and so time) differently.
    one = {
        (r["seed"], r["effort_label"]): r
        for r in runs
        if r["batch"] == 1 and r["passed"] and r["launch_us"]
    }
    for r in runs:
        case = (
            f"mobilenet/seed={r['seed']}/batch={r['batch']}"
            f"/effort={r['effort_label']}"
        )
        if r["passed"] and r["us_per_image"] is not None:
            out.append(
                {
                    "name": f"{case}/us_per_image",
                    "unit": "us",
                    "value": r["us_per_image"],
                    "range": r["us_per_image_range"],
                }
            )
            key = (r["seed"], r["effort_label"])
            if r["batch"] > 1 and key in one:
                b1 = one[key]
                out.append(
                    {
                        "name": f"{case}/streaming_us",
                        "unit": "us",
                        "value": streaming_us(b1, r),
                        "range": f"from b1 {b1['launch_us']:g} and "
                        f"b{r['batch']} {r['launch_us']:g}",
                    }
                )
            if r["e2e_us_per_image"] is not None:
                out.append(
                    {
                        "name": f"{case}/e2e_us_per_image",
                        "unit": "us",
                        "value": r["e2e_us_per_image"],
                        "range": r["e2e_range"],
                    }
                )
        if r["compile_s"] is not None:
            out.append(
                {"name": f"{case}/compile_s", "unit": "s", "value": r["compile_s"]}
            )
        if r["compile_peak_rss_mb"] is not None:
            out.append(
                {
                    "name": f"{case}/compile_peak_rss_mb",
                    "unit": "MB",
                    "value": r["compile_peak_rss_mb"],
                }
            )
        if r["compile_s_warm"] is not None:
            out.append(
                {
                    "name": f"{case}/compile_s_warm",
                    "unit": "s",
                    "value": r["compile_s_warm"],
                }
            )
        if r["compile_peak_rss_mb_warm"] is not None:
            out.append(
                {
                    "name": f"{case}/compile_peak_rss_mb_warm",
                    "unit": "MB",
                    "value": r["compile_peak_rss_mb_warm"],
                }
            )
    return out


def print_table(runs: list[dict], placements: list[dict]) -> None:
    print(
        f"{'seed':>4} {'batch':>5} {'effort':>7}  {'pass':>4}  {'us/image':>9}  "
        f"{'streaming':>9}  {'e2e/image':>9}  {'compile_s':>9}  {'peak_rss_mb':>11}  "
        f"{'warm_s':>7}  {'warm_rss_mb':>11}"
    )
    rows = {r["name"]: r["value"] for r in aggregate(runs, placements)}
    for r in runs:
        case = (
            f"mobilenet/seed={r['seed']}/batch={r['batch']}"
            f"/effort={r['effort_label']}"
        )

        def cell(metric):
            value = rows.get(f"{case}/{metric}")
            return "-" if value is None else f"{value:.1f}"

        print(
            f"{r['seed']:>4} {r['batch']:>5} {r['effort_label']:>7}  "
            f"{'yes' if r['passed'] else 'NO':>4}  "
            f"{cell('us_per_image'):>9}  {cell('streaming_us'):>9}  "
            f"{cell('e2e_us_per_image'):>9}  {cell('compile_s'):>9}  "
            f"{cell('compile_peak_rss_mb'):>11}  {cell('compile_s_warm'):>7}  "
            f"{cell('compile_peak_rss_mb_warm'):>11}"
        )
        if not r["passed"]:
            if r["launch_us"] is None:
                print("no 'NPU time (avg/min/max us)' line in the output")
            print(r["log"])
    for p in placements:
        print(
            f"seed {p['seed']} effort {p['effort_label']}: "
            f"placement {p['placement']} cost {p['placement_cost']}"
        )
        if not p["replay_matches"]:
            print("aie-opt placed the design differently from aiecc; no cost recorded")
        if p["placed_apart"]:
            print(f"batches {p['placed_apart']} placed apart from the first")


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


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


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
    parser.add_argument(
        "--batches",
        type=_positive_int,
        nargs="+",
        default=[1, 4, 16, 64],
        help="images per launch to run each seed at (default: 1 4 16 64; the "
        "streaming time needs 1)",
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
        "efforts": _EFFORTS,
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

    batches = sorted(set(args.batches))
    with tempfile.TemporaryDirectory(prefix="component-checks-design-") as d:
        design = os.path.join(d, "mobilenet.mlir")
        emit_design(design)
        runs, placements = [], []
        for seed in args.seeds:
            mine = [
                run_one(seed, b, args.warmup, args.iters, effort_label)
                for b in batches
                for effort_label in _EFFORTS
            ]
            runs += mine
            placements += place_seed(mine, args.aie_opt, design)
    print_table(runs, placements)

    failed = [
        f"mobilenet/seed={r['seed']}/batch={r['batch']}/effort={r['effort_label']}"
        for r in runs
        if not r["passed"]
    ]
    write(args.out, aggregate(runs, placements))
    detail = [
        {
            k: r[k]
            for k in (
                "seed",
                "batch",
                "effort_label",
                "passed",
                "us_per_image",
                "e2e_us_per_image",
                "compile_s",
                "compile_peak_rss_mb",
                "compile_s_warm",
                "compile_peak_rss_mb_warm",
                "placement",
            )
        }
        for r in runs
    ]
    write(
        args.meta,
        {
            **meta,
            "batches": batches,
            "measurement_sane": True,
            "placements": placements,
            "detail": detail,
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
