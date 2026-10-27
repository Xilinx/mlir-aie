#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Compile + run full mobilenet on NPU2 for a few SA placer seeds and batch sizes.

Complements sa_placer.py's CPU-only seed sweep (placement pass alone, no
hardware) with the thing that actually matters: does the placed design still
compile, verify, and run at the expected speed on real hardware for a
handful of seeds. Shells out to the existing aie2_mobilenet_iron.py CLI
from programming_examples/ml, one fresh NPU_CACHE_HOME per seed and effort
so they don't share a JIT cache (test/python/npu's isolation convention).

Each seed runs at both of ``EFFORTS``: the design's real default, read from
mobilenet.bottleneck._common.sa_placer_flags' own signature, and full effort
as a placement-quality baseline. The batch is a dispatch-time parameter, so
per seed and effort the design compiles once (the CLI's ``--dynamic``
compile-only mode) and every batch runs from it. It compiles twice in the
same fresh NPU_CACHE_HOME: cold (every cache empty, same as any first-time
compile), then warm, after aiecc_stages.clear_for_warm (only the kernel
objects and their precompiled headers kept), which is what an edit to the
design, not its kernels, costs. Both record wall time, the largest
process's peak RSS, aiecc's stages and the kernel objects' compile time. Per batch it records, per image, the fastest launch's
dispatch time, the NPU time (the dispatch less the same run's empty-dispatch
baseline), the host overhead around it and the host's whole end-to-end call
(parse_run); the run builds its own copy of the design through the JIT path,
from the warm caches, unmeasured. Per seed,
per effort, it records the placement: aiecc's placed tiles as a digest, and
its SA cost, from replaying the placement with aie-opt on the design's MLIR
(aiecc does not report it).

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode, the
toolchain, the dispatch mode and the efforts; ``--pmode`` refuses to measure
in any other mode. Before the first compile it writes a placeholder that
fails the check, so a run stopped part way still publishes as a failure.
Every command has a timeout and its output is echoed as it comes. Exits 1
when a run fails or the check refuses, after writing both. One seed and
effort is reproduced, from programming_examples/ml, with

    python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort E \
        --dynamic 1 2 4 8 16 32 64
"""

import argparse
import glob
import hashlib
import inspect
import os
import re
import shutil
import subprocess
import sys
import tempfile

from aie.utils.benchmark import preflight, provenance
from aie.utils.compile.utils import AIECC_PROFILE_NAME, KERNEL_PROFILE_NAME
from aiecc_stages import (
    add_pmode_argument,
    clear_for_warm,
    compile_rows,
    compile_stats,
    kernel_rows,
    refuse,
    row,
    run_measured,
    stage_rows,
    write,
    write_unfinished,
)

_ML_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "programming_examples", "ml"
)
sys.path.insert(0, _ML_DIR)
from mobilenet.bottleneck._common import sa_placer_flags  # noqa: E402

_PASS_RE = re.compile(r"^PASS!$", re.MULTILINE)
_BATCH_RE = re.compile(r"^batch=(\d+)$", re.MULTILINE)
# aie.utils.benchmark.print_benchmark's lines, by the field each one times.
_TIME_RES = {
    field: re.compile(
        rf"^{label}\s+\(avg/min/max us\):\s*[\d.]+\s*/\s*(?P<min>[\d.]+)\s*/\s*[\d.]+\s*"
        r"\[median (?P<median>[\d.]+), MAD (?P<mad>[\d.]+), p95 (?P<p95>[\d.]+)\]",
        re.MULTILINE,
    )
    for field, label in (
        ("dispatch", "Dispatch"),
        ("dispatch_overhead", "Dispatch overhead"),
        ("npu", "NPU"),
        ("e2e", "End-to-end"),
    )
}
# Each per-image row, from the run's field of the same name.
TIMED = (
    "us_per_image",
    "npu_us_per_image",
    "host_overhead_us_per_image",
    "e2e_us_per_image",
)
_COST_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-final-cost")
_TILE_RE = re.compile(r"aie\.tile\(\d+, \d+\)")
EFFORTS = {
    "default": inspect.signature(sa_placer_flags).parameters["effort"].default,
    "full": 1.0,
}
_MOBILENET = [sys.executable, "-m", "mobilenet.aie2_mobilenet_iron"]


def placement_of(mlir: str) -> str | None:
    """Digest the placed tiles, in the order the design names them."""
    tiles = _TILE_RE.findall(mlir)
    return hashlib.sha256("\n".join(tiles).encode()).hexdigest()[:12] if tiles else None


def replay_placement(
    aie_opt: str, design: str, seed: int, effort: float, timeout: float
) -> tuple[str | None, int | None]:
    """Place ``design`` as aiecc does for ``seed``/``effort``: (digest, SA cost)."""
    try:
        proc = subprocess.run(
            [
                aie_opt,
                f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={effort}",
                "--mlir-pass-statistics",
                design,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"aie-opt replaying seed {seed} timed out after {timeout:g} s")
        return None, None
    cost = _COST_RE.search(proc.stderr)
    if proc.returncode or not cost:
        return None, None
    return placement_of(proc.stdout), int(cost.group(1))


def parse_run(out: str, returncode: int, batch: int) -> dict:
    """Read a run of the mobilenet CLI: whether it passed, and its times per image.

    Each time is the fastest call over the batch, its range the median, MAD
    and p95 over the batch too: us_per_image is the dispatch (submit to
    completion), npu_us_per_image the dispatch less the empty-dispatch
    baseline, e2e_us_per_image the whole Python call, and
    host_overhead_us_per_image the fastest end-to-end call less the fastest
    dispatch (its median likewise, its MAD the sum of both).
    dispatch_overhead_us is the baseline itself, per launch.
    """
    t = {}
    for field, pattern in _TIME_RES.items():
        if match := pattern.search(out):
            t[field] = {k: float(v) for k, v in match.groupdict().items()}
    # Time is what this check tracks, so a run that stops printing it (the
    # output format drifted, say) fails rather than publishing nothing.
    run = {
        "passed": returncode == 0
        and _PASS_RE.search(out) is not None
        and "dispatch" in t,
        "dispatch_overhead_us": t.get("dispatch_overhead", {}).get("min"),
    }
    if "dispatch" in t and "e2e" in t:
        t["host_overhead"] = {
            k: t["e2e"][k] - t["dispatch"][k] for k in ("min", "median", "p95")
        } | {"mad": t["e2e"]["mad"] + t["dispatch"]["mad"]}
    for metric, field in zip(TIMED, ("dispatch", "npu", "host_overhead", "e2e")):
        # min, not avg: a single busy neighbor call inflates the mean far
        # more than it moves the min.
        x = {k: v / batch for k, v in t[field].items()} if field in t else None
        run[metric] = round(x["min"], 2) if x else None
        run[f"{metric}_range"] = (
            f"median {x['median']:.2f}, MAD {x['mad']:.2f}, p95 {x['p95']:.2f}"
            if x
            else None
        )
    return run


def split_batches(out: str) -> dict[int, str]:
    """Split a ``--dynamic`` run's output into each batch's own lines."""
    marks = list(_BATCH_RE.finditer(out))
    ends = [m.start() for m in marks[1:]] + [len(out)]
    return {int(m.group(1)): out[m.end() : end] for m, end in zip(marks, ends)}


def compile_cold_warm(
    flags: list[str], batches: list[int], cache_dir: str, timeout: float
) -> dict:
    """Compile the dispatch-time design cold, then warm, into ``cache_dir``."""
    env = dict(os.environ, NPU_CACHE_HOME=cache_dir)
    cmd = [
        *_MOBILENET,
        *flags,
        "--dynamic",
        *map(str, batches),
        "--xclbin-path",
        os.path.join(cache_dir, "final.xclbin"),
    ]
    cold = run_measured(cmd, _ML_DIR, env, timeout, tree_peak=True)
    profile = os.path.join(cache_dir, "final.prj", AIECC_PROFILE_NAME)
    kernels = os.path.join(cache_dir, "final.prj", KERNEL_PROFILE_NAME)
    cold_s = cold.seconds if cold.process.returncode == 0 else None
    stages_cold = stage_rows(profile, cold_s, "") | kernel_rows(kernels, cold_s, "")
    placed = sorted(glob.glob(os.path.join(cache_dir, "devices", "*", "placed.mlir")))
    placement = None
    if placed:
        with open(placed[0]) as f:
            placement = placement_of(f.read())
    clear_for_warm(cache_dir)
    warm = run_measured(cmd, _ML_DIR, env, timeout, tree_peak=True)
    warm_s = warm.seconds if warm.process.returncode == 0 else None
    stages_warm = stage_rows(profile, warm_s, "_warm") | kernel_rows(
        kernels, warm_s, "_warm"
    )
    return {
        **stages_cold,
        **stages_warm,
        "passed": cold.process.returncode == 0 and warm.process.returncode == 0,
        **compile_stats(cold, warm),
        "placement": placement,
    }


def run_seed(
    seed: int,
    effort_label: str,
    batches: list[int],
    warmup: int,
    iters: int,
    compile_timeout: float,
    run_timeout: float,
) -> tuple[dict, list[dict]]:
    """Compile mobilenet once for one SA seed and effort, and run every batch."""
    cache_dir = tempfile.mkdtemp(prefix=f"component-checks-sa{seed}-{effort_label}-")
    flags = ["--sa-seed", str(seed), "--sa-effort", str(EFFORTS[effort_label])]
    try:
        compiled = compile_cold_warm(flags, batches, cache_dir, compile_timeout)
        ran = run_measured(
            [
                *_MOBILENET,
                *flags,
                "--dynamic",
                *map(str, batches),
                "-w",
                str(warmup),
                "-i",
                str(iters),
            ],
            _ML_DIR,
            dict(os.environ, NPU_CACHE_HOME=cache_dir),
            run_timeout,
        ).process
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    out = ran.stdout
    compiled = {"seed": seed, "effort_label": effort_label, **compiled}
    segments = split_batches(out)
    runs = [
        {
            "seed": seed,
            "batch": b,
            "effort_label": effort_label,
            **parse_run(segments.get(b, ""), 0, b),
            "placement": compiled["placement"],
        }
        for b in batches
    ]
    # A nonzero exit no batch's own output explains (a crash after the last
    # PASS, say) is not a pass.
    if not compiled["passed"] or (ran.returncode and all(r["passed"] for r in runs)):
        for r in runs:
            r["passed"] = False
    return compiled, runs


def place_seed(
    compiles: list[dict], aie_opt: str, design: str, timeout: float
) -> list[dict]:
    """Replay each of one seed's compiles' placement: digest and SA cost."""
    out = []
    for c in compiles:
        replayed, cost = replay_placement(
            aie_opt, design, c["seed"], EFFORTS[c["effort_label"]], timeout
        )
        first = c["placement"] or replayed
        out.append(
            {
                "seed": c["seed"],
                "effort_label": c["effort_label"],
                "placement": first,
                # The cost is only the compile's if the replay placed the
                # design the same.
                "placement_cost": cost if first == replayed else None,
                "replay_matches": first == replayed,
            }
        )
    return out


def aggregate(
    runs: list[dict], compiles: list[dict], placements: list[dict]
) -> list[dict]:
    failed_seeds = {r["seed"] for r in runs + compiles if not r["passed"]}
    out = [row("mobilenet", "failed_seeds", len(failed_seeds), "seeds")]
    for p in placements:
        if p["placement_cost"] is not None:
            out.append(
                row(
                    f"mobilenet/seed={p['seed']}/effort={p['effort_label']}",
                    "placement_cost",
                    p["placement_cost"],
                    "cost",
                    f"placement {p['placement']}",
                )
            )
    for c in compiles:
        out += compile_rows(f"mobilenet/seed={c['seed']}/effort={c['effort_label']}", c)
    # A failed run's times (say, a run that printed timing but produced the
    # wrong answer) are not a measurement of the design, so only passing runs
    # contribute a time point; the failure shows up in the count.
    for r in runs:
        case = (
            f"mobilenet/seed={r['seed']}/batch={r['batch']}"
            f"/effort={r['effort_label']}"
        )
        if not r["passed"]:
            continue
        out += [
            row(case, metric, r[metric], "us", r[f"{metric}_range"])
            for metric in TIMED
            if r[metric] is not None
        ]
    return out


def print_table(runs: list[dict], compiles: list[dict], placements: list[dict]) -> None:
    rows = {r["name"]: r["value"] for r in aggregate(runs, compiles, placements)}

    def cell(case, metric):
        value = rows.get(f"{case}/{metric}")
        return "-" if value is None else f"{value:.1f}"

    print(
        f"{'seed':>4} {'effort':>7}  {'pass':>4}  {'compile_s':>9}  {'peak_rss_mb':>11}  "
        f"{'warm_s':>7}  {'warm_rss_mb':>11}"
    )
    for c in compiles:
        case = f"mobilenet/seed={c['seed']}/effort={c['effort_label']}"
        print(
            f"{c['seed']:>4} {c['effort_label']:>7}  {'yes' if c['passed'] else 'NO':>4}  "
            f"{cell(case, 'compile_s'):>9}  {cell(case, 'compile_peak_rss_mb'):>11}  "
            f"{cell(case, 'compile_s_warm'):>7}  "
            f"{cell(case, 'compile_peak_rss_mb_warm'):>11}"
        )
    print(
        f"{'seed':>4} {'batch':>5} {'effort':>7}  {'pass':>4}  {'us/image':>9}  "
        f"{'e2e/image':>9}"
    )
    for r in runs:
        case = (
            f"mobilenet/seed={r['seed']}/batch={r['batch']}"
            f"/effort={r['effort_label']}"
        )
        print(
            f"{r['seed']:>4} {r['batch']:>5} {r['effort_label']:>7}  "
            f"{'yes' if r['passed'] else 'NO':>4}  "
            f"{cell(case, 'us_per_image'):>9}  {cell(case, 'e2e_us_per_image'):>9}"
        )
        if not r["passed"] and r["us_per_image"] is None:
            print("no 'Dispatch (avg/min/max us)' line in this batch's output")
    for p in placements:
        print(
            f"seed {p['seed']} effort {p['effort_label']}: "
            f"placement {p['placement']} cost {p['placement_cost']}"
        )
        if not p["replay_matches"]:
            print("aie-opt placed the design differently from aiecc; no cost recorded")


def emit_design(path: str, timeout: float) -> None:
    """Write mobilenet's MLIR, before placement, to ``path``."""
    try:
        proc = subprocess.run(
            [*_MOBILENET, "--emit-mlir"],
            cwd=_ML_DIR,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        sys.exit(f"mobilenet --emit-mlir timed out after {timeout:g} s")
    if proc.returncode:
        sys.exit(f"mobilenet --emit-mlir failed:\n{proc.stdout}{proc.stderr}")
    with open(path, "w") as f:
        f.write(proc.stdout)


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
        default=[1, 2, 4, 8, 16, 32, 64],
        help="images per launch to run each seed at (default: every power of "
        "two from 1 to 64)",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--compile-timeout",
        type=_positive_int,
        default=1200,
        help="seconds before a compile is killed and fails (default: 1200)",
    )
    parser.add_argument(
        "--run-timeout",
        type=_positive_int,
        default=900,
        help="seconds before a seed's run of every batch is killed and fails "
        "(default: 900)",
    )
    add_pmode_argument(parser)
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

    pre = preflight()
    meta = {
        "preflight": dict(vars(pre)),
        "provenance": provenance(
            device=pre.device, pmode=pre.pmode, dispatch="dynamic"
        ),
        "efforts": EFFORTS,
        "iters": args.iters,
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        return refuse(
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'",
            meta,
            args.out,
            args.meta,
        )

    write_unfinished(meta, "mobilenet", args.out, args.meta)
    batches = sorted(set(args.batches))
    with tempfile.TemporaryDirectory(prefix="component-checks-design-") as d:
        design = os.path.join(d, "mobilenet.mlir")
        emit_design(design, args.compile_timeout)
        runs, compiles, placements = [], [], []
        for seed in args.seeds:
            mine = []
            for effort_label in EFFORTS:
                compiled, seed_runs = run_seed(
                    seed,
                    effort_label,
                    batches,
                    args.warmup,
                    args.iters,
                    args.compile_timeout,
                    args.run_timeout,
                )
                mine.append(compiled)
                runs += seed_runs
            compiles += mine
            placements += place_seed(mine, args.aie_opt, design, args.compile_timeout)
    print_table(runs, compiles, placements)

    failed = [
        f"mobilenet/seed={r['seed']}/batch={r['batch']}/effort={r['effort_label']}"
        for r in runs
        if not r["passed"]
    ]
    write(args.out, aggregate(runs, compiles, placements))
    detail = [
        {
            k: r[k]
            for k in (
                "seed",
                "batch",
                "effort_label",
                "passed",
                *TIMED,
                "dispatch_overhead_us",
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
            "compiles": compiles,
            "placements": placements,
            "detail": detail,
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
