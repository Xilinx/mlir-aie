#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Compile + run full mobilenet on NPU2 for a few SA placer seeds and batch sizes.

Complements sa_placer.py's CPU-only seed sweep (placement pass alone, no
hardware) with the thing that actually matters: does the placed design still
compile, verify, and run at the expected speed on real hardware for a
handful of seeds. Shells out to the existing aie2_mobilenet_iron.py CLI (no
new Python API) from programming_examples/ml, one fresh NPU_CACHE_HOME per
seed and effort so they don't share a JIT cache (test/python/npu's
isolation convention).

Each seed runs at both of _EFFORTS (the design's real default, and full
effort as a placement-quality baseline -- see
mobilenet.bottleneck._common.sa_placer_flags, which the default must keep
matching). The batch is a dispatch-time parameter, so per seed and effort
the design compiles once (the CLI's ``--dynamic`` compile-only mode) and
every batch runs from it. It compiles twice in the same fresh
NPU_CACHE_HOME: cold (every cache empty, same as any first-time compile),
then warm, keeping only the cold compile's kernel objects (the whole-build
and per-device caches and the outputs are cleared first), which is what an
edit to the design, not its kernels, costs. Both record wall time and peak
RSS. Per batch it records, per image, the minimum NPU time of a launch and
of the host's end-to-end call; the run builds its own copy of the design
through the JIT path, from the warm caches, unmeasured.
From a seed's batch 1 and batch N it records the streaming time: what one
more image in a launch costs, (T(N) - T(1)) / (N - 1), with the launch's
fixed cost taken out. Per seed, per effort, it records the placement:
aiecc's placed tiles as a digest, and its SA cost, from replaying the
placement with aie-opt on the design's MLIR (aiecc does not report it).
Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode, the
toolchain and the dispatch mode; ``--pmode`` refuses to measure in any
other mode. Exits 1 when a run fails or the check refuses, after writing
both. One seed and effort is reproduced, from programming_examples/ml, with

    python -m mobilenet.aie2_mobilenet_iron --sa-seed N --sa-effort E \
        --dynamic 1 2 4 8 16 32 64
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

from aiecc_stages import PROFILE_NAME, STAGE_METRICS, stage_rows

_PASS_RE = re.compile(r"^PASS!$", re.MULTILINE)
_BATCH_RE = re.compile(r"^batch=(\d+)$", re.MULTILINE)
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


def split_batches(out: str) -> dict[int, str]:
    """Split a ``--dynamic`` run's output into each batch's own lines."""
    marks = list(_BATCH_RE.finditer(out))
    ends = [m.start() for m in marks[1:]] + [len(out)]
    return {int(m.group(1)): out[m.end() : end] for m, end in zip(marks, ends)}


def compile_cold_warm(flags: list[str], batches: list[int], cache_dir: str) -> dict:
    """Compile the dispatch-time design cold, then warm, into ``cache_dir``."""
    env = dict(os.environ, NPU_CACHE_HOME=cache_dir)
    xclbin = os.path.join(cache_dir, "final.xclbin")
    args = (*flags, "--dynamic", *map(str, batches), "--xclbin-path", xclbin)
    start = time.perf_counter()
    cold, rss_cold = _mobilenet_rss(*args, env=env)
    compile_s = time.perf_counter() - start
    profile = os.path.join(cache_dir, "final.prj", PROFILE_NAME)
    stages_cold = stage_rows(profile, compile_s if cold.returncode == 0 else None, "")
    placed = sorted(glob.glob(os.path.join(cache_dir, "devices", "*", "placed.mlir")))
    placement = None
    if placed:
        with open(placed[0]) as f:
            placement = placement_of(f.read())
    if os.path.exists(xclbin):
        os.remove(xclbin)
    for layer in ("final.prj", "builds", "devices"):
        shutil.rmtree(os.path.join(cache_dir, layer), ignore_errors=True)
    start = time.perf_counter()
    warm, rss_warm = _mobilenet_rss(*args, env=env)
    compile_s_warm = time.perf_counter() - start
    stages_warm = stage_rows(
        profile, compile_s_warm if warm.returncode == 0 else None, "_warm"
    )
    return {
        **stages_cold,
        **stages_warm,
        "passed": cold.returncode == 0 and warm.returncode == 0,
        "compile_s": round(compile_s, 1) if cold.returncode == 0 else None,
        "compile_peak_rss_mb": round(rss_cold, 1) if cold.returncode == 0 else None,
        "compile_s_warm": round(compile_s_warm, 1) if warm.returncode == 0 else None,
        "compile_peak_rss_mb_warm": (
            round(rss_warm, 1) if warm.returncode == 0 else None
        ),
        "placement": placement,
        "log": cold.stdout + warm.stdout,
    }


def run_seed(
    seed: int, effort_label: str, batches: list[int], warmup: int, iters: int
) -> tuple[dict, list[dict]]:
    """Compile mobilenet once for one SA seed and effort, and run every batch."""
    cache_dir = tempfile.mkdtemp(prefix=f"component-checks-sa{seed}-{effort_label}-")
    flags = ["--sa-seed", str(seed), "--sa-effort", str(_EFFORTS[effort_label])]
    try:
        compiled = compile_cold_warm(flags, batches, cache_dir)
        ran = _mobilenet(
            *flags,
            "--dynamic",
            *map(str, batches),
            "-w",
            str(warmup),
            "-i",
            str(iters),
            env=dict(os.environ, NPU_CACHE_HOME=cache_dir),
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    out = ran.stdout + ran.stderr
    compiled = {"seed": seed, "effort_label": effort_label, **compiled}
    compiled["log"] += out
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


def place_seed(compiles: list[dict], aie_opt: str, design: str) -> list[dict]:
    """Replay each of one seed's compiles' placement: digest and SA cost."""
    out = []
    for c in compiles:
        replayed, cost = replay_placement(
            aie_opt, design, c["seed"], _EFFORTS[c["effort_label"]]
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


def streaming_us(one: dict, many: dict) -> float:
    """Return what one more image in a launch costs, from runs at batch 1 and N."""
    return round((many["launch_us"] - one["launch_us"]) / (many["batch"] - 1), 2)


def aggregate(
    runs: list[dict], compiles: list[dict], placements: list[dict]
) -> list[dict]:
    failed_seeds = {r["seed"] for r in runs + compiles if not r["passed"]}
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
    for c in compiles:
        case = f"mobilenet/seed={c['seed']}/effort={c['effort_label']}"
        for metric, unit in (
            ("compile_s", "s"),
            ("compile_peak_rss_mb", "MB"),
            ("compile_s_warm", "s"),
            ("compile_peak_rss_mb_warm", "MB"),
            *((m, "s") for m in STAGE_METRICS),
            *((f"{m}_warm", "s") for m in STAGE_METRICS),
        ):
            if c.get(metric) is not None:
                out.append(
                    {"name": f"{case}/{metric}", "unit": unit, "value": c[metric]}
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
        f"{'streaming':>9}  {'e2e/image':>9}"
    )
    for r in runs:
        case = (
            f"mobilenet/seed={r['seed']}/batch={r['batch']}"
            f"/effort={r['effort_label']}"
        )
        print(
            f"{r['seed']:>4} {r['batch']:>5} {r['effort_label']:>7}  "
            f"{'yes' if r['passed'] else 'NO':>4}  "
            f"{cell(case, 'us_per_image'):>9}  {cell(case, 'streaming_us'):>9}  "
            f"{cell(case, 'e2e_us_per_image'):>9}"
        )
        if not r["passed"] and r["launch_us"] is None:
            print("no 'NPU time (avg/min/max us)' line in this batch's output")
    for c in compiles:
        if not c["passed"] or any(
            not r["passed"]
            for r in runs
            if (r["seed"], r["effort_label"]) == (c["seed"], c["effort_label"])
        ):
            print(c["log"])
    for p in placements:
        print(
            f"seed {p['seed']} effort {p['effort_label']}: "
            f"placement {p['placement']} cost {p['placement_cost']}"
        )
        if not p["replay_matches"]:
            print("aie-opt placed the design differently from aiecc; no cost recorded")


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
        default=[1, 2, 4, 8, 16, 32, 64],
        help="images per launch to run each seed at (default: every power of "
        "two from 1 to 64; the streaming time needs 1)",
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
        "provenance": provenance(
            device=pre.device, pmode=pre.pmode, dispatch="dynamic"
        ),
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
        runs, compiles, placements = [], [], []
        for seed in args.seeds:
            mine = []
            for effort_label in _EFFORTS:
                compiled, seed_runs = run_seed(
                    seed, effort_label, batches, args.warmup, args.iters
                )
                mine.append(compiled)
                runs += seed_runs
            compiles += mine
            placements += place_seed(mine, args.aie_opt, design)
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
                "us_per_image",
                "e2e_us_per_image",
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
            "compiles": [{k: v for k, v in c.items() if k != "log"} for c in compiles],
            "placements": placements,
            "detail": detail,
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
