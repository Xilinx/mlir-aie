#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check the SA placer's lock model against aie-objectfifo-allocate.

For each module with logical tiles and each seed, places the module with the
SA placer, reads its sa-lock-groups, sa-lock-penalty and sa-lock-stopped
statistics, then runs split, packet ids and allocation on that placement. Both
give one verdict: the locks fit, they do not, or the packing search stopped
before it decided. The verdicts must be equal, and when allocation succeeds
its group count must equal the groups the SA counts (an AIE2 segment's
``__prodlock``, an AIE1 pool's ``__lock0``). Placements that fail, or whose
allocation fails for another reason, are skipped and counted. Exits 1 on any
mismatch.

With no inputs it checks every in-tree .mlir with a logical tile and the
mobilenet fixture.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, "..", ".."))
_GROUPS_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-lock-groups")
_PENALTY_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-lock-penalty")
_STOPPED_RE = re.compile(r"\(S\)\s+(\d+)\s+sa-lock-stopped")
# One group each: an AIE2 segment's producer lock, an AIE1 pool's first lock.
_GROUP_RE = re.compile(r"__(?:prodlock\d+|lock0) = aie\.lock")


def in_tree_inputs() -> list:
    """Return every in-tree .mlir naming a logical tile.

    Returns:
        Paths under test/, programming_examples/ and utils/, sorted.
    """
    found = []
    for top in ("test", "programming_examples", "utils"):
        for root, _, files in os.walk(os.path.join(_ROOT, top)):
            for name in files:
                path = os.path.join(root, name)
                if name.endswith(".mlir"):
                    with open(path, errors="replace") as f:
                        if "aie.logical_tile" in f.read():
                            found.append(path)
    return sorted(found)


def modules(path: str) -> list:
    """Split an input as --split-input-file does.

    Args:
        path: The input file.

    Returns:
        The text of each module that names a logical tile.
    """
    with open(path, errors="replace") as f:
        parts = re.split(r"^// -----.*$", f.read(), flags=re.M)
    return [part for part in parts if "aie.logical_tile" in part]


def check(
    aie_opt: str, text: str, seed: int, effort: float, timeout: float, scratch: str
):
    """Place one module and compare the SA's lock count with allocation's.

    Args:
        aie_opt: The aie-opt binary.
        text: The module.
        seed: The SA seed.
        effort: The SA effort.
        timeout: Seconds each step may take.
        scratch: A directory for the intermediate files.

    Returns:
        None when skipped, else (sa_groups, alloc_groups, sa_verdict,
        alloc_verdict, sa_penalty); alloc_groups is None when allocation
        failed, and each verdict is "fits", "fails" or "stopped".
    """
    source = os.path.join(scratch, "in.mlir")
    placed = os.path.join(scratch, "placed.mlir")
    with open(source, "w") as f:
        f.write(text)
    try:
        place = subprocess.run(
            [
                aie_opt,
                f"--aie-place-tiles=placer=sa_placer sa-seed={seed} sa-effort={effort}",
                "--mlir-pass-statistics",
                source,
                "-o",
                placed,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if place.returncode != 0:
            return None
        alloc = subprocess.run(
            [
                aie_opt,
                "--aie-objectfifo-split",
                "--aie-assign-packet-ids",
                "--aie-objectfifo-allocate",
                placed,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None
    if "could not decide whether the locks fit" in alloc.stderr:
        alloc_verdict = "stopped"
    elif "could not place locks" in alloc.stderr:
        alloc_verdict = "fails"
    elif alloc.returncode == 0:
        alloc_verdict = "fits"
    else:
        return None
    groups = sum(int(n) for n in _GROUPS_RE.findall(place.stderr))
    penalty = sum(int(n) for n in _PENALTY_RE.findall(place.stderr))
    if sum(int(n) for n in _STOPPED_RE.findall(place.stderr)):
        sa_verdict = "stopped"
    else:
        sa_verdict = "fails" if penalty else "fits"
    made = len(_GROUP_RE.findall(alloc.stdout)) if alloc_verdict == "fits" else None
    return groups, made, sa_verdict, alloc_verdict, penalty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--aie-opt", required=True, help="aie-opt binary")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="an SA seed, repeatable (default: 3, 11 and 99)",
    )
    parser.add_argument(
        "--sa-effort", type=float, default=1.0, help="SA effort (default 1.0)"
    )
    parser.add_argument(
        "--timeout", type=float, default=300, help="seconds per step (default 300)"
    )
    parser.add_argument("inputs", nargs="*", help="inputs (default: in-tree)")
    args = parser.parse_args()

    inputs = args.inputs or in_tree_inputs()
    seeds = args.seeds or [3, 11, 99]
    agree = mismatch = skipped = failing = stopped = 0
    with tempfile.TemporaryDirectory() as scratch:
        for path in inputs:
            for index, text in enumerate(modules(path)):
                for seed in seeds:
                    result = check(
                        args.aie_opt, text, seed, args.sa_effort, args.timeout, scratch
                    )
                    if result is None:
                        skipped += 1
                        continue
                    groups, made, sa_verdict, alloc_verdict, penalty = result
                    if sa_verdict == alloc_verdict and made in (None, groups):
                        agree += 1
                        failing += alloc_verdict == "fails"
                        stopped += alloc_verdict == "stopped"
                        continue
                    mismatch += 1
                    print(
                        f"MISMATCH {path} module {index} seed {seed}: SA "
                        f"{groups} groups (penalty {penalty}, {sa_verdict}), "
                        f"allocation {made} groups ({alloc_verdict})"
                    )
    print(
        f"agree={agree} mismatch={mismatch} skipped={skipped} "
        f"agreeing_lock_failures={failing} agreeing_lock_stops={stopped}"
    )
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
