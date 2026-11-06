#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check the SA placer's lock model against aie-objectfifo-allocate.

For each module with logical tiles and each seed, places the module with the
SA placer, reads its sa-lock-groups and sa-lock-penalty statistics, then runs
split, packet ids and allocation on that placement. The SA's group
count must equal the lock groups allocation makes (one ``__prodlock`` each),
and the SA must charge a lock penalty exactly when allocation cannot place the
locks. Placements that fail, or whose allocation fails for another reason,
are skipped and counted. Exits 1 on any mismatch.

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
_LOCK_RE = re.compile(r"__prodlock\d+ = aie\.lock")


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


def check(aie_opt: str, text: str, seed: int, timeout: float, scratch: str):
    """Place one module and compare the SA's lock count with allocation's.

    Args:
        aie_opt: The aie-opt binary.
        text: The module.
        seed: The SA seed.
        timeout: Seconds each step may take.
        scratch: A directory for the intermediate files.

    Returns:
        None when skipped, else (sa_groups, alloc_groups, sa_penalty,
        alloc_lock_failure).
    """
    source = os.path.join(scratch, "in.mlir")
    placed = os.path.join(scratch, "placed.mlir")
    with open(source, "w") as f:
        f.write(text)
    try:
        place = subprocess.run(
            [
                aie_opt,
                f"--aie-place-tiles=placer=sa_placer sa-seed={seed}",
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
    lock_failure = "could not place locks" in alloc.stderr
    if alloc.returncode != 0 and not lock_failure:
        return None
    groups = sum(int(n) for n in _GROUPS_RE.findall(place.stderr))
    penalty = sum(int(n) for n in _PENALTY_RE.findall(place.stderr))
    return groups, len(_LOCK_RE.findall(alloc.stdout)), penalty, lock_failure


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
        "--timeout", type=float, default=300, help="seconds per step (default 300)"
    )
    parser.add_argument("inputs", nargs="*", help="inputs (default: in-tree)")
    args = parser.parse_args()

    inputs = args.inputs or in_tree_inputs()
    seeds = args.seeds or [3, 11, 99]
    agree = mismatch = skipped = 0
    with tempfile.TemporaryDirectory() as scratch:
        for path in inputs:
            for index, text in enumerate(modules(path)):
                for seed in seeds:
                    result = check(args.aie_opt, text, seed, args.timeout, scratch)
                    if result is None:
                        skipped += 1
                        continue
                    groups, made, penalty, lock_failure = result
                    if groups == made and (penalty > 0) == lock_failure:
                        agree += 1
                        continue
                    mismatch += 1
                    print(
                        f"MISMATCH {path} module {index} seed {seed}: SA "
                        f"{groups} groups (penalty {penalty}), allocation "
                        f"{made} groups (lock failure {lock_failure})"
                    )
    print(f"agree={agree} mismatch={mismatch} skipped={skipped}")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
