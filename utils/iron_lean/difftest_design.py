# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Differential test: the Lean evaluation of an exported symbolic design against the numpy oracle."""

import argparse
import subprocess
from collections import Counter
from pathlib import Path

from export import LEAN_ROOT, WHOLE_ARRAY, whole_array_program
from sim import Simulation


def compare(
    source: Path,
    lean_root: Path,
    name: str,
    shapes: list[str],
    m: int,
    k: int,
    n: int,
    n_aie_cols: int,
    device: str = "npu2",
) -> bool:
    """Evaluate Lean design `name` at each shape and compare it with the oracle on `source`.

    Returns:
        Whether every shape agrees line for line and the Lean sequence's guards hold.
    """
    lean_eval = lean_root / ".lake/build/bin/ironeval"
    passed = True
    for shape in shapes:
        program = whole_array_program(source, shape, m, k, n, n_aie_cols, device)
        want = Simulation(program).lines()
        got = subprocess.run(
            [lean_eval, "design", name, *shape.split("x")],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
        missing = Counter(want) - Counter(got[1:])
        extra = Counter(got[1:]) - Counter(want)
        ok = got[0] == "ok true" and not missing and not extra
        passed &= ok
        print(
            f"{'PASS' if ok else 'FAIL'} {name} {shape}: {len(want)} lines, {got[0]}, "
            f"{sum(missing.values())} missing, {sum(extra.values())} extra"
        )
        for line in list(missing)[:3]:
            print(f"  numpy only: {line[:160]}")
        for line in list(extra)[:3]:
            print(f"  lean only:  {line[:160]}")
    return passed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="exported Lean design")
    p.add_argument("--source", type=Path, default=WHOLE_ARRAY)
    p.add_argument("--lean-root", type=Path, default=LEAN_ROOT)
    p.add_argument("--device", choices=["npu1", "npu2"], default="npu2")
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=1)
    p.add_argument("--m", type=int, default=64)
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--shapes", nargs="+", required=True, metavar="MxKxN")
    opts = p.parse_args()

    ok = compare(
        opts.source,
        opts.lean_root,
        opts.name,
        opts.shapes,
        opts.m,
        opts.k,
        opts.n,
        opts.n_aie_cols,
        opts.device,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
