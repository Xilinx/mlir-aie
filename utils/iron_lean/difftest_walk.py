# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Differential test: Lean `Walk.idx` against taplib's own walk (`gather(arange)`)."""

import argparse
import random
import subprocess
from pathlib import Path

import numpy as np
from aie.helpers.taplib import TensorAccessPattern

LEAN_EVAL = Path(__file__).parent / "IronVerify/.lake/build/bin/ironeval"


def random_tap(rng: random.Random) -> TensorAccessPattern:
    rank = rng.choice([1, 2, 3])
    dims = [rng.choice([1, 2, 3, 4, 6, 8, 12]) for _ in range(rank)]
    tap = TensorAccessPattern.full(dims)
    for _ in range(rng.randint(1, 4)):
        op = rng.choice(["tile", "repeat", "permute", "slice", "split", "partition"])
        sizes = tap.sizes
        if op == "tile":
            tiles = [
                rng.choice([d for d in range(1, s + 1) if s % d == 0]) for s in sizes
            ]
            tap = tap.tile(tiles)
        elif op == "repeat":
            tap = tap.repeat(rng.choice([1, 2, 3]))
        elif op == "permute":
            axes = list(range(len(sizes)))
            rng.shuffle(axes)
            tap = tap.permute(axes)
        elif op == "slice":
            start = rng.randrange(sizes[0])
            if len(sizes) > 1 and rng.random() < 0.5:
                tap = tap[start]
            else:
                tap = tap[start :: rng.choice([1, 2])]
        elif op == "split":
            dim = rng.randrange(len(sizes))
            divisors = [d for d in range(1, sizes[dim] + 1) if sizes[dim] % d == 0]
            tap = tap.split(dim, rng.choice(divisors))
        else:
            dim = rng.randrange(len(sizes))
            divisors = [d for d in range(1, sizes[dim] + 1) if sizes[dim] % d == 0]
            tap = tap.partition(rng.choice(divisors), dim)
        if len(tap.sizes) > 8:
            break
    return tap


def whole_array_taps(M: int, K: int, N: int, m: int, k: int, n: int, cols: int):
    n_A_tiles_per_shim = max(1, 4 // cols)
    A = (
        TensorAccessPattern.full((M, K))
        .tile((m * n_A_tiles_per_shim, k))
        .repeat(N // n // cols)
        .permute((1, 0, 2, 3, 4))
    )
    B = TensorAccessPattern.full((K, N)).tile((k, n)).permute((1, 0, 2, 3))
    C = TensorAccessPattern.full((M, N)).tile((m * 4, n))
    taps = []
    for row in range(M // m // 4):
        for col in range(cols):
            taps.append(C[row, col::cols])
            if col < min(4, cols):
                taps.append(A[row * min(4, cols) + col])
            taps.append(B[col::cols])
    return taps


def lean_line(tap: TensorAccessPattern) -> str:
    fields = [int(tap.offset)]
    for size, stride in tap.transformation_dims:
        fields += [int(size), int(stride)]
    return " ".join(str(f) for f in fields)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=2000)
    opts = p.parse_args()

    rng = random.Random(opts.seed)
    taps = [random_tap(rng) for _ in range(opts.count)]
    taps += whole_array_taps(512, 256, 256, 64, 64, 32, 1)
    taps += whole_array_taps(512, 256, 256, 64, 64, 32, 4)
    taps += whole_array_taps(256, 128, 256, 32, 32, 16, 2)

    lines = "\n".join(lean_line(t) for t in taps) + "\n"
    out = subprocess.run(
        [LEAN_EVAL, "walks"], input=lines, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    assert len(out) == len(taps), f"{len(out)} Lean results for {len(taps)} walks"

    bad = 0
    for tap, got in zip(taps, out):
        want = tap.gather(np.arange(int(np.prod(tap.tensor_dims)))).reshape(-1)
        if [int(x) for x in got.split()] != want.tolist():
            bad += 1
            print(f"MISMATCH {tap}: lean={got[:80]} taplib={want[:16]}")
    print(f"{len(taps) - bad}/{len(taps)} walks agree")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
