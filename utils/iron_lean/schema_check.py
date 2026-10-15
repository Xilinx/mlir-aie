# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Check the parametric whole_array schema against the numpy oracle on the live design.

For each configuration (device, column count, dtypes, tile, column-major switches) and
shape, `ironeval schema` evaluates `Schema.WholeArray.schema p` and its provenance must
match, line for line, the oracle's simulation of whole_array specialized to that shape.
"""

import argparse
import itertools
import subprocess
import time
from collections import Counter

import aie.iron.kernels as kernels
from aie.iron import str_to_dtype

from export import LEAN_ROOT, WHOLE_ARRAY, whole_array_program
from sim import Simulation

ARCH = {"npu1": "aie2", "npu2": "aie2p"}
MAX_COLS = {"npu1": 4, "npu2": 8}
# (dtype_in, dtype_out, emulate_bf16): the combinations whole_array's kernel builds.
DTYPES = [
    ("i8", "i8", 0),
    ("i8", "i32", 0),
    ("i16", "i16", 0),
    ("i16", "i32", 0),
    ("bf16", "bf16", 0),
    ("bf16", "f32", 0),
    ("bf16", "bf16", 1),
]
TILES = [(64, 64, 32), (32, 32, 32), (16, 32, 16)]
ROWS = 4


def mac_dims(device: str, dtype_in: str, dtype_out: str, emulate: int):
    return kernels.mm.mac_dims(
        str_to_dtype(dtype_in),
        str_to_dtype(dtype_out),
        arch=ARCH[device],
        emulate_bf16_mmul_with_bfp16=bool(emulate),
    )


def check(
    device, cols, dtypes, tile, b_col_maj, c_col_maj, mkn_tiles, lean_eval
) -> tuple[bool, str]:
    """Compare schema and oracle at one configuration and shape; return (ok, summary)."""
    dtype_in, dtype_out, emulate = dtypes
    m, k, n = tile
    r, s, t = mac_dims(device, dtype_in, dtype_out, emulate)
    Mt, Kt, Nt = mkn_tiles
    M, K, N = ROWS * m * Mt, k * Kt, n * cols * Nt
    shape = f"{M}x{K}x{N}"
    program = whole_array_program(
        WHOLE_ARRAY,
        shape,
        m,
        k,
        n,
        cols,
        device,
        dtype_in,
        dtype_out,
        b_col_maj,
        c_col_maj,
        emulate,
    )
    want = Simulation(program).lines()
    args = [ROWS, cols, m, k, n, r, s, t, b_col_maj, c_col_maj]
    got = subprocess.run(
        [lean_eval, "schema", *map(str, args), shape],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    missing = Counter(want) - Counter(got[1:])
    extra = Counter(got[1:]) - Counter(want)
    ok = got[0] == "ok true" and not missing and not extra
    detail = (
        "" if ok else f" missing {sum(missing.values())} extra {sum(extra.values())}"
    )
    if not ok and missing:
        detail += f"\n  oracle only: {next(iter(missing))[:150]}"
    if not ok and extra:
        detail += f"\n  schema only: {next(iter(extra))[:150]}"
    return ok, f"{len(want)} lines, mac {r}x{s}x{t}{detail}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--devices", nargs="+", choices=sorted(ARCH), default=["npu2"])
    p.add_argument("--cols", nargs="+", type=int, choices=range(1, 9), default=None)
    p.add_argument(
        "--dtypes",
        nargs="+",
        choices=[f"{i}-{o}{'-emu' if e else ''}" for i, o, e in DTYPES],
        default=None,
    )
    p.add_argument("--tiles", nargs="+", choices=["x".join(map(str, t)) for t in TILES])
    p.add_argument(
        "--shapes",
        nargs="+",
        default=["1x1x1", "4x2x2"],
        metavar="MtxKtxNt",
        help="shapes in tiles: M = 4·m·Mt, K = k·Kt, N = n·cols·Nt",
    )
    p.add_argument(
        "--col-maj", nargs="+", choices=["00", "01", "10", "11"], default=None
    )
    opts = p.parse_args()

    lean_eval = LEAN_ROOT / ".lake/build/bin/ironeval"
    dtypes = [
        d
        for d in DTYPES
        if opts.dtypes is None
        or f"{d[0]}-{d[1]}{'-emu' if d[2] else ''}" in opts.dtypes
    ]
    tiles = [
        t for t in TILES if opts.tiles is None or "x".join(map(str, t)) in opts.tiles
    ]
    col_maj = [(0, 0), (0, 1), (1, 0), (1, 1)]
    if opts.col_maj is not None:
        col_maj = [(int(c[0]), int(c[1])) for c in opts.col_maj]
    shapes = [tuple(int(x) for x in s.split("x")) for s in opts.shapes]

    failed = skipped = passed = 0
    for device in opts.devices:
        cols_list = opts.cols or list(range(1, MAX_COLS[device] + 1))
        for cols, d, tile, (bc, cc), shape in itertools.product(
            cols_list, dtypes, tiles, col_maj, shapes
        ):
            if cols > MAX_COLS[device] or (d[2] and device != "npu2"):
                continue
            r, s, t = mac_dims(device, *d)
            m, k, n = tile
            label = f"{device} c{cols} {d[0]}-{d[1]}{'-emu' if d[2] else ''} {m}x{k}x{n} b{bc}c{cc} {'x'.join(map(str, shape))}"
            if m % r or k % s or n % t:
                skipped += 1
                continue
            start = time.time()
            try:
                ok, summary = check(device, cols, d, tile, bc, cc, shape, lean_eval)
            except (
                Exception
            ) as exc:  # noqa: BLE001 -- a generation failure is a FAIL row
                ok, summary = False, f"{type(exc).__name__}: {exc}"[:200]
            print(
                f"{'PASS' if ok else 'FAIL'} {label}: {summary} ({time.time() - start:.1f} s)",
                flush=True,
            )
            passed += ok
            failed += not ok
    print(f"{passed} passed, {failed} failed, {skipped} skipped (tile not divisible)")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
