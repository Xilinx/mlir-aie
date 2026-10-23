#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The prefill's GEMM array (aie2_prefill.py) on the NPU, against the oracle.

Runs one projection of 64 rows for each of the first n layers, on the
decode's weight blob of real (or seeded random) layers and a seeded random
int8 activation block per layer, and checks every output row bit for bit
against llama_oracle.Layer.project. Then times the dispatch: the weights it
streams over its time, and its total DDR traffic.

The oracle runs before the NPU lock is taken; `--lock` serializes the device
part with other users of the NPU.
"""

import argparse
import fcntl
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie2_gemv_array import N_CORES
from aie2_prefill import (
    A_BYTES,
    ACT_BYTES,
    K_CHUNKS,
    KC,
    TABLE_PROJS,
    TILE_COLS,
    TILE_TRAILERS,
    M,
    Projection,
    build,
    pack_trailers,
)
from llama_config import D
from llama_decode import DecodeShape
from llama_npu import DispatchSession
from llama_oracle import Layer
from ml_dtypes import bfloat16


def act_block(rng, w_i8, edge_rows):
    """A seeded random activation block and its per-row scales.

    Args:
        rng: The numpy Generator.
        w_i8: The projection's int8 weights (rows, D).
        edge_rows: Whether two rows take the epilogue's scalar path.

    Returns:
        (int8 (M, D), fp32 (M,)): uniform int8 rows with scales like
        dyn_scale's; with edge_rows, row 0 is 127 times the signs of the
        weight row with the largest absolute sum (its sum there is the
        largest the weights allow), and row 1's scale is below the vector
        form's exponent range.
    """
    x = rng.integers(-128, 128, (M, D), dtype=np.int8)
    scales = (10.0 ** rng.uniform(-3.0, 0.5, M)).astype(np.float32)
    if edge_rows:
        heaviest = np.abs(w_i8.astype(np.int32)).sum(axis=1).argmax()
        x[0] = np.where(w_i8[heaviest] < 0, -127, 127)
        scales[1] = np.float32(1e-20)
    return x, scales


def act_chunks(x, scales):
    """A block in the design's `act` layout: [chunk][row][KC bytes], K's
    chunks, then the rows' scales."""
    out = np.zeros(ACT_BYTES, np.int8)
    out[:A_BYTES] = x.reshape(M, K_CHUNKS, KC).transpose(1, 0, 2).ravel()
    out[A_BYTES:] = scales.view(np.int8)
    return out


def rows_of(out, p):
    """The design's `out` ([layer][core][tile][row][64 columns]) as
    [layer][row][column]."""
    n = out.shape[0]
    tiles = out.reshape(n, N_CORES, p.tiles, M, TILE_COLS)
    return np.ascontiguousarray(tiles.transpose(0, 3, 1, 2, 4)).reshape(n, M, p.rows)


def main():
    ap = argparse.ArgumentParser(prog="The prefill GEMM array against the oracle")
    ap.add_argument("--proj", choices=TABLE_PROJS, default="q")
    ap.add_argument("--n-layers", type=int, choices=range(1, 17), default=1)
    ap.add_argument("--t", type=int, default=128, help="the blob's RoPE rows")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--synthetic-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20, help="timed dispatches; 0: none")
    ap.add_argument("--epi-cores", type=int, choices=(1, 2), default=2)
    ap.add_argument("--lock", type=Path, help="flock this file around NPU work")
    opts = ap.parse_args()

    rng = np.random.default_rng(opts.seed)
    n = opts.n_layers
    if opts.synthetic_weights:
        layers = [Layer.random(rng, opts.t) for _ in range(n)]
    else:
        layers = [Layer.real(i, opts.data, opts.t) for i in range(n)]
    shape = DecodeShape(n, opts.t)
    wblob = shape.pack_wblob(layers, np.zeros((opts.t, 128), bfloat16))
    # Layer 0's block has rows the epilogue takes its scalar path for; the
    # other layers' are typical, as the timing should see.
    blocks = [
        act_block(rng, layer.weights[opts.proj][0], L == 0)
        for L, layer in enumerate(layers)
    ]
    act = np.stack([act_chunks(x, s) for x, s in blocks])

    t0 = time.perf_counter()
    want = np.stack(
        [
            np.stack([layer.project(opts.proj, x[r], s[r]) for r in range(M)])
            for layer, (x, s) in zip(layers, blocks)
        ]
    )
    print(f"oracle: {n} layers x {M} rows in {time.perf_counter() - t0:.1f} s")
    w0 = layers[0].weights[opts.proj][0].astype(np.int64)
    wide = int((np.abs(w0 @ blocks[0][0][0]) >= 1 << 23).sum())
    print(f"layer 0 row 0's sums at or past 2^23 (the epilogue's scalar path): {wide}")

    p = Projection(opts.proj)
    kwargs = dict(
        PROJ=opts.proj, N_LAYERS=n, W_LAYERS=n, T=opts.t, EPI_CORES=opts.epi_cores
    )
    build.specialize(full_elf=True, **kwargs).compile()

    lock = open(opts.lock, "w") if opts.lock else None
    if lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        session = DispatchSession(build, **kwargs)
        w_t = iron.tensor(wblob.view(np.int32), dtype=np.int32)
        tr_t = iron.tensor(pack_trailers(wblob, n).view(np.int32), dtype=np.int32)
        a_t = iron.tensor(act.view(np.int32), dtype=np.int32)
        out_t = iron.tensor(
            np.full((n, N_CORES, p.tiles * M * TILE_COLS), np.nan, np.float32),
            dtype=np.float32,
        )
        tensors = [w_t, tr_t, a_t, out_t]
        first_ms = session(tensors)
        out_t.to("cpu")
        got = rows_of(out_t.numpy(), p)
        times, again = [], got
        if opts.iters:
            for _ in range(opts.warmup):
                session(tensors)
            times = [session(tensors) for _ in range(opts.iters)]
            out_t.to("cpu")
            again = rows_of(out_t.numpy(), p)
    finally:
        if lock:
            fcntl.flock(lock, fcntl.LOCK_UN)

    print(f"first dispatch: {first_ms:.3f} ms")
    bad = got.view(np.uint32) != want.view(np.uint32)
    rows_bad = bad.any(axis=2)
    for L, r in zip(*np.nonzero(rows_bad)):
        cols = np.nonzero(bad[L, r])[0]
        print(
            f"layer {L} row {r}: {cols.size} columns differ, first {cols[0]}: "
            f"{got[L, r, cols[0]]!r} vs {want[L, r, cols[0]]!r}"
        )
    exact = not rows_bad.any()
    print(
        f"{opts.proj} x {n} layers, {M} rows each: "
        f"{rows_bad.size - rows_bad.sum()}/{rows_bad.size} rows bit-exact: "
        f"{'PASS' if exact else 'FAIL'}"
    )
    if times:
        repeat = np.array_equal(again.view(np.uint32), want.view(np.uint32))
        print(
            f"after timing, every row still bit-exact: {'PASS' if repeat else 'FAIL'}"
        )
        exact &= repeat
        w_bytes = n * p.rows * D
        ddr = w_bytes + n * (p.tiles * TILE_TRAILERS + ACT_BYTES + M * p.rows * 4)
        lo, med = min(times), float(np.median(times))
        print(
            f"dispatch: min {lo:.3f} ms, median {med:.3f} ms; weights {w_bytes / 1e6:.1f} MB"
            f" at {w_bytes / lo / 1e6:.1f} GB/s (min); DDR traffic {ddr / 1e6:.1f} MB"
            f" at {ddr / lo / 1e6:.1f} GB/s"
        )
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
