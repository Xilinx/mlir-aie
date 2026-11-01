#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The prefill's GEMM array (aie2_prefill.py) on the NPU, against the oracle.

Runs a layer's projections (aie2_prefill.PREFILL_PROJS) for 64 rows, for
each of the first n layers, on the decode's weight blob of real (or seeded
random) layers and a seeded random int8 activation block per layer and
projection, and checks every output row bit for bit against
llama_oracle.Layer.project. Then times the dispatch: the weights it streams
over its time.

The oracle runs before the NPU lock is taken; `--lock` serializes the device
part with other users of the NPU.
"""

import argparse
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie2_gemv_array import N_CORES
from aie2_prefill import (
    PREFILL_PROJS,
    TILE_COLS,
    GemmArray,
    M,
    act_layout,
    build_gemms,
    pack_trailers,
)
from llama_config import PROJECTIONS
from llama_decode import DecodeShape
from llama_npu import DispatchSession
from llama_oracle import Layer
from ml_dtypes import bfloat16


def act_block(rng, k, w_i8, edge_rows):
    """A seeded random activation block and its per-row scales.

    Args:
        rng: The numpy Generator.
        k: The block's columns.
        w_i8: A projection's int8 weights (rows, k) the block meets.
        edge_rows: Whether two rows take the epilogue's scalar path.

    Returns:
        (int8 (M, k), fp32 (M,)): uniform int8 rows with scales like
        dyn_scale's; with edge_rows, row 0 is 127 times the signs of the
        weight row with the largest absolute sum (its sum there is the
        largest the weights allow), and row 1's scale is below the vector
        form's exponent range.
    """
    x = rng.integers(-128, 128, (M, k), dtype=np.int8)
    scales = (10.0 ** rng.uniform(-3.0, 0.5, M)).astype(np.float32)
    if edge_rows:
        heaviest = np.abs(w_i8.astype(np.int32)).sum(axis=1).argmax()
        x[0] = np.where(w_i8[heaviest] < 0, -127, 127)
        scales[1] = np.float32(1e-20)
    return x, scales


def pack_rows(x, scales, row_bytes):
    """Rows as the array reads them: [K int8 | fp32 scale | pad]."""
    out = np.zeros((M, row_bytes), np.int8)
    out[:, : x.shape[1]] = x
    out[:, x.shape[1] : x.shape[1] + 4] = scales.view(np.int8).reshape(M, 4)
    return out.ravel()


def rows_of(out, p):
    """A projection's output ([row][core][tile * 64 + column]) as its
    letters' [row][column] blocks."""
    per_core = out.reshape(M, N_CORES, p.tiles * TILE_COLS)
    blocks, at = {}, 0
    for letter in p.name:
        n = PROJECTIONS[letter][0] // N_CORES
        blocks[letter] = per_core[:, :, at : at + n].reshape(M, -1)
        at += n
    return blocks


def main():
    ap = argparse.ArgumentParser(prog="The prefill GEMM array against the oracle")
    ap.add_argument("--n-layers", type=int, choices=range(1, 17), default=1)
    ap.add_argument("--t", type=int, default=128, help="the blob's RoPE rows")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--synthetic-weights", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=20, help="timed dispatches; 0: none")
    ap.add_argument(
        "--projs",
        nargs="+",
        choices=[p.name for p in PREFILL_PROJS],
        default=[p.name for p in PREFILL_PROJS],
    )
    opts = ap.parse_args()

    rng = np.random.default_rng(opts.seed)
    n = opts.n_layers
    if opts.synthetic_weights:
        layers = [Layer.random(rng, opts.t) for _ in range(n)]
    else:
        layers = [Layer.real(i, opts.data, opts.t) for i in range(n)]
    wblob = DecodeShape(n, opts.t).pack_wblob(layers, np.zeros((opts.t, 128), bfloat16))
    projs = [p for name in opts.projs for p in PREFILL_PROJS if p.name == name]
    act_at, act_bytes = act_layout(projs)
    act = np.zeros((n, act_bytes), np.int8)
    blocks = {}
    for L, layer in enumerate(layers):
        for p in projs:
            # Layer 0's blocks have rows the epilogue takes its scalar path
            # for; the other layers' are typical, as the timing should see.
            x, s = act_block(rng, p.k, layer.weights[p.name[0]][0], L == 0)
            blocks[L, p.name] = x, s
            at = act_at[p.name]
            act[L, at : at + M * p.row_bytes] = pack_rows(x, s, p.row_bytes)

    t0 = time.perf_counter()
    want = {
        (L, letter): np.stack([layer.project(letter, x[r], s[r]) for r in range(M)])
        for L, layer in enumerate(layers)
        for p in projs
        for x, s in [blocks[L, p.name]]
        for letter in p.name
    }
    print(f"oracle: {n} layers x {M} rows in {time.perf_counter() - t0:.1f} s")

    kwargs = dict(N_LAYERS=n, W_LAYERS=n, T=opts.t, PROJS=" ".join(opts.projs))
    build_gemms.specialize(full_elf=True, **kwargs).compile()
    out_layer = sum(GemmArray.out_words(p) for p in projs)

    session = DispatchSession(build_gemms, **kwargs)
    out_t = iron.tensor(np.full(n * out_layer, np.nan, np.float32), dtype=np.float32)
    tensors = [
        iron.tensor(wblob.view(np.int32), dtype=np.int32),
        iron.tensor(pack_trailers(wblob, n).view(np.int32).ravel(), dtype=np.int32),
        iron.tensor(act.view(np.int32).ravel(), dtype=np.int32),
        out_t,
    ]
    first_ms = session(tensors)
    out_t.to("cpu")
    runs = [out_t.numpy().copy()]
    times = []
    if opts.iters:
        for _ in range(opts.warmup):
            session(tensors)
        times = [session(tensors) for _ in range(opts.iters)]
        out_t.to("cpu")
        runs.append(out_t.numpy().copy())

    print(f"first dispatch: {first_ms:.3f} ms")
    exact = True
    for i, run in enumerate(runs):
        at = 0
        for L in range(n):
            for p in projs:
                words = GemmArray.out_words(p)
                for letter, got in rows_of(run[at : at + words], p).items():
                    w = want[L, letter]
                    bad = (got.view(np.uint32) != w.view(np.uint32)).any(axis=1)
                    exact &= not bad.any()
                    if bad.any() and i == 0:
                        print(f"layer {L} {letter}: rows {np.nonzero(bad)[0]} differ")
                at += words
        print(
            f"{'after timing' if i else 'first dispatch'}: {n} layers x "
            f"{' '.join(opts.projs)}, {M} rows each, "
            f"every row bit-exact: {'PASS' if exact else 'FAIL'}"
        )
    if times:
        w_bytes = n * sum(
            PROJECTIONS[letter][0] * p.k for p in projs for letter in p.name
        )
        lo, med = min(times), float(np.median(times))
        print(
            f"dispatch: min {lo:.3f} ms, median {med:.3f} ms; weights "
            f"{w_bytes / 1e6:.1f} MB at {w_bytes / lo / 1e6:.1f} GB/s (min); "
            f"{lo / n:.3f} ms a layer"
        )
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
