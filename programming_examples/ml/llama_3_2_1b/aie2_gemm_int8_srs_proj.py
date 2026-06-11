"""Phase 1.6 dataflow stub: full 16-CT projection fan-out (2 rows x 8 cols).

Mirrors the decode-overlay `projection` placement from
`placement.DECODE_PLACEMENT` (rows 2-3, cols 0-7). Per-column structure
follows the 2-CT stub in `aie2_gemm_int8_srs_col.py`:

  - one shim col per AIE column reads its slice (soft broadcast for act
    via per-col tap on the same buffer; matches `whole_array_iron`'s
    consolidated-runtime-args style)
  - per-col memtile splits a combined weight buffer into 2 per-CT slices
  - per-col memtile joins 2 per-CT outputs into one combined per-col buffer

3 runtime buffers (act, weights_all, out_all). Per-col DMAs are
described via TensorAccessPattern offsets into the big buffers. All 16
CTs run the stub (act -> out passthrough). The test verifies that the
combined output equals act concatenated 16 times.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.helpers.taplib import TensorAccessPattern

N_COLS = 8
N_ROWS = 2  # rows 2 and 3 in DECODE_PLACEMENT


def build(M: int, K: int, N: int):
    w_blob_bytes = N * K + N * 4 + N * 4  # per-tile, matches stub kernel
    w_col_bytes = N_ROWS * w_blob_bytes  # per-col weight slice
    out_per_tile = M * N
    out_col_bytes = N_ROWS * out_per_tile

    total_w = N_COLS * w_col_bytes
    total_out = N_COLS * out_col_bytes

    act_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    w_blob_ty = np.ndarray[(w_blob_bytes,), np.dtype[np.int8]]
    w_col_ty = np.ndarray[(w_col_bytes,), np.dtype[np.int8]]
    out_per_tile_ty = np.ndarray[(out_per_tile,), np.dtype[np.int8]]
    out_col_ty = np.ndarray[(out_col_bytes,), np.dtype[np.int8]]

    # Runtime-arg-level types (consolidated big buffers).
    rt_act_ty = np.ndarray[(M * K,), np.dtype[np.int8]]
    rt_w_ty = np.ndarray[(total_w,), np.dtype[np.int8]]
    rt_out_ty = np.ndarray[(total_out,), np.dtype[np.int8]]

    kernel = Kernel(
        "llama_gemm_int8_srs_pt",
        "llama_gemm_int8_srs_pt.cc.o",
        [act_ty, w_blob_ty, out_per_tile_ty],
    )

    def core_fn(of_act, of_w, of_out, gemm):
        a = of_act.acquire(1)
        b = of_w.acquire(1)
        o = of_out.acquire(1)
        gemm(a, b, o)
        of_act.release(1)
        of_w.release(1)
        of_out.release(1)

    # Per-column fifos.
    act_fifos = []
    w_fifos = []
    w_splits = []
    out_fifos = []
    out_joins = []
    workers = []

    for c in range(N_COLS):
        of_act_c = ObjectFifo(act_ty, name=f"act_{c}")
        of_w_c = ObjectFifo(w_col_ty, name=f"w_{c}")
        of_out_c = ObjectFifo(out_col_ty, name=f"out_{c}")

        w_split = of_w_c.cons().split(
            offsets=[r * w_blob_bytes for r in range(N_ROWS)],
            obj_types=[w_blob_ty] * N_ROWS,
            names=[f"w_{c}_r{r}" for r in range(N_ROWS)],
        )
        o_join = of_out_c.prod().join(
            offsets=[r * out_per_tile for r in range(N_ROWS)],
            obj_types=[out_per_tile_ty] * N_ROWS,
            names=[f"out_{c}_r{r}" for r in range(N_ROWS)],
        )

        act_fifos.append(of_act_c)
        w_fifos.append(of_w_c)
        w_splits.append(w_split)
        out_fifos.append(of_out_c)
        out_joins.append(o_join)

        for r in range(N_ROWS):
            workers.append(
                Worker(
                    core_fn,
                    [
                        of_act_c.cons(),
                        w_split[r].cons(),
                        o_join[r].prod(),
                        kernel,
                    ],
                    # Pin to (col c, row 2+r) so the placer keeps each col's
                    # fifos on its own shim+memtile.
                    tile=Tile(c, 2 + r),
                )
            )

    rt = Runtime()
    with rt.sequence(rt_act_ty, rt_w_ty, rt_out_ty) as (a, w, o):
        rt.start(*workers)

        # Act: broadcast -- every col reads the entire buffer from offset 0.
        act_tap = TensorAccessPattern(
            tensor_dims=[M * K],
            offset=0,
            sizes=[1, 1, 1, M * K],
            strides=[0, 0, 0, 1],
        )
        for c in range(N_COLS):
            rt.fill(act_fifos[c].prod(), a, tap=act_tap)

        # Weights: each col reads its own contiguous slice of w.
        for c in range(N_COLS):
            wtap = TensorAccessPattern(
                tensor_dims=[total_w],
                offset=c * w_col_bytes,
                sizes=[1, 1, 1, w_col_bytes],
                strides=[0, 0, 0, 1],
            )
            rt.fill(w_fifos[c].prod(), w, tap=wtap)

        # Outputs: each col writes its own slice of o.
        for c in range(N_COLS):
            otap = TensorAccessPattern(
                tensor_dims=[total_out],
                offset=c * out_col_bytes,
                sizes=[1, 1, 1, out_col_bytes],
                strides=[0, 0, 0, 1],
            )
            rt.drain(out_fifos[c].cons(), o, tap=otap, wait=(c == N_COLS - 1))

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-M", type=int, default=8)
    p.add_argument("-K", type=int, default=64)
    p.add_argument("-N", type=int, default=64)
    args = p.parse_args(sys.argv[1:])
    print(build(args.M, args.K, args.N))


if __name__ == "__main__":
    main()
