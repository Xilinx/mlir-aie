#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The prefill's GEMM array: a projection of M=64 prompt rows per weight pass.

Eight GEMM cores, one per column, each own one shim weight lane and the same
eighth of the projection's output rows as the decode's GEMV core of that
lane, read from the decode's weight blob unchanged (aie2_decode.wblob_layout,
llama_decode.pack_lane_mma). A GEMM core computes C tiles of 64 rows by 64
output columns on aie::mmul (kernels/llama_gemm_mma.cc), K in chunks of 128,
and hands each tile through shared memory to an epilogue core beside it
(`EPI_CORES` of them):

- Weights: the lane's slots without their trailers, staged a tile (16
  slots: 8 row groups by both halves of K) at a time in the core's column's
  memtile, which walks each into 16 chunks of the 8 groups' 1 KB of K.
- Activations: `act` holds a projection's rows in A chunks, [chunk][row]
  [128 bytes], then the rows' fp32 scales. One memtile stages the chunks and
  broadcasts them to the eight GEMM cores in mmul's A layout, once per
  output tile; the scales go to the epilogue cores.
- Epilogue: the decode's (half_epilogue), OUT_ROWS rows at a time, on the
  slot trailers (biases and w_scales) of the tile's row groups, which
  `pack_trailers` copies out of the blob so that a tile's, for all eight
  cores, are one 4 KB transfer broadcast to the epilogue cores. fp32 to
  `out`, each core's tiles in turn ([core][tile][row][64 columns]), in runs
  of whole tiles.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import index_cast  # pyright: ignore[reportMissingImports]
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie2_decode import LANE_BYTES, wblob_layout
from aie2_gemv_array import LAYER, MMA_SLOT, N_CORES, lane_slots
from llama_config import N_LAYERS as MODEL_LAYERS
from llama_config import PROJECTIONS, D

M = 64
TILE_COLS = 64
KC = 128
K_CHUNKS = D // KC
A_CHUNK = M * KC
A_BYTES = K_CHUNKS * A_CHUNK
ACT_BYTES = A_BYTES + M * 4
SLOT_W = 8 * 1024
TRAILER = MMA_SLOT - SLOT_W
TILE_SLOTS = 16
TILE_TRAILERS = N_CORES * 8 * TRAILER
OUT_ROWS = 8
A_TILE = 1024
A_PLACE = Tile(3, 1)
# The K = D projections, in a layer's lane order, whose trailers the
# trailer table holds.
TABLE_PROJS = [p for p in LAYER if PROJECTIONS[p][1] == D]
TABLE_LAYER = sum(PROJECTIONS[p][0] for p in TABLE_PROJS) // 8 * TRAILER


class Projection:
    """One K = D projection's shape on the GEMM array.

    Args:
        proj: The projection's letter in PROJECTIONS.

    Attributes:
        rows: Its output rows (columns of the GEMM's C).
        tiles: Each core's 64-column output tiles.
        at: Where it starts in a core's lane of a layer, in bytes.
        table_at: Where its trailers start in a layer's trailer table.
    """

    def __init__(self, proj):
        self.rows, k = PROJECTIONS[proj]
        if k != D:
            raise ValueError(f"projection {proj} has K={k}, not {D}")
        self.tiles = self.rows // N_CORES // TILE_COLS
        self.at = lane_slots(LAYER[: LAYER.index(proj)]) * MMA_SLOT
        before = TABLE_PROJS[: TABLE_PROJS.index(proj)]
        self.table_at = sum(PROJECTIONS[q][0] for q in before) // 8 * TRAILER


def pack_trailers(wblob, n_layers):
    """The trailer table: the weight blob's slot trailers the GEMM array's
    epilogue reads, copied out so that a tile's are one linear transfer.

    Args:
        wblob: The decode's weight blob (int8), llama_decode.DecodeShape's.
        n_layers: Its layers.

    Returns:
        int8 (n_layers, TABLE_LAYER): per layer and TABLE_PROJS projection,
        per output tile, core and row group, the group's first slot's
        trailer (its 8 biases and w_scales).
    """
    table = np.empty((n_layers, TABLE_LAYER), np.int8)
    for proj in TABLE_PROJS:
        p = Projection(proj)
        t, c, g = np.meshgrid(
            np.arange(p.tiles), np.arange(N_CORES), np.arange(8), indexing="ij"
        )
        slots = p.at + (t * TILE_SLOTS + 2 * g).ravel() * MMA_SLOT + SLOT_W
        for L in range(n_layers):
            src = (L * N_CORES + c.ravel()) * LANE_BYTES + slots
            at = p.table_at
            table[L, at : at + src.size * TRAILER] = wblob[
                src[:, None] + np.arange(TRAILER)
            ].ravel()
    return table


@jit
def build(
    *,
    PROJ: CompileTime[str] = "q",
    N_LAYERS: CompileTime[int] = 1,
    W_LAYERS: CompileTime[int] = MODEL_LAYERS,
    T: CompileTime[int] = 128,
    EPI_CORES: CompileTime[int] = 2,
):
    """Projection PROJ of M rows for each of the first N_LAYERS layers.

    Args:
        PROJ: The projection, one with K = D.
        N_LAYERS: Layers run per dispatch.
        W_LAYERS: Layers of the weight blob (aie2_decode's N_LAYERS).
        T: The weight blob's RoPE table rows (aie2_decode's T).
        EPI_CORES: Epilogue cores per GEMM core, 1 (the GEMM cores on row 2,
            the epilogue cores on row 3) or 2 (the GEMM cores on row 3,
            handing even tiles to row 2 and odd tiles to row 4).
    """
    p = Projection(PROJ)
    if EPI_CORES not in (1, 2) or p.tiles % EPI_CORES:
        raise ValueError(f"{EPI_CORES} epilogue cores for {p.tiles} tiles a core")
    gemm_row = 2 if EPI_CORES == 1 else 3
    epi_rows = [3] if EPI_CORES == 1 else [2, 4]
    epi_tiles = p.tiles // EPI_CORES
    _, _, w_total = wblob_layout(W_LAYERS, T)
    t_a = np.ndarray[(A_CHUNK // 4,), np.dtype[np.int32]]
    t_w = np.ndarray[(SLOT_W // 4,), np.dtype[np.int32]]
    t_tr = np.ndarray[(TILE_TRAILERS // 4,), np.dtype[np.int32]]
    t_out = np.ndarray[(OUT_ROWS * TILE_COLS,), np.dtype[np.float32]]
    t_c = np.ndarray[(M * TILE_COLS,), np.dtype[np.int32]]
    t_sc = np.ndarray[(M,), np.dtype[np.float32]]

    # In words: A chunk q's 8x8 tile (row block mb, kt) at mb*1024 + kt*64.
    a_words = A_BYTES // 4
    of_a = ObjectFifo(np.ndarray[(a_words,), np.dtype[np.int32]], depth=1, name="a")
    of_ac = of_a.cons().forward(
        tile=A_PLACE,
        obj_type=t_a,
        depth=2,
        name="ac",
        to_stream=TensorAccessPattern(
            tensor_dims=[a_words],
            offset=0,
            sizes=[A_BYTES // A_TILE, KC // 8, 8, 2],
            strides=[A_TILE // 4, 2, KC // 4, 1],
        ),
        repeat_count=p.tiles,
    )
    # A tile's slots, group g's half h at (2g + h) * SLOT_W, walked per K
    # half and 1 KB of it, each group's 1 KB.
    tile_words = TILE_SLOTS * SLOT_W // 4
    of_wm = [
        ObjectFifo(
            np.ndarray[(tile_words,), np.dtype[np.int32]], depth=2, name=f"wm{c}"
        )
        for c in range(N_CORES)
    ]
    of_w = [
        of_wm[c]
        .cons()
        .forward(
            tile=Tile(c, 1),
            obj_type=t_w,
            depth=2,
            name=f"w{c}",
            to_stream=TensorAccessPattern(
                tensor_dims=[tile_words],
                offset=0,
                sizes=[2, 8, 8, 256],
                strides=[SLOT_W // 4, 256, 2 * SLOT_W // 4, 1],
            ),
        )
        for c in range(N_CORES)
    ]
    of_sc = ObjectFifo(t_sc, depth=2, name="sc")
    of_tr = [ObjectFifo(t_tr, depth=2, name=f"tr{e}") for e in range(EPI_CORES)]
    # Each GEMM core hands its C tiles to its epilogue cores in turn, in
    # their memory.
    of_c = [
        [
            ObjectFifo(t_c, depth=3, name=f"c{c}_{e}", delegate_tile=Tile(c, row))
            for e, row in enumerate(epi_rows)
        ]
        for c in range(N_CORES)
    ]
    of_out = [
        [ObjectFifo(t_out, depth=2, name=f"out{c}_{e}") for e in range(EPI_CORES)]
        for c in range(N_CORES)
    ]

    km = llama_kernels.gemm_mma()
    k_chunk = km.bind("llama_gemm_chunk", [t_a, t_w, t_c, np.int32])
    k_out = km.bind("llama_gemm_out", [t_c, t_sc, t_tr, t_out, np.int32, np.int32])

    def gemm_core(c_a, c_w, kc, *c_cs):
        for _ in range_(N_LAYERS):
            for _ in range_(epi_tiles):
                for c_c in c_cs:
                    acc = c_c.acquire(1)
                    a = c_a.acquire(1)
                    w = c_w.acquire(1)
                    kc(a, w, acc, 1)
                    c_a.release(1)
                    c_w.release(1)
                    for _ in range_(K_CHUNKS - 1):
                        a = c_a.acquire(1)
                        w = c_w.acquire(1)
                        kc(a, w, acc, 0)
                        c_a.release(1)
                        c_w.release(1)
                    c_c.release(1)

    def epilogue_core(c_c, c_sc, c_tr, c_out, core, ko):
        for _ in range_(N_LAYERS):
            sc = c_sc.acquire(1)
            for _ in range_(epi_tiles):
                acc = c_c.acquire(1)
                tr = c_tr.acquire(1)
                for part in range_(M // OUT_ROWS):
                    o = c_out.acquire(1)
                    ko(acc, sc, tr, o, index_cast(i32(), part), core)
                    c_out.release(1)
                c_tr.release(1)
                c_c.release(1)
            c_sc.release(1)

    workers = [
        Worker(
            gemm_core,
            [of_ac.cons(), of_w[c].cons(), k_chunk] + [f.prod() for f in of_c[c]],
            tile=Tile(c, gemm_row),
        )
        for c in range(N_CORES)
    ] + [
        Worker(
            epilogue_core,
            [
                of_c[c][e].cons(),
                of_sc.cons(),
                of_tr[e].cons(),
                of_out[c][e].prod(),
                c,
                k_out,
            ],
            tile=Tile(c, row),
        )
        for c in range(N_CORES)
        for e, row in enumerate(epi_rows)
    ]

    w_words = w_total // 4
    out_words = p.tiles * M * TILE_COLS

    def sequence(wblob, table, act, out, a_in, sc_in, *fifos):
        tr_in = fifos[:EPI_CORES]
        w_in = fifos[EPI_CORES : EPI_CORES + N_CORES]
        outs = fifos[EPI_CORES + N_CORES :]

        def fills(L):
            tg = TaskGroup()
            for c in range(N_CORES):
                w_in[c].fill(
                    wblob,
                    tap=TensorAccessPattern(
                        tensor_dims=[w_words],
                        offset=((L * N_CORES + c) * LANE_BYTES + p.at) // 4,
                        sizes=[1, p.tiles * TILE_SLOTS, SLOT_W // 512, 128],
                        strides=[0, MMA_SLOT // 4, 128, 1],
                    ),
                    group=tg,
                    wait=True,
                )
            for e in range(EPI_CORES):
                tr_in[e].fill(
                    table,
                    tap=TensorAccessPattern(
                        tensor_dims=[W_LAYERS, TABLE_LAYER // 4],
                        offset=(L * TABLE_LAYER + p.table_at + e * TILE_TRAILERS) // 4,
                        sizes=[1, epi_tiles, TILE_TRAILERS // 4 // 512, 512],
                        strides=[0, EPI_CORES * TILE_TRAILERS // 4, 512, 1],
                    ),
                    group=tg,
                    wait=True,
                )
            a_in.fill(
                act,
                tap=TensorAccessPattern(
                    tensor_dims=[N_LAYERS, ACT_BYTES // 4],
                    offset=L * ACT_BYTES // 4,
                    sizes=[1, 1, a_words // 512, 512],
                    strides=[0, 0, 512, 1],
                ),
                group=tg,
                wait=True,
            )
            sc_in.fill(
                act,
                tap=TensorAccessPattern(
                    tensor_dims=[N_LAYERS, ACT_BYTES // 4],
                    offset=(L * ACT_BYTES + A_BYTES) // 4,
                    sizes=[1, 1, 1, M],
                    strides=[0, 0, 0, 1],
                ),
                group=tg,
                wait=True,
            )
            return tg

        def drains(L):
            tg = TaskGroup()
            c_words = M * TILE_COLS
            for c in range(N_CORES):
                for e in range(EPI_CORES):
                    outs[c * EPI_CORES + e].drain(
                        out,
                        tap=TensorAccessPattern(
                            tensor_dims=[N_LAYERS, N_CORES, out_words],
                            offset=(L * N_CORES + c) * out_words + e * c_words,
                            sizes=[1, epi_tiles, c_words // 512, 512],
                            strides=[0, EPI_CORES * c_words, 512, 1],
                        ),
                        group=tg,
                        wait=True,
                    )
            return tg

        # A layer's transfers go out a layer ahead; its A object lands as
        # soon as the previous layer is done with the memtile's.
        filled = [fills(0)]
        for L in range(N_LAYERS):
            drained = drains(L)
            if L + 1 < N_LAYERS:
                filled.append(fills(L + 1))
            drained.finish()
            filled[L].finish()

    rt = Runtime(
        sequence,
        [
            np.ndarray[(w_total // 4,), np.dtype[np.int32]],
            np.ndarray[(W_LAYERS, TABLE_LAYER // 4), np.dtype[np.int32]],
            np.ndarray[(N_LAYERS, ACT_BYTES // 4), np.dtype[np.int32]],
            np.ndarray[(N_LAYERS, N_CORES, out_words), np.dtype[np.float32]],
            of_a.prod(),
            of_sc.prod(),
            *[f.prod() for f in of_tr],
            *[f.prod() for f in of_wm],
            *[f.cons() for fs in of_out for f in fs],
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B prefill GEMM array")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    p.add_argument("--proj", choices=TABLE_PROJS, default="q")
    p.add_argument(
        "--n-layers", type=int, choices=range(1, MODEL_LAYERS + 1), default=1
    )
    p.add_argument("--t", type=int, default=128)
    p.add_argument("--epi-cores", type=int, choices=(1, 2), default=2)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build,
        opts,
        compile_kwargs=lambda o: dict(
            PROJ=o.proj, N_LAYERS=o.n_layers, T=o.t, EPI_CORES=o.epi_cores
        ),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
