#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The prefill's GEMM array: a layer's projections for M=64 prompt rows.

Eight cores, one per column, each own one shim weight lane and the same
eighth of every projection's output rows as the decode's GEMV core of that
lane, read from the decode's weight blob unchanged (aie2_decode.wblob_layout,
llama_decode.pack_lane_mma). A core runs a layer's projections in turn
(PREFILL_PROJS: qkv fused, o, g, u, d), each as C tiles of 64 rows by 64
output columns on aie::mmul (kernels/llama_gemm_mma.cc), K in chunks of
128, then the decode's epilogue on each tile:

- Weights: the lane's slots without their trailers, staged a tile (16
  slots: 8 row groups by both halves of 2048 columns of K) at a time in the
  core's column's memtile, which walks each into 16 chunks of the 8 groups'
  1 KB of K.
- Activations: a projection's rows, [row][K int8 | fp32 scale | pad], are
  read once per output tile, K chunk by K chunk, then a chunk whose rows
  start with their scales; one memtile tiles each chunk into mmul's A
  layout and broadcasts it to the eight cores. Ahead of them come the
  projection's weight slot trailers (biases and w_scales) for all eight
  cores, from a table `pack_trailers` copies out of the blob, each core
  keeping its own.
- Output: fp32, 8 rows by 64 columns at a time, four cores' joined in a
  memtile, written row-major (each core's columns in turn), one transfer
  per tile and half of the array.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import index_cast  # pyright: ignore[reportMissingImports]
from aie.extras.types import i32  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
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
A_CHUNK = M * KC
SLOT_W = 8 * 1024
TRAILER = MMA_SLOT - SLOT_W
TILE_SLOTS = 16
TILE_TRAILERS = 8 * TRAILER
OUT_ROWS = 8
A_TILE = 1024
A_PLACE = Tile(3, 1)
JOIN_PLACES = (Tile(1, 1), Tile(6, 1))
CORE_ROW = 2


class Projection:
    """One of a layer's projections on the GEMM array, its PROJECTIONS
    letters fused (consecutive in a lane and in the trailer table).

    Args:
        letters: The fused projections' letters, in lane order.
        table_at: Where its trailers start in a layer's trailer table.

    Attributes:
        name: The letters.
        rows: Its output rows (columns of the GEMM's C), letter by letter.
        k: Its K.
        k_chunks: K chunks of KC a tile.
        tiles: Each core's 64-column output tiles.
        at: Where it starts in a core's lane of a layer, in bytes.
        table_at: Where its trailers start in a layer's trailer table.
        trailer_bytes: Its trailers, [core][tile][group][TRAILER].
        trailer_chunks: A chunks they take.
        row_bytes: An activation row: K int8, its fp32 scale, padding.
    """

    def __init__(self, letters, table_at):
        self.name = letters
        self.rows = sum(PROJECTIONS[p][0] for p in letters)
        self.k = PROJECTIONS[letters[0]][1]
        if any(PROJECTIONS[p][1] != self.k for p in letters):
            raise ValueError(f"{letters} fuses projections of different K")
        self.k_chunks = self.k // KC
        self.tiles = self.rows // N_CORES // TILE_COLS
        self.at = lane_slots(LAYER[: LAYER.index(letters[0])]) * MMA_SLOT
        self.table_at = table_at
        self.trailer_bytes = N_CORES * self.tiles * TILE_TRAILERS
        self.trailer_chunks = -(-self.trailer_bytes // A_CHUNK)
        self.row_bytes = self.k + 16


def _projections():
    out, at = [], 0
    for letters in ("qkv", "o", "g", "u", "d"):
        p = Projection(letters, at)
        out.append(p)
        at += p.trailer_chunks * A_CHUNK
    return out, at


PREFILL_PROJS, TABLE_LAYER = _projections()


def act_layout(projs):
    """Each projection's activation rows' offset in a layer's `act` region.

    Returns:
        ({name: byte offset}, the region's bytes): a projection's M rows of
        row_bytes, then KC bytes of padding the scale chunk reads past the
        last row.
    """
    offsets, at = {}, 0
    for p in projs:
        offsets[p.name] = at
        at += M * p.row_bytes + KC
    return offsets, at


def pack_trailers(wblob, n_layers):
    """The trailer table: the weight blob's slot trailers the GEMM array's
    epilogue reads, copied out so that a projection's are one transfer.

    Args:
        wblob: The decode's weight blob (int8), llama_decode.DecodeShape's.
        n_layers: Its layers.

    Returns:
        int8 (n_layers, TABLE_LAYER): per layer and PREFILL_PROJS
        projection, per core, output tile and row group, the group's first
        slot's trailer (its 8 biases and w_scales), each A_CHUNK laid out so
        that the A memtile's walk to mmul's A layout restores this order.
    """
    table = np.zeros((n_layers, TABLE_LAYER), np.int8)
    for p in PREFILL_PROJS:
        c, t, g = np.meshgrid(
            np.arange(N_CORES), np.arange(p.tiles), np.arange(8), indexing="ij"
        )
        slots = p.at + 2 * (t * 8 + g).ravel() * MMA_SLOT + SLOT_W
        for L in range(n_layers):
            src = (L * N_CORES + c.ravel()) * LANE_BYTES + slots
            y = np.zeros(p.trailer_chunks * A_CHUNK, np.int8)
            y[: p.trailer_bytes] = wblob[src[:, None] + np.arange(TRAILER)].ravel()
            x = y.reshape(-1, 8, 16, 8, 8).transpose(0, 1, 3, 2, 4)
            table[L, p.table_at : p.table_at + y.size] = x.ravel()
    return table


class GemmArray:
    """The eight GEMM cores and their dataflow, for a design's Program.

    Args:
        projs: The projections each layer runs, in order.
        n_layers: Layers a dispatch runs.

    Attributes:
        workers: The cores' Workers.
        fifos: The shim ends the runtime sequence takes, in order: the A
            stream, the eight weight lanes, then the two output joins.
    """

    def __init__(self, projs, n_layers):
        self.projs = projs
        t_a = np.ndarray[(A_CHUNK // 4,), np.dtype[np.int32]]
        t_w = np.ndarray[(SLOT_W // 4,), np.dtype[np.int32]]
        t_c = np.ndarray[(M * TILE_COLS,), np.dtype[np.int32]]
        t_sc = np.ndarray[(M,), np.dtype[np.float32]]
        t_out = np.ndarray[(OUT_ROWS * TILE_COLS,), np.dtype[np.float32]]
        tr_words = max(p.tiles for p in projs) * TILE_TRAILERS // 4
        t_tr = np.ndarray[(tr_words,), np.dtype[np.int32]]

        # A chunk in words: 8x8 tile (row block mb, kt) at mb*1024 + kt*64.
        self.of_a = ObjectFifo(t_a, depth=2, name="a")
        of_ac = self.of_a.cons().forward(
            tile=A_PLACE,
            obj_type=t_a,
            depth=2,
            name="ac",
            to_stream=TensorAccessPattern(
                tensor_dims=[A_CHUNK // 4],
                offset=0,
                sizes=[M // 8, KC // 8, 8, 2],
                strides=[A_TILE // 4, 2, KC // 4, 1],
            ),
        )
        # A tile's slots, group g's half h at (2g + h) * SLOT_W, walked per
        # K half and 1 KB of it, each group's 1 KB.
        tile_words = TILE_SLOTS * SLOT_W // 4
        self.of_wm = [
            ObjectFifo(
                np.ndarray[(tile_words,), np.dtype[np.int32]],
                depth=2,
                name=f"wm{c}",
            )
            for c in range(N_CORES)
        ]
        of_w = [
            self.of_wm[c]
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
        out_words = OUT_ROWS * TILE_COLS
        self.of_join = [
            ObjectFifo(
                np.ndarray[(4 * out_words,), np.dtype[np.float32]],
                depth=2,
                name=f"join{j}",
            )
            for j in range(2)
        ]
        of_out = [
            f
            for j, join in enumerate(self.of_join)
            for f in join.prod().join(
                [i * out_words for i in range(4)],
                obj_types=[t_out] * 4,
                names=[f"out{4 * j + i}" for i in range(4)],
                tile=JOIN_PLACES[j],
            )
        ]

        km = llama_kernels.gemm_mma()
        k_chunk = km.bind("llama_gemm_chunk", [t_a, t_w, t_c, np.int32])
        k_tr = km.bind("llama_gemm_take_trailers", [t_a, t_tr] + [np.int32] * 3)
        k_sc = km.bind("llama_gemm_act_scales", [t_a, t_sc])
        k_out = km.bind("llama_gemm_out", [t_c, t_sc, t_tr, t_out] + [np.int32] * 2)

        def core_body(c_a, c_w, c_out, acc, sc, tr, core, kc, kt, ks, ko):
            for _ in range_(n_layers):
                for p in projs:
                    for i in range_(p.trailer_chunks):
                        a = c_a.acquire(1)
                        kt(a, tr, index_cast(i32(), i), core, p.tiles)
                        c_a.release(1)
                    for t in range_(p.tiles):
                        a = c_a.acquire(1)
                        w = c_w.acquire(1)
                        kc(a, w, acc, 1)
                        c_a.release(1)
                        c_w.release(1)
                        for _ in range_(p.k_chunks - 1):
                            a = c_a.acquire(1)
                            w = c_w.acquire(1)
                            kc(a, w, acc, 0)
                            c_a.release(1)
                            c_w.release(1)
                        a = c_a.acquire(1)
                        ks(a, sc)
                        c_a.release(1)
                        # The tile's 8 groups are block t of the core's
                        # trailers.
                        for part in range_(M // OUT_ROWS):
                            o = c_out.acquire(1)
                            ko(acc, sc, tr, o, index_cast(i32(), part), t)
                            c_out.release(1)

        self.workers = [
            Worker(
                core_body,
                [
                    of_ac.cons(),
                    of_w[c].cons(),
                    of_out[c].prod(),
                    Buffer(t_c, name=f"acc{c}"),
                    Buffer(t_sc, name=f"act_sc{c}"),
                    Buffer(t_tr, name=f"trailers{c}"),
                    c,
                    k_chunk,
                    k_tr,
                    k_sc,
                    k_out,
                ],
                tile=Tile(c, CORE_ROW),
            )
            for c in range(N_CORES)
        ]
        self.fifos = [self.of_a.prod(), *[f.prod() for f in self.of_wm]] + [
            f.cons() for f in self.of_join
        ]

    @staticmethod
    def out_words(p):
        """A projection's output, fp32 words."""
        return p.tiles * N_CORES * TILE_COLS * M

    def fill(self, fifos, p, w, table, act):
        """Issue projection p's fills in a runtime sequence.

        Args:
            fifos: The sequence's handles for `fifos`.
            p: The projection.
            w: (the weight blob, int32 words; its words; p's layer there).
            table: (the trailer table, int32 words; its words; p's layer).
            act: (the activations, int32 words; its words; byte offset of
                p's rows).

        Returns:
            The fills' TaskGroup.
        """
        a_in, w_in = fifos[0], fifos[1 : 1 + N_CORES]
        fills = TaskGroup()
        w_buf, w_words, w_layer = w
        k_parts = p.k // D
        for c in range(N_CORES):
            lane = ((w_layer * N_CORES + c) * LANE_BYTES + p.at) // 4
            # K = D: the lane's run of tiles; else per tile its K parts,
            # each part's slots p.tiles tiles apart.
            for t in range(p.tiles if k_parts > 1 else 1):
                w_in[c].fill(
                    w_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[w_words],
                        offset=lane + t * TILE_SLOTS * MMA_SLOT // 4,
                        sizes=[k_parts, (p.tiles if k_parts == 1 else 1) * TILE_SLOTS]
                        + [SLOT_W // 512, 128],
                        strides=[p.tiles * TILE_SLOTS * MMA_SLOT // 4, MMA_SLOT // 4]
                        + [128, 1],
                    ),
                    group=fills,
                    wait=True,
                )
        t_buf, t_words, t_layer = table
        a_in.fill(
            t_buf,
            tap=TensorAccessPattern(
                tensor_dims=[t_words],
                offset=(t_layer * TABLE_LAYER + p.table_at) // 4,
                sizes=[1, 1, p.trailer_chunks * A_CHUNK // 4 // 512, 512],
                strides=[0, 0, 512, 1],
            ),
            group=fills,
            wait=True,
        )
        act_buf, act_words, act_at = act
        a_in.fill(
            act_buf,
            tap=TensorAccessPattern(
                tensor_dims=[act_words],
                offset=act_at // 4,
                sizes=[p.tiles, p.k_chunks + 1, M, KC // 4],
                strides=[0, KC // 4, p.row_bytes // 4, 1],
            ),
            group=fills,
            wait=True,
        )
        return fills

    def drain(self, fifos, p, out, tiles):
        """Issue the drains of output tiles `tiles` of projection p.

        Args:
            fifos: The sequence's handles for `fifos`.
            p: The projection.
            out: (the outputs, fp32 or int32 words; its words; word offset
                of p's row 0; words between rows). A row holds each core's
                columns in turn, its tiles' 64 at a time.
            tiles: The tiles.

        Returns:
            The drains' TaskGroup.
        """
        joins = fifos[1 + N_CORES :]
        out_buf, out_words, out_at, row_words = out
        rc = p.tiles * TILE_COLS
        drains = TaskGroup()
        for t in tiles:
            for j in range(2):
                joins[j].drain(
                    out_buf,
                    tap=TensorAccessPattern(
                        tensor_dims=[out_words],
                        offset=out_at + j * 4 * rc + t * TILE_COLS,
                        sizes=[M // OUT_ROWS, 4, OUT_ROWS, TILE_COLS],
                        strides=[OUT_ROWS * row_words, rc, row_words, 1],
                    ),
                    group=drains,
                    wait=True,
                )
        return drains

    def run(self, fifos, projs, fill_args, outs, batch=4):
        """Issue a run of projections' transfers, each one's fills before
        the previous one's last drains, so the weights keep streaming.

        Args:
            fifos: The sequence's handles for `fifos`.
            projs: The projections, in the cores' order.
            fill_args: Per projection, fill's (w, table, act).
            outs: Per projection, drain's out.
            batch: Tiles a group of drains takes (a shim tile has 16 BDs).
        """
        fills = [self.fill(fifos, projs[0], *fill_args[0])]
        for k, p in enumerate(projs):
            starts = range(0, p.tiles, batch)
            for b in starts:
                if b == starts[-1] and k + 1 < len(projs):
                    fills.append(self.fill(fifos, projs[k + 1], *fill_args[k + 1]))
                tiles = range(b, min(b + batch, p.tiles))
                self.drain(fifos, p, outs[k], tiles).finish()
            fills[k].finish()


@jit
def build_gemms(
    *,
    N_LAYERS: CompileTime[int] = 1,
    W_LAYERS: CompileTime[int] = MODEL_LAYERS,
    T: CompileTime[int] = 128,
    PROJS: CompileTime[str] = "qkv o g u d",
):
    """A layer's projections on M rows, for each of the first N_LAYERS
    layers: each projection's rows from `act`, its outputs to `out`.

    Args:
        PROJS: The PREFILL_PROJS a layer runs, by name, in order.
        N_LAYERS: Layers run per dispatch.
        W_LAYERS: Layers of the weight blob (aie2_decode's N_LAYERS).
        T: The weight blob's RoPE table rows (aie2_decode's T).
    """
    projs = [p for name in PROJS.split() for p in PREFILL_PROJS if p.name == name]
    array = GemmArray(projs, N_LAYERS)
    _, w_total = wblob_layout(W_LAYERS, T)
    act_at, act_bytes = act_layout(projs)
    out_layer = sum(GemmArray.out_words(p) for p in projs)

    act_words = N_LAYERS * act_bytes // 4
    out_words = N_LAYERS * out_layer

    def sequence(wblob, table, act, out, *fifos):
        fill_args, outs = [], []
        for L in range(N_LAYERS):
            at = L * out_layer
            for p in projs:
                fill_args.append(
                    (
                        (wblob, w_total // 4, L),
                        (table, W_LAYERS * TABLE_LAYER // 4, L),
                        (act, act_words, L * act_bytes + act_at[p.name]),
                    )
                )
                outs.append((out, out_words, at, p.tiles * N_CORES * TILE_COLS))
                at += GemmArray.out_words(p)
        array.run(fifos, projs * N_LAYERS, fill_args, outs)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(w_total // 4,), np.dtype[np.int32]],
            np.ndarray[(W_LAYERS * TABLE_LAYER // 4,), np.dtype[np.int32]],
            np.ndarray[(act_words,), np.dtype[np.int32]],
            np.ndarray[(out_words,), np.dtype[np.float32]],
            *array.fifos,
        ],
    )
    return Program(NPU2(), rt, workers=array.workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B prefill GEMM array")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    p.add_argument(
        "--n-layers", type=int, choices=range(1, MODEL_LAYERS + 1), default=1
    )
    p.add_argument("--t", type=int, default=128)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build_gemms,
        opts,
        compile_kwargs=lambda o: dict(N_LAYERS=o.n_layers, T=o.t),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
