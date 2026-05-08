#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Cascade bottleneck blocks (bn13-bn14) for MobileNet V3 IRON API rewrite.

bn13 and bn14 each use 5 compute tiles, 2 of which use cascade streams (put/get
pairs). Kernel names and signatures are derived from aie2_bottleneckC.py.

Architecture overview (cascade-split convolutions):
  Layer-1 PUT tile:  1x1-relu conv on first input half, puts onto cascade stream
  Layer-1 GET tile:  reads cascade, runs 1x1-relu on second half → full 960-ch row
  Layer-2 tile:      DW-3x3 depthwise producing two split output fifos
  Layer-3 PUT tile:  1x1-proj on first DW-split half, puts onto cascade stream
  Layer-3 GET tile:  reads cascade, runs 1x1-proj+skip on second split, writes output

Weight delivery:
  L1 and L3 split weights are streamed via ObjectFifos (Shim → MemTile → tiles).
  L2 DW weights are static Buffers (baked into the tile at compile time).

Tile placements (col, row):
  bn13: l1_put=Tile(4,5), l1_get=Tile(5,5), l2=Tile(5,4),
        l3_put=Tile(4,3), l3_get=Tile(5,3)
  bn14: l1_put=Tile(6,5), l1_get=Tile(7,5), l2=Tile(6,2),
        l3_put=Tile(4,2), l3_get=Tile(5,2)
"""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.dataflow.cascadeflow import CascadeFlow
from aie.iron.device import Tile, AnyMemTile
from aie.iron.controlflow import range_

# ---------------------------------------------------------------------------
# Dimensions (from aie2_bottleneckC.py)
# ---------------------------------------------------------------------------
_InW = 7
_InH = 7
_InC = 80  # input channels
_L1_OutC = 960  # expanded channels (L1 output)
_InputSplit = 2  # cascade splits: each cascade tile handles half the channels
_OutputSplit = 2
_OutputSplit2 = 2
_L1_SplitC = _L1_OutC // _InputSplit  # 480 channels per cascade tile
_OC8 = _L1_OutC // (8 * _OutputSplit)  # inner loop count for L1 kernel  = 60
_L3_OutC = 80  # final projection output channels
_OC8_out = _L3_OutC // (8 * _OutputSplit2)  # inner loop count for L3 kernel = 5

# ---------------------------------------------------------------------------
# Weight sizes (derived from aie2_bottleneckC.py type definitions)
# ---------------------------------------------------------------------------
# L1 split weight per tile:  (InC // InputSplit) * (L1_OutC // OutputSplit)
_l1_split_wts_sz = (_InC // _InputSplit) * (_L1_OutC // _OutputSplit)  # 40*480 = 19200
# L2 DW weight:     3 * 3 * L1_OutC (full depthwise filter)
_l2_wts_sz = 3 * 3 * _L1_OutC * 1  # 8640
# L3 split weight per tile (per put/get chunk):
#   (L1_OutC // InputSplit) * (L3_OutC // OutputSplit2) = 480 * 40 = 19200
# Matches placed-API: each MemTile put/get streams 19200-byte chunks to its compute tile.
_l3_split_wts_sz = (_L1_OutC // _InputSplit) * (_L3_OutC // _OutputSplit2)  # 19200

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
_ty_act_in = np.ndarray[(_InW, 1, _InC), np.dtype[np.int8]]
_ty_l1_out_split = np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]]
_ty_l1_out_full = np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]]
_ty_l1_split_wts = np.ndarray[(_l1_split_wts_sz,), np.dtype[np.int8]]
_ty_l2_wts = np.ndarray[(_l2_wts_sz,), np.dtype[np.int8]]
_ty_l3_split_wts = np.ndarray[(_l3_split_wts_sz,), np.dtype[np.int8]]
_ty_act_out = np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]]

# Full L1/L3 weight tensors (host → MemTile, then split into halves).
_l1_full_wts_sz = _InC * _L1_OutC          # 80*960  = 76800
_l3_full_wts_sz = _L1_OutC * _L3_OutC      # 960*80  = 76800
_ty_l1_full_wts = np.ndarray[(_l1_full_wts_sz,), np.dtype[np.int8]]
_ty_l3_full_wts = np.ndarray[(_l3_full_wts_sz,), np.dtype[np.int8]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_wts_arr(data_dir, filename):
    """Load int8 weights from a comma-separated text file; return None if missing."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return np.fromfile(path, sep=",", dtype=np.int8)
    return None


def _make_static_wts(data_dir, size, filename, name):
    """Create a static Buffer for weights (L2 DW, baked into tile at compile time)."""
    arr = _load_wts_arr(data_dir, filename)
    if arr is None:
        arr = np.zeros((size,), dtype=np.int8)
    return Buffer(
        np.ndarray[(size,), np.dtype[np.int8]],
        initial_value=arr,
        name=name,
    )


def _make_wts_fifo(wts_ty, depth, name):
    """Create a streaming ObjectFifo for split weight delivery (L1 and L3)."""
    return ObjectFifo(wts_ty, depth=depth, name=name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cascade_bottlenecks(
    act_in: ObjectFifo,
    sf: dict,
    *,
    data_dir: str,
) -> tuple:
    """Implement bn13 and bn14 cascade bottleneck blocks.

    Each block has 5 compute workers plus a skip-forwarder worker,
    with 2 cascade stream put/get pairs per block (4 cascade pairs total).

    Args:
        act_in:        Input ObjectFifo carrying (7,1,80) int8 activations, depth=2.
        *scale_factors: Eight scale factors in order:
                        bn13_s1, bn13_s2, bn13_s3, bn13_sAdd,
                        bn14_s1, bn14_s2, bn14_s3, bn14_sAdd.
        data_dir:      Directory containing compiled kernel .o files and
                       optional weight data text files.

    Returns:
        (workers, act_bn14_out, wts_fifos) where:
          workers:       flat list of all Worker objects
          act_bn14_out:  output ObjectFifo carrying (7,1,80) int8, depth=2
          wts_fifos:     list of streaming weight ObjectFifos the host DMA writes into:
                         [bn13_wts_l1_put, bn13_wts_l1_get,
                          bn13_wts_l3_put, bn13_wts_l3_get,
                          bn14_wts_l1_put, bn14_wts_l1_get,
                          bn14_wts_l3_put, bn14_wts_l3_get]

    Cascade flows between put/get worker pairs are constructed as side effects
    inside this function; they self-register on their source workers and are
    discovered automatically by Program.resolve().
    """
    workers = []
    bn13_s1, bn13_s2, bn13_s3, bn13_sAdd = (
        sf["BN13"]["conv1x1_1"], sf["BN13"]["conv3x3"],
        sf["BN13"]["conv1x1_2"], sf["BN13"]["skip_add"],
    )
    bn14_s1, bn14_s2, bn14_s3, bn14_sAdd = (
        sf["BN14"]["conv1x1_1"], sf["BN14"]["conv3x3"],
        sf["BN14"]["conv1x1_2"], sf["BN14"]["skip_add"],
    )

    # ========================================================================
    # Shared worker function bodies
    # (Same logic for bn13 and bn14 — different kernels and tiles.)
    # ========================================================================

    # L1 PUT: for each input row, loop over OutputSplit weight tiles and OC8
    # inner iterations, putting partial 1x1 results onto the cascade stream.
    def l1_put_fn(
        of_in,
        wts_fifo,
        k,
        InW,
        InC,
        OutC,
        InputSplit,
        OutputSplit,
        OC8,
        sf1,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            for WeightIndex in range_(OutputSplit):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8):
                    k(row_in, row_wts, InW, InC, OutC, InputSplit, WeightIndex, 0, oc)
                wts_fifo.release(1)
            of_in.release(1)

    # L1 GET: for each input row, accumulates cascade and produces full L1 output row.
    def l1_get_fn(of_in, of_out,
        wts_fifo,
        k,
        InW,
        InC,
        OutC,
        InputSplit,
        OutputSplit,
        OC8,
        sf1,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            for WeightIndex in range_(OutputSplit):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8):
                    k(
                        row_in,
                        row_wts,
                        row_out,
                        InW,
                        InC,
                        OutC,
                        sf1,
                        InputSplit,
                        OutputSplit,
                        WeightIndex,
                        0,
                        oc,
                    )
                wts_fifo.release(1)
            of_in.release(1)
            of_out.release(1)

    # L2 DW: depthwise 3x3 producing two split-channel output rows.
    def l2_fn(of_in, of_out_first, of_out_second, wts_buf, k, InW, OutC2, sf2):
        # preamble: top row (zero-pad above)
        rows = of_in.acquire(2)
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(
            rows[0],
            rows[0],
            rows[1],
            wts_buf,
            row_out_a,
            row_out_b,
            InW,
            1,
            OutC2,
            3,
            3,
            0,
            sf2,
            0,
        )
        of_out_first.release(1)
        of_out_second.release(1)

        # middle rows
        for _ in range_(_InH - 2):
            rows = of_in.acquire(3)
            row_out_a = of_out_first.acquire(1)
            row_out_b = of_out_second.acquire(1)
            k(
                rows[0],
                rows[1],
                rows[2],
                wts_buf,
                row_out_a,
                row_out_b,
                InW,
                1,
                OutC2,
                3,
                3,
                1,
                sf2,
                0,
            )
            of_in.release(1)
            of_out_first.release(1)
            of_out_second.release(1)

        # last row (zero-pad below)
        rows = of_in.acquire(2)
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(
            rows[0],
            rows[1],
            rows[1],
            wts_buf,
            row_out_a,
            row_out_b,
            InW,
            1,
            OutC2,
            3,
            3,
            2,
            sf2,
            0,
        )
        of_in.release(2)
        of_out_first.release(1)
        of_out_second.release(1)

    # L3 PUT: reads first DW split half, puts partial projection result onto cascade.
    def l3_put_fn(
        of_in,
        wts_fifo,
        k,
        InW,
        OutC2,
        OutC3,
        InputSplit,
        OutputSplit2,
        OC8_out,
        sf3,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8_out):
                    k(
                        row_in,
                        row_wts,
                        InW,
                        OutC2,
                        OutC3,
                        InputSplit,
                        WeightIndex,
                        0,
                        oc,
                    )
                wts_fifo.release(1)
            of_in.release(1)

    # L3 GET: reads second DW split half + cascade + skip → final output row.
    def l3_get_fn(of_in, skip_in,
        act_out,
        wts_fifo,
        k,
        InW,
        OutC2,
        OutC3,
        InputSplit,
        OutputSplit2,
        OC8_out,
        sf3,
        sfAdd,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = act_out.acquire(1)
            skip_row = skip_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8_out):
                    k(
                        row_in,
                        row_wts,
                        row_out,
                        skip_row,
                        InW,
                        OutC2,
                        OutC3,
                        sf3,
                        sfAdd,
                        InputSplit,
                        OutputSplit2,
                        WeightIndex,
                        0,
                        oc,
                    )
                wts_fifo.release(1)
            of_in.release(1)
            act_out.release(1)
            skip_in.release(1)

    # Skip connections are forwarded via MemTile DMA using ObjectFifo.forward().
    # No separate copy worker is needed.

    def _make_cascade_block(
        name, act_in, skip_in, scales, tiles,
    ):
        """One full cascade block (5 compute workers + 2 weight fifos).

        name:     "bn13" or "bn14"
        act_in:   the activation input ObjectFifo (drives L1 PUT and L1 GET)
        skip_in:  ObjectFifo whose .cons() forwards a skip row to L3 GET (often
                  the same as act_in, but for bn14 it is bn13's output)
        scales:   (s1, s2, s3, s_add)
        tiles:    dict with keys: l1_put, l1_get, l2, l3_put, l3_get,
                                   mem_l1, mem_l3, mem_skip
        Returns (out_fifo, wts_l1_full, wts_l3_full, [workers]).
        """
        s1, s2, s3, s_add = scales
        n = name[2:]  # "13" or "14"  (kernel sym names use bn_13_2_..., bn_14_2_...)

        k_l1_put = Kernel(
            f"{name}_1_conv2dk1_i8_ui8_partial_width_put_new",
            f"{name}_1_conv2dk1_put.o",
            [_ty_act_in, _ty_l1_split_wts] + [np.int32] * 7,
        )
        k_l1_get = Kernel(
            f"{name}_1_conv2dk1_i8_ui8_partial_width_get_new",
            f"{name}_1_conv2dk1_get.o",
            [_ty_act_in, _ty_l1_split_wts, _ty_l1_out_full] + [np.int32] * 9,
        )
        k_l2_dw = Kernel(
            f"{name}_conv2dk3_ui8_out_split",
            f"{name}_conv2dk3_dw.o",
            [_ty_l1_out_full, _ty_l1_out_full, _ty_l1_out_full,
             _ty_l2_wts, _ty_l1_out_split, _ty_l1_out_split] + [np.int32] * 8,
        )
        k_l3_put = Kernel(
            f"{name}_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
            f"{name}_conv2dk1_put.o",
            [_ty_l1_out_split, _ty_l3_split_wts] + [np.int32] * 7,
        )
        k_l3_get = Kernel(
            f"bn_{n}_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
            f"{name}_conv2dk1_skip_get.o",
            [_ty_l1_out_split, _ty_l3_split_wts, _ty_act_out, _ty_act_in]
            + [np.int32] * 10,
        )

        # Streaming weight fifos (Shim → MemTile → split → put/get tiles)
        wts_l1_full = ObjectFifo(_ty_l1_full_wts, depth=1)
        wts_l1_put_h, wts_l1_get_h = wts_l1_full.cons().split(
            offsets=[0, _l1_full_wts_sz // 2],
            depths=[1, 1],
            obj_types=[_ty_l1_split_wts, _ty_l1_split_wts],
            tile=tiles["mem_l1"],
            repeat_counts=[_InH, _InH],
        )
        wts_l3_full = ObjectFifo(_ty_l3_full_wts, depth=1)
        wts_l3_put_h, wts_l3_get_h = wts_l3_full.cons().split(
            offsets=[0, _l3_full_wts_sz // 2],
            depths=[1, 1],
            obj_types=[_ty_l3_split_wts, _ty_l3_split_wts],
            tile=tiles["mem_l3"],
            repeat_counts=[_InH, _InH],
        )

        # L2 DW weights are static (compile-time bake-in)
        l2_wts = _make_static_wts(
            data_dir, _l2_wts_sz, f"{name}_2_chain.txt", f"{name}_2_wts_static"
        )

        of_l1_l2 = ObjectFifo(
            np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
            depth=4, via_DMA=True,
        )
        of_l2_l3_first = ObjectFifo(
            np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]], depth=2,
        )
        of_l2_l3_second = ObjectFifo(
            np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]], depth=2,
        )
        out_fifo = ObjectFifo(
            np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]], depth=2,
        )

        # Pre-establish .cons() ordering to match expected MLIR placement.
        l1_put_cons = act_in.cons()
        l1_get_cons = act_in.cons()
        skip_fifo = skip_in.cons(depth=6).forward(depth=2, tile=tiles["mem_skip"])

        bws = [
            Worker(
                l1_put_fn,
                fn_args=[l1_put_cons, wts_l1_put_h.cons(), k_l1_put,
                         _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, s1],
                tile=tiles["l1_put"],
            ),
            Worker(
                l1_get_fn,
                fn_args=[l1_get_cons, of_l1_l2.prod(), wts_l1_get_h.cons(), k_l1_get,
                         _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, s1],
                tile=tiles["l1_get"],
            ),
            Worker(
                l2_fn,
                fn_args=[of_l1_l2.cons(), of_l2_l3_first.prod(),
                         of_l2_l3_second.prod(), l2_wts, k_l2_dw,
                         _InW, _L1_OutC, s2],
                tile=tiles["l2"],
            ),
            Worker(
                l3_put_fn,
                fn_args=[of_l2_l3_first.cons(), wts_l3_put_h.cons(), k_l3_put,
                         _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2,
                         _OC8_out, s3],
                tile=tiles["l3_put"],
            ),
            Worker(
                l3_get_fn,
                fn_args=[of_l2_l3_second.cons(), skip_fifo.cons(), out_fifo.prod(),
                         wts_l3_get_h.cons(), k_l3_get,
                         _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2,
                         _OC8_out, s3, s_add],
                tile=tiles["l3_get"],
            ),
        ]
        # Cascade flows: L1 put→get and L3 put→get share streams between adjacent tiles.
        CascadeFlow(bws[0], bws[1])
        CascadeFlow(bws[3], bws[4])
        return out_fifo, wts_l1_full, wts_l3_full, bws

    # bn13: cascade-split bottleneck (5 compute workers).
    act_bn13_out, bn13_wts_l1_full, bn13_wts_l3_full, bn13_workers = _make_cascade_block(
        "bn13", act_in, skip_in=act_in,
        scales=(bn13_s1, bn13_s2, bn13_s3, bn13_sAdd),
        tiles={
            "l1_put": Tile(4, 5), "l1_get": Tile(5, 5), "l2": Tile(5, 4),
            "l3_put": Tile(4, 3), "l3_get": Tile(5, 3),
            "mem_l1": Tile(0, 1), "mem_l3": Tile(1, 1), "mem_skip": Tile(5, 1),
        },
    )
    workers += bn13_workers


    # bn14: cascade-split bottleneck (5 compute workers, skip = bn13 output).
    act_bn14_out, bn14_wts_l1_full, bn14_wts_l3_full, bn14_workers = _make_cascade_block(
        "bn14", act_bn13_out, skip_in=act_bn13_out,
        scales=(bn14_s1, bn14_s2, bn14_s3, bn14_sAdd),
        tiles={
            "l1_put": Tile(6, 5), "l1_get": Tile(7, 5), "l2": Tile(6, 2),
            "l3_put": Tile(4, 2), "l3_get": Tile(5, 2),
            "mem_l1": Tile(2, 1), "mem_l3": Tile(3, 1), "mem_skip": Tile(7, 1),
        },
    )
    workers += bn14_workers


    wts_fifos = [
        bn13_wts_l1_full, bn13_wts_l3_full,
        bn14_wts_l1_full, bn14_wts_l3_full,
    ]

    return workers, act_bn14_out, wts_fifos
