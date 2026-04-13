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
  bn14: l1_put=Tile(6,5), l1_get=Tile(7,5), l2=Tile(7,4),
        l3_put=Tile(4,2), l3_get=Tile(5,2)
"""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_


# ---------------------------------------------------------------------------
# Dimensions (from aie2_bottleneckC.py)
# ---------------------------------------------------------------------------
_InW = 7
_InH = 7
_InC = 80           # input channels
_L1_OutC = 960      # expanded channels (L1 output)
_InputSplit = 2     # cascade splits: each cascade tile handles half the channels
_OutputSplit = 2
_OutputSplit2 = 2
_L1_SplitC = _L1_OutC // _InputSplit        # 480 channels per cascade tile
_OC8 = _L1_OutC // (8 * _OutputSplit)      # inner loop count for L1 kernel  = 60
_L3_OutC = 80       # final projection output channels
_OC8_out = _L3_OutC // (8 * _OutputSplit2)  # inner loop count for L3 kernel = 5

# ---------------------------------------------------------------------------
# Weight sizes (derived from aie2_bottleneckC.py type definitions)
# ---------------------------------------------------------------------------
# L1 split weight per tile:  (InC // InputSplit) * (L1_OutC // OutputSplit)
_l1_split_wts_sz = (_InC // _InputSplit) * (_L1_OutC // _OutputSplit)   # 40*480 = 19200
# L2 DW weight:     3 * 3 * L1_OutC (full depthwise filter)
_l2_wts_sz = 3 * 3 * _L1_OutC * 1                                        # 8640
# L3 split weight per tile:  (L1_OutC // InputSplit) * (L3_OutC // OutputSplit2)
_l3_split_wts_sz = (_L1_OutC // _InputSplit) * _L3_OutC  # 480*80 = 38400 (split by InputSplit only)

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
    *scale_factors: int,
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
        (workers, act_bn14_out, wts_fifos, cascade_pairs) where:
          workers:       flat list of all Worker objects
          act_bn14_out:  output ObjectFifo carrying (7,1,80) int8, depth=2
          wts_fifos:     list of streaming weight ObjectFifos the host DMA writes into:
                         [bn13_wts_l1_put, bn13_wts_l1_get,
                          bn13_wts_l3_put, bn13_wts_l3_get,
                          bn14_wts_l1_put, bn14_wts_l1_get,
                          bn14_wts_l3_put, bn14_wts_l3_get]
          cascade_pairs: list of (put_worker, get_worker) tuples:
                         [(w_bn13_l1_put, w_bn13_l1_get),
                          (w_bn13_l3_put, w_bn13_l3_get),
                          (w_bn14_l1_put, w_bn14_l1_get),
                          (w_bn14_l3_put, w_bn14_l3_get)]
    """
    workers = []

    (
        bn13_s1, bn13_s2, bn13_s3, bn13_sAdd,
        bn14_s1, bn14_s2, bn14_s3, bn14_sAdd,
    ) = scale_factors

    # ========================================================================
    # Shared worker function bodies
    # (Same logic for bn13 and bn14 — different kernels and tiles.)
    # ========================================================================

    # L1 PUT: for each input row, loop over OutputSplit weight tiles and OC8
    # inner iterations, putting partial 1x1 results onto the cascade stream.
    def l1_put_fn(
        of_in, wts_fifo, k,
        InW, InC, OutC, InputSplit, OutputSplit, OC8, sf1,
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
    def l1_get_fn(
        of_in, of_out, wts_fifo, k,
        InW, InC, OutC, InputSplit, OutputSplit, OC8, sf1,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            for WeightIndex in range_(OutputSplit):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8):
                    k(
                        row_in, row_wts, row_out,
                        InW, InC, OutC, sf1,
                        InputSplit, OutputSplit, WeightIndex, 0, oc,
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
        k(rows[0], rows[0], rows[1], wts_buf, row_out_a, row_out_b,
          InW, 1, OutC2, 3, 3, 0, sf2, 0)
        of_out_first.release(1)
        of_out_second.release(1)

        # middle rows
        for _ in range_(_InH - 2):
            rows = of_in.acquire(3)
            row_out_a = of_out_first.acquire(1)
            row_out_b = of_out_second.acquire(1)
            k(rows[0], rows[1], rows[2], wts_buf, row_out_a, row_out_b,
              InW, 1, OutC2, 3, 3, 1, sf2, 0)
            of_in.release(1)
            of_out_first.release(1)
            of_out_second.release(1)

        # last row (zero-pad below)
        rows = of_in.acquire(2)
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(rows[0], rows[1], rows[1], wts_buf, row_out_a, row_out_b,
          InW, 1, OutC2, 3, 3, 2, sf2, 0)
        of_in.release(2)
        of_out_first.release(1)
        of_out_second.release(1)

    # L3 PUT: reads first DW split half, puts partial projection result onto cascade.
    def l3_put_fn(
        of_in, wts_fifo, k,
        InW, OutC2, OutC3, InputSplit, OutputSplit2, OC8_out, sf3,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8_out):
                    k(row_in, row_wts, InW, OutC2, OutC3, InputSplit, WeightIndex, 0, oc)
                wts_fifo.release(1)
            of_in.release(1)

    # L3 GET: reads second DW split half + cascade + skip → final output row.
    def l3_get_fn(
        of_in, skip_in, act_out, wts_fifo, k,
        InW, OutC2, OutC3, InputSplit, OutputSplit2, OC8_out, sf3, sfAdd,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = act_out.acquire(1)
            skip_row = skip_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts_fifo.acquire(1)
                for oc in range_(OC8_out):
                    k(
                        row_in, row_wts, row_out, skip_row,
                        InW, OutC2, OutC3, sf3, sfAdd,
                        InputSplit, OutputSplit2, WeightIndex, 0, oc,
                    )
                wts_fifo.release(1)
            of_in.release(1)
            act_out.release(1)
            skip_in.release(1)

    # Skip connections are forwarded via MemTile DMA using ObjectFifo.forward().
    # No separate copy worker is needed.

    # ========================================================================
    # bn13 block
    # ========================================================================

    # -- Kernels (names from aie2_bottleneckC.py external_func declarations) --

    bn13_k_l1_put = Kernel(
        "bn13_1_conv2dk1_i8_ui8_partial_width_put_new",
        "bn13_1_conv2dk1_put.o",
        [
            _ty_act_in,        # act row in
            _ty_l1_split_wts,  # weight tile
            np.int32,          # W
            np.int32,          # InC
            np.int32,          # OutC (full)
            np.int32,          # InputSplit
            np.int32,          # WeightIndex
            np.int32,          # x_start
            np.int32,          # oc (inner oc8 index)
        ],
    )

    bn13_k_l1_get = Kernel(
        "bn13_1_conv2dk1_i8_ui8_partial_width_get_new",
        "bn13_1_conv2dk1_get.o",
        [
            _ty_act_in,        # act row in
            _ty_l1_split_wts,  # weight tile
            _ty_l1_out_full,   # act row out (full 960 channels)
            np.int32,          # W
            np.int32,          # InC
            np.int32,          # OutC (full)
            np.int32,          # scale (sf1)
            np.int32,          # InputSplit
            np.int32,          # OutputSplit
            np.int32,          # WeightIndex
            np.int32,          # x_start
            np.int32,          # oc
        ],
    )

    bn13_k_l2_dw = Kernel(
        "bn13_conv2dk3_ui8_out_split",
        "bn13_conv2dk3_dw.o",
        [
            _ty_l1_out_full,   # row above
            _ty_l1_out_full,   # row center
            _ty_l1_out_full,   # row below
            _ty_l2_wts,        # DW weights (full channels)
            _ty_l1_out_split,  # output first split half
            _ty_l1_out_split,  # output second split half
            np.int32,          # W
            np.int32,          # (constant 1)
            np.int32,          # OutC2 (full DW channels)
            np.int32,          # kernel H (3)
            np.int32,          # kernel W (3)
            np.int32,          # position flag (0=top,1=mid,2=bot)
            np.int32,          # scale (sf2)
            np.int32,          # (constant 0)
        ],
    )

    bn13_k_l3_put = Kernel(
        "bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
        "bn13_conv2dk1_put.o",
        [
            _ty_l1_out_split,  # act row in (first DW split)
            _ty_l3_split_wts,  # weight tile
            np.int32,          # W
            np.int32,          # OutC2 (full DW channels)
            np.int32,          # OutC3 (final output channels)
            np.int32,          # InputSplit
            np.int32,          # WeightIndex
            np.int32,          # x_start
            np.int32,          # oc
        ],
    )

    bn13_k_l3_get = Kernel(
        "bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
        "bn13_conv2dk1_skip_get.o",
        [
            _ty_l1_out_split,  # act row in (second DW split)
            _ty_l3_split_wts,  # weight tile
            _ty_act_out,       # act row out
            _ty_act_in,        # skip row in (bn13 input)
            np.int32,          # W
            np.int32,          # OutC2 (full DW channels)
            np.int32,          # OutC3 (final output channels)
            np.int32,          # scale (sf3)
            np.int32,          # scale_skip (sfAdd)
            np.int32,          # InputSplit
            np.int32,          # OutputSplit2
            np.int32,          # WeightIndex
            np.int32,          # x_start
            np.int32,          # oc
        ],
    )

    # -- Weight ObjectFifos for bn13 (streamed from host via Shim DMA) --
    # A single full-weight fifo comes from L3 (host DMA → MemTile).
    # The MemTile splits it into two halves, one per cascade tile.
    # This matches the original object_fifo_link pattern with offsets.
    # repeat_count=InH so the MemTile DMA replays each weight tile once per row.
    _l1_full_wts_sz = _InC * _L1_OutC           # 80 * 960 = 76800
    _l3_full_wts_sz = _L1_OutC * _L3_OutC       # 960 * 80 = 76800
    _ty_l1_full_wts = np.ndarray[(_l1_full_wts_sz,), np.dtype[np.int8]]
    _ty_l3_full_wts = np.ndarray[(_l3_full_wts_sz,), np.dtype[np.int8]]

    bn13_wts_l1_full = ObjectFifo(_ty_l1_full_wts, depth=1, name="bn13_wts_l1_full")
    bn13_wts_l1_put, bn13_wts_l1_get = bn13_wts_l1_full.cons().split(
        offsets=[0, _l1_split_wts_sz],
        depths=[1, 1],
        obj_types=[_ty_l1_split_wts, _ty_l1_split_wts],
        names=["bn13_wts_l1_put", "bn13_wts_l1_get"],
    )
    bn13_wts_l1_put.set_repeat_count(_InH)
    bn13_wts_l1_get.set_repeat_count(_InH)

    bn13_wts_l3_full = ObjectFifo(_ty_l3_full_wts, depth=1, name="bn13_wts_l3_full")
    bn13_wts_l3_put, bn13_wts_l3_get = bn13_wts_l3_full.cons().split(
        offsets=[0, _l3_split_wts_sz],
        depths=[1, 1],
        obj_types=[_ty_l3_split_wts, _ty_l3_split_wts],
        names=["bn13_wts_l3_put", "bn13_wts_l3_get"],
    )

    # L2 weights are static (baked in at compile time, from aie2_bn_13_14.py pattern)
    bn13_l2_wts = _make_static_wts(
        data_dir, _l2_wts_sz, "bn13_2_chain.txt", "bn13_l2_wts"
    )

    # -- Inter-tile activation ObjectFifos for bn13 --
    bn13_of_l1_l2 = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
        depth=4
    )
    bn13_of_l2_l3_first = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2
    )
    bn13_of_l2_l3_second = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2
    )
    act_bn13_out = ObjectFifo(
        np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]],
        depth=2
    )

    # Skip fifo: forward bn13 input to the L3 GET tile via MemTile DMA.
    # act_in has 3 consumers: l1_put, l1_get, and the skip forward.
    # We acquire depth=6 to allow all 7 rows to be buffered before the skip
    # consumer drains them (matching the original bn13_skip depth from source).
    bn13_skip_fifo = act_in.cons(depth=6).forward(name="bn13_skip_fifo", depth=2)

    # -- Create bn13 Workers --
    w_bn13_l1_put = Worker(
        l1_put_fn,
        fn_args=[
            act_in.cons(),
            bn13_wts_l1_put.cons(),
            bn13_k_l1_put,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn13_s1,
        ],
        placement=Tile(4, 5)
    )

    w_bn13_l1_get = Worker(
        l1_get_fn,
        fn_args=[
            act_in.cons(),
            bn13_of_l1_l2.prod(),
            bn13_wts_l1_get.cons(),
            bn13_k_l1_get,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn13_s1,
        ],
        placement=Tile(5, 5)
    )

    w_bn13_l2 = Worker(
        l2_fn,
        fn_args=[
            bn13_of_l1_l2.cons(),
            bn13_of_l2_l3_first.prod(),
            bn13_of_l2_l3_second.prod(),
            bn13_l2_wts,
            bn13_k_l2_dw,
            _InW, _L1_OutC, bn13_s2,
        ],
        placement=Tile(5, 4)
    )

    w_bn13_l3_put = Worker(
        l3_put_fn,
        fn_args=[
            bn13_of_l2_l3_first.cons(),
            bn13_wts_l3_put.cons(),
            bn13_k_l3_put,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out, bn13_s3,
        ],
        placement=Tile(4, 3)
    )

    w_bn13_l3_get = Worker(
        l3_get_fn,
        fn_args=[
            bn13_of_l2_l3_second.cons(),
            bn13_skip_fifo.cons(),
            act_bn13_out.prod(),
            bn13_wts_l3_get.cons(),
            bn13_k_l3_get,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out,
            bn13_s3, bn13_sAdd,
        ],
        placement=Tile(5, 3)
    )

    workers += [
        w_bn13_l1_put,
        w_bn13_l1_get,
        w_bn13_l2,
        w_bn13_l3_put,
        w_bn13_l3_get,
    ]

    # ========================================================================
    # bn14 block
    # ========================================================================

    bn14_k_l1_put = Kernel(
        "bn14_1_conv2dk1_i8_ui8_partial_width_put_new",
        "bn14_1_conv2dk1_put.o",
        [
            _ty_act_in,
            _ty_l1_split_wts,
            np.int32,  # W
            np.int32,  # InC
            np.int32,  # OutC
            np.int32,  # InputSplit
            np.int32,  # WeightIndex
            np.int32,  # x_start
            np.int32,  # oc
        ],
    )

    bn14_k_l1_get = Kernel(
        "bn14_1_conv2dk1_i8_ui8_partial_width_get_new",
        "bn14_1_conv2dk1_get.o",
        [
            _ty_act_in,
            _ty_l1_split_wts,
            _ty_l1_out_full,
            np.int32,  # W
            np.int32,  # InC
            np.int32,  # OutC
            np.int32,  # scale
            np.int32,  # InputSplit
            np.int32,  # OutputSplit
            np.int32,  # WeightIndex
            np.int32,  # x_start
            np.int32,  # oc
        ],
    )

    bn14_k_l2_dw = Kernel(
        "bn14_conv2dk3_ui8_out_split",
        "bn14_conv2dk3_dw.o",
        [
            _ty_l1_out_full,
            _ty_l1_out_full,
            _ty_l1_out_full,
            _ty_l2_wts,
            _ty_l1_out_split,
            _ty_l1_out_split,
            np.int32,  # W
            np.int32,  # (constant 1)
            np.int32,  # OutC2
            np.int32,  # kernel H (3)
            np.int32,  # kernel W (3)
            np.int32,  # position flag
            np.int32,  # scale
            np.int32,  # (constant 0)
        ],
    )

    bn14_k_l3_put = Kernel(
        "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
        "bn14_conv2dk1_put.o",
        [
            _ty_l1_out_split,
            _ty_l3_split_wts,
            np.int32,  # W
            np.int32,  # OutC2
            np.int32,  # OutC3
            np.int32,  # InputSplit
            np.int32,  # WeightIndex
            np.int32,  # x_start
            np.int32,  # oc
        ],
    )

    bn14_k_l3_get = Kernel(
        "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
        "bn14_conv2dk1_skip_get.o",
        [
            _ty_l1_out_split,
            _ty_l3_split_wts,
            _ty_act_out,
            _ty_act_in,        # skip (bn14 input = bn13 output)
            np.int32,  # W
            np.int32,  # OutC2
            np.int32,  # OutC3
            np.int32,  # scale
            np.int32,  # scale_skip
            np.int32,  # InputSplit
            np.int32,  # OutputSplit2
            np.int32,  # WeightIndex
            np.int32,  # x_start
            np.int32,  # oc
        ],
    )

    # -- Weight ObjectFifos for bn14 (same split pattern as bn13) --
    bn14_wts_l1_full = ObjectFifo(_ty_l1_full_wts, depth=1, name="bn14_wts_l1_full")
    bn14_wts_l1_put, bn14_wts_l1_get = bn14_wts_l1_full.cons().split(
        offsets=[0, _l1_split_wts_sz],
        depths=[1, 1],
        obj_types=[_ty_l1_split_wts, _ty_l1_split_wts],
        names=["bn14_wts_l1_put", "bn14_wts_l1_get"],
    )
    bn14_wts_l1_put.set_repeat_count(_InH)
    bn14_wts_l1_get.set_repeat_count(_InH)

    bn14_wts_l3_full = ObjectFifo(_ty_l3_full_wts, depth=1, name="bn14_wts_l3_full")
    bn14_wts_l3_put, bn14_wts_l3_get = bn14_wts_l3_full.cons().split(
        offsets=[0, _l3_split_wts_sz],
        depths=[1, 1],
        obj_types=[_ty_l3_split_wts, _ty_l3_split_wts],
        names=["bn14_wts_l3_put", "bn14_wts_l3_get"],
    )

    bn14_l2_wts = _make_static_wts(
        data_dir, _l2_wts_sz, "bn14_2_chain.txt", "bn14_l2_wts"
    )

    # -- Inter-tile activation ObjectFifos for bn14 --
    bn14_of_l1_l2 = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
        depth=4
    )
    bn14_of_l2_l3_first = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2
    )
    bn14_of_l2_l3_second = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2
    )
    act_bn14_out = ObjectFifo(
        np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]],
        depth=2
    )

    # bn14 skip: bn13 output forwarded to bn14 L3 GET via MemTile DMA.
    bn14_skip_fifo = act_bn13_out.cons(depth=6).forward(name="bn14_skip_fifo", depth=2)

    w_bn14_l1_put = Worker(
        l1_put_fn,
        fn_args=[
            act_bn13_out.cons(),
            bn14_wts_l1_put.cons(),
            bn14_k_l1_put,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn14_s1,
        ],
        placement=Tile(6, 5)
    )

    w_bn14_l1_get = Worker(
        l1_get_fn,
        fn_args=[
            act_bn13_out.cons(),
            bn14_of_l1_l2.prod(),
            bn14_wts_l1_get.cons(),
            bn14_k_l1_get,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn14_s1,
        ],
        placement=Tile(7, 5)
    )

    w_bn14_l2 = Worker(
        l2_fn,
        fn_args=[
            bn14_of_l1_l2.cons(),
            bn14_of_l2_l3_first.prod(),
            bn14_of_l2_l3_second.prod(),
            bn14_l2_wts,
            bn14_k_l2_dw,
            _InW, _L1_OutC, bn14_s2,
        ],
        placement=Tile(7, 4)
    )

    w_bn14_l3_put = Worker(
        l3_put_fn,
        fn_args=[
            bn14_of_l2_l3_first.cons(),
            bn14_wts_l3_put.cons(),
            bn14_k_l3_put,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out, bn14_s3,
        ],
        placement=Tile(4, 2)
    )

    w_bn14_l3_get = Worker(
        l3_get_fn,
        fn_args=[
            bn14_of_l2_l3_second.cons(),
            bn14_skip_fifo.cons(),
            act_bn14_out.prod(),
            bn14_wts_l3_get.cons(),
            bn14_k_l3_get,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out,
            bn14_s3, bn14_sAdd,
        ],
        placement=Tile(5, 2)
    )

    workers += [
        w_bn14_l1_put,
        w_bn14_l1_get,
        w_bn14_l2,
        w_bn14_l3_put,
        w_bn14_l3_get,
    ]

    # ========================================================================
    # Cascade pairs and weight fifos
    # ========================================================================
    cascade_pairs = [
        (w_bn13_l1_put, w_bn13_l1_get),
        (w_bn13_l3_put, w_bn13_l3_get),
        (w_bn14_l1_put, w_bn14_l1_get),
        (w_bn14_l3_put, w_bn14_l3_get),
    ]

    # wts_fifos: the full-weight ObjectFifos that the host DMA writes into.
    # The MemTile splits each into two halves via cons().split() above.
    # The orchestrator calls rt.fill(wf.prod(), wts_data) for each.
    wts_fifos = [
        bn13_wts_l1_full,
        bn13_wts_l3_full,
        bn14_wts_l1_full,
        bn14_wts_l3_full,
    ]

    return workers, act_bn14_out, wts_fifos, cascade_pairs
