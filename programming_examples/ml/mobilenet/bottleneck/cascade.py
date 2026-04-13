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
  Layer-1 GET tile:  reads cascade, runs DW-3x3 on full 960-ch activations
  Layer-2 tile:      DW-3x3 depthwise producing two split output fifos
  Layer-3 PUT tile:  1x1-proj on first DW-split half, puts onto cascade stream
  Layer-3 GET tile:  reads cascade, runs 1x1-proj+skip on second split, writes output

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
_InC = 80          # input channels
_L1_OutC = 960     # expanded channels (L1 output)
_InputSplit = 2    # cascade splits: each cascade tile handles half the channels
_OutputSplit = 2
_OutputSplit2 = 2
_L1_SplitC = _L1_OutC // _InputSplit   # 480 channels per cascade tile
_OC8 = _L1_OutC // (8 * _OutputSplit)  # inner loop count for L1 kernel
_L3_OutC = 80      # final projection output channels
_OC8_out = _L3_OutC // (8 * _OutputSplit2)  # inner loop count for L3 kernel

# ---------------------------------------------------------------------------
# Weight sizes (derived from aie2_bottleneckC.py type definitions)
# ---------------------------------------------------------------------------
# L1 split weight:  (InC // InputSplit) * (L1_OutC // OutputSplit)
_l1_split_wts_sz = (_InC // _InputSplit) * (_L1_OutC // _OutputSplit)  # 40*480 = 19200
# L2 DW weight:     3 * 3 * L1_OutC (full depthwise filter)
_l2_wts_sz = 3 * 3 * _L1_OutC * 1  # 8640
# L3 split weight:  (L1_OutC // InputSplit) * (L3_OutC // OutputSplit2)
_l3_split_wts_sz = (_L1_OutC // _InputSplit) * (_L3_OutC // _OutputSplit2)  # 480*40 = 19200
# L3 full weight (for reference):  L1_OutC * L3_OutC = 960*80 = 76800


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_weights(data_dir, filename):
    """Load int8 weights from a binary file; return None if missing."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.int8)
    return None


def _make_wts(data_dir, size, filename, name):
    arr = _load_weights(data_dir, filename)
    if arr is None:
        arr = np.zeros((size,), dtype=np.int8)
    return Buffer(
        np.ndarray[(size,), np.dtype[np.int8]],
        initial_value=arr,
        name=name,
    )


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
# Public API
# ---------------------------------------------------------------------------

def cascade_bottlenecks(
    act_in: ObjectFifo,
    *scale_factors: int,
    data_dir: str,
) -> tuple:
    """Implement bn13 and bn14 cascade bottleneck blocks.

    Each block has 5 compute workers, with 2 cascade stream put/get pairs per
    block (4 cascade pairs total).

    Args:
        act_in:        Input ObjectFifo carrying (7,1,80) int8 activations, depth=2.
        *scale_factors: Eight scale factors in order:
                        bn13_s1, bn13_s2, bn13_s3, bn13_sAdd,
                        bn14_s1, bn14_s2, bn14_s3, bn14_sAdd.
        data_dir:      Directory containing compiled kernel .o files and
                       optional weight data binary files.

    Returns:
        (workers, act_bn14_out, cascade_pairs) where:
          workers:       flat list of all 10 Worker objects
          act_bn14_out:  output ObjectFifo carrying (7,1,80) int8, depth=2
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
    # bn13 block
    # ========================================================================

    # -- Kernels (names from aie2_bottleneckC.py external_func declarations) --
    # L1 PUT: bn13_1_conv2dk1_i8_ui8_partial_width_put_new
    #   inputs: [act_in, wts_split, W, InC, OutC, InputSplit, WeightIndex, x_start, oc]
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

    # L1 GET: bn13_1_conv2dk1_i8_ui8_partial_width_get_new
    #   inputs: [act_in, wts_split, act_out_full, W, InC, OutC, scale,
    #            InputSplit, OutputSplit, WeightIndex, x_start, oc]
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

    # L2 DW: bn13_conv2dk3_ui8_out_split
    #   inputs: [in0, in1, in2, wts, out_split_first, out_split_second,
    #            W, 1, OutC2, 3, 3, pos, scale, 0]
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

    # L3 PUT: bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new
    #   inputs: [in_split, wts_split, W, OutC2, OutC3, InputSplit,
    #            WeightIndex, x_start, oc]
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

    # L3 GET: bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new
    #   inputs: [in_split, wts_split, act_out, skip_in, W, OutC2, OutC3,
    #            scale, scale_skip, InputSplit, OutputSplit2, WeightIndex, x_start, oc]
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

    # -- Weight buffers --
    bn13_l1_put_wts = _make_wts(
        data_dir, _l1_split_wts_sz, "bn13_l1_put_wts.bin", "bn13_l1_put_wts"
    )
    bn13_l1_get_wts = _make_wts(
        data_dir, _l1_split_wts_sz, "bn13_l1_get_wts.bin", "bn13_l1_get_wts"
    )
    bn13_l2_wts = _make_wts(
        data_dir, _l2_wts_sz, "bn13_l2_wts.bin", "bn13_l2_wts"
    )
    bn13_l3_put_wts = _make_wts(
        data_dir, _l3_split_wts_sz, "bn13_l3_put_wts.bin", "bn13_l3_put_wts"
    )
    bn13_l3_get_wts = _make_wts(
        data_dir, _l3_split_wts_sz, "bn13_l3_get_wts.bin", "bn13_l3_get_wts"
    )

    # -- Inter-tile ObjectFifos --
    # act_in fans out to: l1_put worker, l1_get worker, and bn13_skip forward
    # The skip connection (bn13 input -> bn13 L3 GET) is modelled as a
    # forward fifo that the L3 GET worker reads from. In the reference design
    # this goes through a MemTile via object_fifo_link; here we declare it as
    # a separate consumer port (depth=6 for the forward path to ensure the
    # pipeline does not stall while rows are consumed across two blocks).

    # Output fifo for the full L1 activations (960 ch, uint8), produced by
    # the L1 GET tile and consumed by the L2 DW tile.
    bn13_of_l1_l2 = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
        depth=4,
        name="bn13_of_l1_l2",
    )

    # L2 DW produces two split halves (480 ch each) for L3 PUT and L3 GET.
    bn13_of_l2_l3_first = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
        name="bn13_of_l2_l3_first",
    )
    bn13_of_l2_l3_second = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
        name="bn13_of_l2_l3_second",
    )

    # bn13 output (also the input to bn14).
    act_bn13_out = ObjectFifo(
        np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]],
        depth=2,
        name="act_bn13_out",
    )

    # -- Worker function bodies --

    # L1 PUT: reads act_in row by row, calls put kernel in inner loops
    # (WeightIndex x OC8 inner loops, matching the reference core body).
    def bn13_l1_put_fn(
        of_in, wts, k,
        InW, InC, OutC, InputSplit, OutputSplit, OC8, sf1,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            for WeightIndex in range_(OutputSplit):
                row_wts = wts.acquire(1)
                for oc in range_(OC8):
                    k(row_in, row_wts, InW, InC, OutC, InputSplit, WeightIndex, 0, oc)
                wts.release(1)
            of_in.release(1)

    # L1 GET: reads act_in row by row, calls get kernel to produce full output row.
    def bn13_l1_get_fn(
        of_in, of_out, wts, k,
        InW, InC, OutC, InputSplit, OutputSplit, OC8, sf1,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = of_out.acquire(1)
            for WeightIndex in range_(OutputSplit):
                row_wts = wts.acquire(1)
                for oc in range_(OC8):
                    k(
                        row_in, row_wts, row_out,
                        InW, InC, OutC, sf1,
                        InputSplit, OutputSplit, WeightIndex, 0, oc,
                    )
                wts.release(1)
            of_in.release(1)
            of_out.release(1)

    # L2 DW: produces two split-channel output rows per input row.
    def bn13_l2_fn(of_in, of_out_first, of_out_second, wts, k, InW, OutC2, sf2):
        # preamble: top row (zero-pad above)
        rows = of_in.acquire(2)
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out_a, row_out_b,
          InW, 1, OutC2, 3, 3, 0, sf2, 0)
        of_out_first.release(1)
        of_out_second.release(1)

        # middle rows
        for _ in range_(_InH - 2):
            rows = of_in.acquire(3)
            row_out_a = of_out_first.acquire(1)
            row_out_b = of_out_second.acquire(1)
            k(rows[0], rows[1], rows[2], wts, row_out_a, row_out_b,
              InW, 1, OutC2, 3, 3, 1, sf2, 0)
            of_in.release(1)
            of_out_first.release(1)
            of_out_second.release(1)

        # last row (zero-pad below)
        rows = of_in.acquire(2)
        row_out_a = of_out_first.acquire(1)
        row_out_b = of_out_second.acquire(1)
        k(rows[0], rows[1], rows[1], wts, row_out_a, row_out_b,
          InW, 1, OutC2, 3, 3, 2, sf2, 0)
        of_in.release(2)
        of_out_first.release(1)
        of_out_second.release(1)

    # L3 PUT: reads first DW split half, calls put kernel, puts half on cascade.
    def bn13_l3_put_fn(
        of_in, wts, k,
        InW, OutC2, OutC3, InputSplit, OutputSplit2, OC8_out, sf3,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts.acquire(1)
                for oc in range_(OC8_out):
                    k(row_in, row_wts, InW, OutC2, OutC3, InputSplit, WeightIndex, 0, oc)
                wts.release(1)
            of_in.release(1)

    # L3 GET: reads second DW split half + skip, produces final output row.
    def bn13_l3_get_fn(
        of_in, skip_in, act_out, wts, k,
        InW, OutC2, OutC3, InputSplit, OutputSplit2, OC8_out, sf3, sfAdd,
    ):
        for _ in range_(_InH):
            row_in = of_in.acquire(1)
            row_out = act_out.acquire(1)
            skip_row = skip_in.acquire(1)
            for WeightIndex in range_(OutputSplit2):
                row_wts = wts.acquire(1)
                for oc in range_(OC8_out):
                    k(
                        row_in, row_wts, row_out, skip_row,
                        InW, OutC2, OutC3, sf3, sfAdd,
                        InputSplit, OutputSplit2, WeightIndex, 0, oc,
                    )
                wts.release(1)
            of_in.release(1)
            act_out.release(1)
            skip_in.release(1)

    # -- Create bn13 Workers --
    w_bn13_l1_put = Worker(
        bn13_l1_put_fn,
        fn_args=[
            act_in.cons(),
            bn13_l1_put_wts,
            bn13_k_l1_put,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn13_s1,
        ],
        placement=Tile(4, 5),
        name="w_bn13_l1_put",
    )

    w_bn13_l1_get = Worker(
        bn13_l1_get_fn,
        fn_args=[
            act_in.cons(),
            bn13_of_l1_l2.prod(),
            bn13_l1_get_wts,
            bn13_k_l1_get,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn13_s1,
        ],
        placement=Tile(5, 5),
        name="w_bn13_l1_get",
    )

    w_bn13_l2 = Worker(
        bn13_l2_fn,
        fn_args=[
            bn13_of_l1_l2.cons(),
            bn13_of_l2_l3_first.prod(),
            bn13_of_l2_l3_second.prod(),
            bn13_l2_wts,
            bn13_k_l2_dw,
            _InW, _L1_OutC, bn13_s2,
        ],
        placement=Tile(5, 4),
        name="w_bn13_l2",
    )

    w_bn13_l3_put = Worker(
        bn13_l3_put_fn,
        fn_args=[
            bn13_of_l2_l3_first.cons(),
            bn13_l3_put_wts,
            bn13_k_l3_put,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out, bn13_s3,
        ],
        placement=Tile(4, 3),
        name="w_bn13_l3_put",
    )

    # The bn13 skip connection carries the bn13 block input forward to the L3 GET
    # tile. act_in is consumed by l1_put, l1_get, and the skip path. The skip
    # fifo holds up to 6 rows (InH=7 rows, pipeline depth).
    bn13_skip_fifo = ObjectFifo(
        np.ndarray[(_InW, 1, _InC), np.dtype[np.int8]],
        depth=2,
        name="bn13_skip_fifo",
    )

    # Skip forwarder: copies rows from act_in to bn13_skip_fifo element-wise.
    def bn13_skip_fn(src, dst, total_elems):
        for _ in range_(_InH):
            row_in = src.acquire(1)
            row_out = dst.acquire(1)
            for i in range_(total_elems):
                row_out[i] = row_in[i]
            src.release(1)
            dst.release(1)

    w_bn13_skip = Worker(
        bn13_skip_fn,
        fn_args=[
            act_in.cons(),
            bn13_skip_fifo.prod(),
            _InW * _InC,
        ],
        name="w_bn13_skip",
    )

    w_bn13_l3_get = Worker(
        bn13_l3_get_fn,
        fn_args=[
            bn13_of_l2_l3_second.cons(),
            bn13_skip_fifo.cons(),
            act_bn13_out.prod(),
            bn13_l3_get_wts,
            bn13_k_l3_get,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out,
            bn13_s3, bn13_sAdd,
        ],
        placement=Tile(5, 3),
        name="w_bn13_l3_get",
    )

    workers += [
        w_bn13_l1_put,
        w_bn13_l1_get,
        w_bn13_l2,
        w_bn13_skip,
        w_bn13_l3_put,
        w_bn13_l3_get,
    ]

    # ========================================================================
    # bn14 block
    # ========================================================================

    # -- Kernels (bn14 variants from aie2_bottleneckC.py) --
    # L1 PUT: bn14_1_conv2dk1_i8_ui8_partial_width_put_new
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

    # L1 GET: bn14_1_conv2dk1_i8_ui8_partial_width_get_new
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

    # L2 DW: bn14_conv2dk3_ui8_out_split
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

    # L3 PUT: bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new
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

    # L3 GET: bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new
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

    # -- Weight buffers --
    bn14_l1_put_wts = _make_wts(
        data_dir, _l1_split_wts_sz, "bn14_l1_put_wts.bin", "bn14_l1_put_wts"
    )
    bn14_l1_get_wts = _make_wts(
        data_dir, _l1_split_wts_sz, "bn14_l1_get_wts.bin", "bn14_l1_get_wts"
    )
    bn14_l2_wts = _make_wts(
        data_dir, _l2_wts_sz, "bn14_l2_wts.bin", "bn14_l2_wts"
    )
    bn14_l3_put_wts = _make_wts(
        data_dir, _l3_split_wts_sz, "bn14_l3_put_wts.bin", "bn14_l3_put_wts"
    )
    bn14_l3_get_wts = _make_wts(
        data_dir, _l3_split_wts_sz, "bn14_l3_get_wts.bin", "bn14_l3_get_wts"
    )

    # -- Inter-tile ObjectFifos for bn14 --
    bn14_of_l1_l2 = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_OutC), np.dtype[np.uint8]],
        depth=4,
        name="bn14_of_l1_l2",
    )
    bn14_of_l2_l3_first = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
        name="bn14_of_l2_l3_first",
    )
    bn14_of_l2_l3_second = ObjectFifo(
        np.ndarray[(_InW, 1, _L1_SplitC), np.dtype[np.uint8]],
        depth=2,
        name="bn14_of_l2_l3_second",
    )
    act_bn14_out = ObjectFifo(
        np.ndarray[(_InW, 1, _L3_OutC), np.dtype[np.int8]],
        depth=2,
        name="act_bn14_out",
    )

    # bn14 skip: bn13 output forwarded to bn14 L3 GET.
    bn14_skip_fifo = ObjectFifo(
        np.ndarray[(_InW, 1, _InC), np.dtype[np.int8]],
        depth=2,
        name="bn14_skip_fifo",
    )

    def bn14_skip_fn(src, dst, total_elems):
        for _ in range_(_InH):
            row_in = src.acquire(1)
            row_out = dst.acquire(1)
            for i in range_(total_elems):
                row_out[i] = row_in[i]
            src.release(1)
            dst.release(1)

    w_bn14_skip = Worker(
        bn14_skip_fn,
        fn_args=[
            act_bn13_out.cons(),
            bn14_skip_fifo.prod(),
            _InW * _InC,
        ],
        name="w_bn14_skip",
    )

    # -- Create bn14 Workers (same fn bodies as bn13, different kernels/tiles) --
    w_bn14_l1_put = Worker(
        bn13_l1_put_fn,
        fn_args=[
            act_bn13_out.cons(),
            bn14_l1_put_wts,
            bn14_k_l1_put,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn14_s1,
        ],
        placement=Tile(6, 5),
        name="w_bn14_l1_put",
    )

    w_bn14_l1_get = Worker(
        bn13_l1_get_fn,
        fn_args=[
            act_bn13_out.cons(),
            bn14_of_l1_l2.prod(),
            bn14_l1_get_wts,
            bn14_k_l1_get,
            _InW, _InC, _L1_OutC, _InputSplit, _OutputSplit, _OC8, bn14_s1,
        ],
        placement=Tile(7, 5),
        name="w_bn14_l1_get",
    )

    w_bn14_l2 = Worker(
        bn13_l2_fn,
        fn_args=[
            bn14_of_l1_l2.cons(),
            bn14_of_l2_l3_first.prod(),
            bn14_of_l2_l3_second.prod(),
            bn14_l2_wts,
            bn14_k_l2_dw,
            _InW, _L1_OutC, bn14_s2,
        ],
        placement=Tile(7, 4),
        name="w_bn14_l2",
    )

    w_bn14_l3_put = Worker(
        bn13_l3_put_fn,
        fn_args=[
            bn14_of_l2_l3_first.cons(),
            bn14_l3_put_wts,
            bn14_k_l3_put,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out, bn14_s3,
        ],
        placement=Tile(4, 2),
        name="w_bn14_l3_put",
    )

    w_bn14_l3_get = Worker(
        bn13_l3_get_fn,
        fn_args=[
            bn14_of_l2_l3_second.cons(),
            bn14_skip_fifo.cons(),
            act_bn14_out.prod(),
            bn14_l3_get_wts,
            bn14_k_l3_get,
            _InW, _L1_OutC, _L3_OutC, _InputSplit, _OutputSplit2, _OC8_out,
            bn14_s3, bn14_sAdd,
        ],
        placement=Tile(5, 2),
        name="w_bn14_l3_get",
    )

    workers += [
        w_bn14_l1_put,
        w_bn14_l1_get,
        w_bn14_l2,
        w_bn14_skip,
        w_bn14_l3_put,
        w_bn14_l3_get,
    ]

    # ========================================================================
    # Cascade pairs: (put_worker, get_worker) for each cascade stream
    # ========================================================================
    cascade_pairs = [
        (w_bn13_l1_put, w_bn13_l1_get),
        (w_bn13_l3_put, w_bn13_l3_get),
        (w_bn14_l1_put, w_bn14_l1_get),
        (w_bn14_l3_put, w_bn14_l3_get),
    ]

    return workers, act_bn14_out, cascade_pairs
