#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Pipeline bottleneck blocks (bn10-bn12) for MobileNet V3 IRON API rewrite."""

import numpy as np
import os

from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import range_


# ---------------------------------------------------------------------------
# Tensor shape constants derived from aie2_bottleneckBStatic.py
# ---------------------------------------------------------------------------
# bn10: 3-layer (1x1-relu -> DW-stride1 3x3 -> 1x1, no skip)
#   L1: in=(14,1,80) int8,   out=(14,1,480) uint8
#   L2: in=(14,1,480) uint8, out=(14,1,480) uint8   (DW stride-1)
#   L3: in=(14,1,480) uint8, out=(14,1,112) int8
_BN10_IN_W = 14
_BN10_IN_H = 14
_BN10_IN_C = 80
_BN10_DW_CH = 480
_BN10_OUT_C = 112

# bn11: 3-layer (1x1-relu -> DW-stride1 3x3 -> 1x1-skip)
#   L1: in=(14,1,112) int8,  out=(14,1,336) uint8
#   L2: in=(14,1,336) uint8, out=(14,1,336) uint8   (DW stride-1)
#   L3: in=(14,1,336) uint8, out=(14,1,112) int8    (with skip add)
_BN11_IN_W = 14
_BN11_IN_H = 14
_BN11_IN_C = 112
_BN11_DW_CH = 336
_BN11_OUT_C = 112

# bn12: 2-tile (L1 on tile1, L2+L3 interleaved on tile2, DW stride-2)
#   L1: in=(14,1,112) int8,  out=(14,1,336) uint8
#   L2: in=(14,1,336) uint8, out=(7,1,336)  uint8   (DW stride-2)
#   L3: in=(7,1,336)  uint8, out=(7,1,80)   int8
_BN12_IN_W = 14
_BN12_IN_H = 14
_BN12_IN_C = 112
_BN12_DW_CH = 336
_BN12_OUT_C = 80
_BN12_OUT_W = 7
_BN12_OUT_H = 7


def _load_weights(data_dir, filename):
    """Load int8 weights from a comma-separated file; return zeros if missing."""
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return np.fromfile(path, sep=",", dtype=np.int8)
    # Return zeros shaped to match what Buffer expects (reshaped later by caller)
    return None


def pipeline_bottlenecks(
    act_in: ObjectFifo,
    *scale_factors: int,
    data_dir: str,
) -> tuple:
    """Implement bn10-bn12 pipeline bottleneck blocks.

    Args:
        act_in: from regular_bottlenecks: (14,1,80) int8, depth=2
        *scale_factors: bn10_s1, bn10_s2, bn10_s3,
                        bn11_s1, bn11_s2, bn11_s3, bn11_sAdd,
                        bn12_s1, bn12_s2, bn12_s3
        data_dir: Directory containing kernel .o files and weight data

    Returns:
        (workers, act_bn12_out) where act_bn12_out is (7,1,80) int8, depth=2
    """
    workers = []

    (
        bn10_s1,
        bn10_s2,
        bn10_s3,
        bn11_s1,
        bn11_s2,
        bn11_s3,
        bn11_sAdd,
        bn12_s1,
        bn12_s2,
        bn12_s3,
    ) = scale_factors

    # -------------------------------------------------------------------
    # Weight sizes (exact integers)
    # -------------------------------------------------------------------
    _bn10_l1_wts_size = _BN10_IN_C * _BN10_DW_CH          # 80*480 = 38400
    _bn10_l2_wts_size = 3 * 3 * _BN10_DW_CH * 1           # 9*480 = 4320
    _bn10_l3_wts_size = _BN10_DW_CH * _BN10_OUT_C          # 480*112 = 53760

    _bn11_l1_wts_size = _BN11_IN_C * _BN11_DW_CH          # 112*336 = 37632
    _bn11_l2_wts_size = 3 * 3 * _BN11_DW_CH * 1           # 9*336 = 3024
    _bn11_l3_wts_size = _BN11_DW_CH * _BN11_OUT_C          # 336*112 = 37632

    _bn12_l1_wts_size = _BN12_IN_C * _BN12_DW_CH          # 112*336 = 37632
    _bn12_l2_wts_size = 3 * 3 * _BN12_DW_CH * 1           # 9*336 = 3024
    _bn12_l3_wts_size = _BN12_DW_CH * _BN12_OUT_C          # 336*80 = 26880
    _bn12_l23_wts_size = _bn12_l2_wts_size + _bn12_l3_wts_size  # 29904

    # -------------------------------------------------------------------
    # Weight Buffers (static; zeros if file not present)
    # -------------------------------------------------------------------
    def _make_wts(size, filename, name):
        arr = _load_weights(data_dir, filename)
        if arr is None:
            arr = np.zeros((size,), dtype=np.int8)
        return Buffer(
            np.ndarray[(size,), np.dtype[np.int8]],
            initial_value=arr,
            name=name,
        )

    # Weight files use comma-separated text format: bn10_1_chain.txt etc.
    bn10_l1_wts = _make_wts(_bn10_l1_wts_size, "bn10_1_chain.txt", "bn10_l1_wts")
    bn10_l2_wts = _make_wts(_bn10_l2_wts_size, "bn10_2_chain.txt", "bn10_l2_wts")
    bn10_l3_wts = _make_wts(_bn10_l3_wts_size, "bn10_3_chain.txt", "bn10_l3_wts")

    bn11_l1_wts = _make_wts(_bn11_l1_wts_size, "bn11_1_chain.txt", "bn11_l1_wts")
    bn11_l2_wts = _make_wts(_bn11_l2_wts_size, "bn11_2_chain.txt", "bn11_l2_wts")
    bn11_l3_wts = _make_wts(_bn11_l3_wts_size, "bn11_3_chain.txt", "bn11_l3_wts")

    bn12_l1_wts = _make_wts(_bn12_l1_wts_size, "bn12_1_chain.txt", "bn12_l1_wts")
    bn12_l23_wts = _make_wts(_bn12_l23_wts_size, "bn12_2_3_chain.txt", "bn12_l23_wts")

    # -------------------------------------------------------------------
    # Type aliases
    # -------------------------------------------------------------------
    # bn10
    bn10_l1_in_ty = np.ndarray[(_BN10_IN_W, 1, _BN10_IN_C), np.dtype[np.int8]]
    bn10_l1_wts_ty = np.ndarray[(_bn10_l1_wts_size,), np.dtype[np.int8]]
    bn10_l1_out_ty = np.ndarray[(_BN10_IN_W, 1, _BN10_DW_CH), np.dtype[np.uint8]]
    bn10_l2_wts_ty = np.ndarray[(_bn10_l2_wts_size,), np.dtype[np.int8]]
    bn10_l2_out_ty = np.ndarray[(_BN10_IN_W, 1, _BN10_DW_CH), np.dtype[np.uint8]]
    bn10_l3_wts_ty = np.ndarray[(_bn10_l3_wts_size,), np.dtype[np.int8]]
    bn10_l3_out_ty = np.ndarray[(_BN10_IN_W, 1, _BN10_OUT_C), np.dtype[np.int8]]

    # bn11
    bn11_l1_in_ty = np.ndarray[(_BN11_IN_W, 1, _BN11_IN_C), np.dtype[np.int8]]
    bn11_l1_wts_ty = np.ndarray[(_bn11_l1_wts_size,), np.dtype[np.int8]]
    bn11_l1_out_ty = np.ndarray[(_BN11_IN_W, 1, _BN11_DW_CH), np.dtype[np.uint8]]
    bn11_l2_wts_ty = np.ndarray[(_bn11_l2_wts_size,), np.dtype[np.int8]]
    bn11_l2_out_ty = np.ndarray[(_BN11_IN_W, 1, _BN11_DW_CH), np.dtype[np.uint8]]
    bn11_l3_wts_ty = np.ndarray[(_bn11_l3_wts_size,), np.dtype[np.int8]]
    bn11_l3_out_ty = np.ndarray[(_BN11_IN_W, 1, _BN11_OUT_C), np.dtype[np.int8]]

    # bn12
    bn12_l1_in_ty = np.ndarray[(_BN12_IN_W, 1, _BN12_IN_C), np.dtype[np.int8]]
    bn12_l1_wts_ty = np.ndarray[(_bn12_l1_wts_size,), np.dtype[np.int8]]
    bn12_l1_out_ty = np.ndarray[(_BN12_IN_W, 1, _BN12_DW_CH), np.dtype[np.uint8]]
    bn12_l2_in_ty = np.ndarray[(_BN12_IN_W, 1, _BN12_DW_CH), np.dtype[np.uint8]]
    bn12_l2_wts_ty = np.ndarray[(_bn12_l2_wts_size,), np.dtype[np.int8]]
    bn12_l2_out_ty = np.ndarray[(_BN12_OUT_W, 1, _BN12_DW_CH), np.dtype[np.uint8]]
    bn12_l3_wts_ty = np.ndarray[(_bn12_l3_wts_size,), np.dtype[np.int8]]
    bn12_l3_out_ty = np.ndarray[(_BN12_OUT_W, 1, _BN12_OUT_C), np.dtype[np.int8]]
    bn12_l23_wts_ty = np.ndarray[(_bn12_l23_wts_size,), np.dtype[np.int8]]

    # -------------------------------------------------------------------
    # Kernel declarations
    # -------------------------------------------------------------------
    # bn10
    bn10_conv2dk1_relu = Kernel(
        "bn10_conv2dk1_relu_i8_ui8",
        "bn10_conv2dk1_fused_relu.o",
        [bn10_l1_in_ty, bn10_l1_wts_ty, bn10_l1_out_ty,
         np.int32, np.int32, np.int32, np.int32],
    )
    bn10_conv2dk3_dw = Kernel(
        "bn10_conv2dk3_dw_stride1_relu_ui8_ui8",
        "bn10_conv2dk3_dw.o",
        [bn10_l1_out_ty, bn10_l1_out_ty, bn10_l1_out_ty,
         bn10_l2_wts_ty, bn10_l2_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    bn10_conv2dk1_ui8 = Kernel(
        "bn10_conv2dk1_ui8_i8",
        "bn10_conv2dk1_ui8.o",
        [bn10_l2_out_ty, bn10_l3_wts_ty, bn10_l3_out_ty,
         np.int32, np.int32, np.int32, np.int32],
    )

    # bn11
    bn11_conv2dk1_relu = Kernel(
        "bn11_conv2dk1_relu_i8_ui8",
        "bn11_conv2dk1_fused_relu.o",
        [bn11_l1_in_ty, bn11_l1_wts_ty, bn11_l1_out_ty,
         np.int32, np.int32, np.int32, np.int32],
    )
    bn11_conv2dk3_dw = Kernel(
        "bn11_conv2dk3_dw_stride1_relu_ui8_ui8",
        "bn11_conv2dk3_dw.o",
        [bn11_l1_out_ty, bn11_l1_out_ty, bn11_l1_out_ty,
         bn11_l2_wts_ty, bn11_l2_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    bn11_conv2dk1_skip = Kernel(
        "bn11_conv2dk1_skip_ui8_i8_i8",
        "bn11_conv2dk1_skip.o",
        [bn11_l2_out_ty, bn11_l3_wts_ty, bn11_l3_out_ty, bn11_l1_in_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32],
    )

    # bn12
    bn12_conv2dk1_relu = Kernel(
        "bn12_conv2dk1_relu_i8_ui8",
        "bn12_conv2dk1_fused_relu.o",
        [bn12_l1_in_ty, bn12_l1_wts_ty, bn12_l1_out_ty,
         np.int32, np.int32, np.int32, np.int32],
    )
    bn12_conv2dk3_dw_stride2 = Kernel(
        "bn12_conv2dk3_dw_stride2_relu_ui8_ui8",
        "bn12_conv2dk3_dw_stride2.o",
        [bn12_l2_in_ty, bn12_l2_in_ty, bn12_l2_in_ty,
         bn12_l2_wts_ty, bn12_l2_out_ty,
         np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32, np.int32],
    )
    bn12_conv2dk1_ui8 = Kernel(
        "bn12_conv2dk1_ui8_i8",
        "bn12_conv2dk1_ui8.o",
        [bn12_l2_out_ty, bn12_l3_wts_ty, bn12_l3_out_ty,
         np.int32, np.int32, np.int32, np.int32],
    )

    # -------------------------------------------------------------------
    # Inter-layer ObjectFifos
    # -------------------------------------------------------------------
    # bn10: act_in -> L1 -> L2 -> L3 -> bn11
    act_bn10_l1_l2 = ObjectFifo(
        np.ndarray[(_BN10_IN_W, 1, _BN10_DW_CH), np.dtype[np.uint8]],
        depth=4,
        name="act_bn10_l1_l2",
    )
    act_bn10_l2_l3 = ObjectFifo(
        np.ndarray[(_BN10_IN_W, 1, _BN10_DW_CH), np.dtype[np.uint8]],
        depth=2,
        name="act_bn10_l2_l3",
    )
    act_bn10_l3_bn11 = ObjectFifo(
        np.ndarray[(_BN11_IN_W, 1, _BN11_IN_C), np.dtype[np.int8]],
        depth=2,
        name="act_bn10_l3_bn11",
    )

    # bn11 skip: forward bn10 output through a MemTile to bn11 L3.
    # The ObjectFifo.forward() creates the MemTile-mediated copy automatically.
    act_bn11_skip = act_bn10_l3_bn11.cons(depth=2).forward(
        name="act_bn11_skip", depth=2
    )
    act_bn11_l1_l2 = ObjectFifo(
        np.ndarray[(_BN11_IN_W, 1, _BN11_DW_CH), np.dtype[np.uint8]],
        depth=4,
        name="act_bn11_l1_l2",
    )
    act_bn11_l2_l3 = ObjectFifo(
        np.ndarray[(_BN11_IN_W, 1, _BN11_DW_CH), np.dtype[np.uint8]],
        depth=2,
        name="act_bn11_l2_l3",
    )
    act_bn11_l3_bn12 = ObjectFifo(
        np.ndarray[(_BN12_IN_W, 1, _BN12_IN_C), np.dtype[np.int8]],
        depth=2,
        name="act_bn11_l3_bn12",
    )

    # bn12: L1 -> L2 fifo; output
    act_bn12_l1_l2 = ObjectFifo(
        np.ndarray[(_BN12_IN_W, 1, _BN12_DW_CH), np.dtype[np.uint8]],
        depth=4,
        name="act_bn12_l1_l2",
    )
    act_bn12_out = ObjectFifo(
        np.ndarray[(_BN12_OUT_W, 1, _BN12_OUT_C), np.dtype[np.int8]],
        depth=2,
        name="act_bn12_out",
    )

    # -------------------------------------------------------------------
    # bn10 Workers
    # -------------------------------------------------------------------

    # bn10 L1: 1x1 conv relu, row by row
    def bn10_l1_fn(act_in_fifo, of_12, wts_buf, k_l1, W, InC, OutC, sf1):
        for _ in range_(_BN10_IN_H):
            row_in = act_in_fifo.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, W, InC, OutC, sf1)
            act_in_fifo.release(1)
            of_12.release(1)

    bn10_l1_worker = Worker(
        bn10_l1_fn,
        fn_args=[
            act_in.cons(),
            act_bn10_l1_l2.prod(),
            bn10_l1_wts,
            bn10_conv2dk1_relu,
            _BN10_IN_W,
            _BN10_IN_C,
            _BN10_DW_CH,
            bn10_s1,
        ],
        name="bn10_l1_worker",
    )
    workers.append(bn10_l1_worker)

    # bn10 L2: DW stride-1 3x3
    def bn10_l2_fn(of_12, of_23, wts_buf, k_l2, W, C, sf2):
        # preamble: top row (pad above = row 0)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[0], rows[1], wts_buf, row_out, W, 1, C, 3, 3, 0, sf2, 0)
        of_23.release(1)

        for _ in range_(_BN10_IN_H - 2):
            rows = of_12.acquire(3)
            row_out = of_23.acquire(1)
            k_l2(rows[0], rows[1], rows[2], wts_buf, row_out, W, 1, C, 3, 3, 1, sf2, 0)
            of_12.release(1)
            of_23.release(1)

        # last row (pad below = last row)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[1], rows[1], wts_buf, row_out, W, 1, C, 3, 3, 2, sf2, 0)
        of_12.release(2)
        of_23.release(1)

    bn10_l2_worker = Worker(
        bn10_l2_fn,
        fn_args=[
            act_bn10_l1_l2.cons(),
            act_bn10_l2_l3.prod(),
            bn10_l2_wts,
            bn10_conv2dk3_dw,
            _BN10_IN_W,
            _BN10_DW_CH,
            bn10_s2,
        ],
        name="bn10_l2_worker",
    )
    workers.append(bn10_l2_worker)

    # bn10 L3: 1x1 conv (no relu), row by row -> feeds bn11
    def bn10_l3_fn(of_23, act_out, wts_buf, k_l3, W, InC, OutC, sf3):
        for _ in range_(_BN10_IN_H):
            row_in = of_23.acquire(1)
            row_out = act_out.acquire(1)
            k_l3(row_in, wts_buf, row_out, W, InC, OutC, sf3)
            of_23.release(1)
            act_out.release(1)

    bn10_l3_worker = Worker(
        bn10_l3_fn,
        fn_args=[
            act_bn10_l2_l3.cons(),
            act_bn10_l3_bn11.prod(),
            bn10_l3_wts,
            bn10_conv2dk1_ui8,
            _BN10_IN_W,
            _BN10_DW_CH,
            _BN10_OUT_C,
            bn10_s3,
        ],
        name="bn10_l3_worker",
    )
    workers.append(bn10_l3_worker)

    # -------------------------------------------------------------------
    # bn11 Workers (3-tile pipeline, WITH skip)
    # The skip uses act_bn11_skip (forward() declared above), which carries
    # bn10 L3 output rows via MemTile DMA to bn11 L3.
    # -------------------------------------------------------------------

    def bn11_l1_fn(act_in_fifo, of_12, wts_buf, k_l1, W, InC, OutC, sf1):
        for _ in range_(_BN11_IN_H):
            row_in = act_in_fifo.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, W, InC, OutC, sf1)
            act_in_fifo.release(1)
            of_12.release(1)

    bn11_l1_worker = Worker(
        bn11_l1_fn,
        fn_args=[
            act_bn10_l3_bn11.cons(),
            act_bn11_l1_l2.prod(),
            bn11_l1_wts,
            bn11_conv2dk1_relu,
            _BN11_IN_W,
            _BN11_IN_C,
            _BN11_DW_CH,
            bn11_s1,
        ],
        name="bn11_l1_worker",
    )
    workers.append(bn11_l1_worker)

    # bn11 skip: forwarded via MemTile DMA (declared above as act_bn11_skip).
    # No separate copy worker needed.

    # bn11 L2: DW stride-1 3x3
    def bn11_l2_fn(of_12, of_23, wts_buf, k_l2, W, C, sf2):
        # preamble
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[0], rows[1], wts_buf, row_out, W, 1, C, 3, 3, 0, sf2, 0)
        of_23.release(1)

        for _ in range_(_BN11_IN_H - 2):
            rows = of_12.acquire(3)
            row_out = of_23.acquire(1)
            k_l2(rows[0], rows[1], rows[2], wts_buf, row_out, W, 1, C, 3, 3, 1, sf2, 0)
            of_12.release(1)
            of_23.release(1)

        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[1], rows[1], wts_buf, row_out, W, 1, C, 3, 3, 2, sf2, 0)
        of_12.release(2)
        of_23.release(1)

    bn11_l2_worker = Worker(
        bn11_l2_fn,
        fn_args=[
            act_bn11_l1_l2.cons(),
            act_bn11_l2_l3.prod(),
            bn11_l2_wts,
            bn11_conv2dk3_dw,
            _BN11_IN_W,
            _BN11_DW_CH,
            bn11_s2,
        ],
        name="bn11_l2_worker",
    )
    workers.append(bn11_l2_worker)

    # bn11 L3: 1x1 conv with skip add
    def bn11_l3_fn(of_23, skip_fifo, act_out, wts_buf, k_l3, W, InC, OutC, sf3, sfAdd):
        for _ in range_(_BN11_IN_H):
            row_in = of_23.acquire(1)
            skip_row = skip_fifo.acquire(1)
            row_out = act_out.acquire(1)
            k_l3(row_in, wts_buf, row_out, skip_row, W, InC, OutC, sf3, sfAdd)
            of_23.release(1)
            skip_fifo.release(1)
            act_out.release(1)

    bn11_l3_worker = Worker(
        bn11_l3_fn,
        fn_args=[
            act_bn11_l2_l3.cons(),
            act_bn11_skip.cons(),
            act_bn11_l3_bn12.prod(),
            bn11_l3_wts,
            bn11_conv2dk1_skip,
            _BN11_IN_W,
            _BN11_DW_CH,
            _BN11_OUT_C,
            bn11_s3,
            bn11_sAdd,
        ],
        name="bn11_l3_worker",
    )
    workers.append(bn11_l3_worker)

    # -------------------------------------------------------------------
    # bn12 Workers
    # -------------------------------------------------------------------

    # bn12 L1: 1x1 conv relu, row by row (14 rows input -> 14 rows output)
    def bn12_l1_fn(act_in_fifo, of_12, wts_buf, k_l1, W, InC, OutC, sf1):
        for _ in range_(_BN12_IN_H):
            row_in = act_in_fifo.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, W, InC, OutC, sf1)
            act_in_fifo.release(1)
            of_12.release(1)

    bn12_l1_worker = Worker(
        bn12_l1_fn,
        fn_args=[
            act_bn11_l3_bn12.cons(),
            act_bn12_l1_l2.prod(),
            bn12_l1_wts,
            bn12_conv2dk1_relu,
            _BN12_IN_W,
            _BN12_IN_C,
            _BN12_DW_CH,
            bn12_s1,
        ],
        name="bn12_l1_worker",
    )
    workers.append(bn12_l1_worker)

    # bn12 L2+L3 fused on tile2:
    # DW stride-2 reduces 14 input rows -> 7 output rows, then 1x1 applied per row.
    # The reference code (aie2_bottleneckBStatic.py lines 959-1146) interleaves
    # the DW-stride2 call and the 1x1 call for each output row within one tile.
    # Pattern (from reference):
    #   preamble:  acquire 2 in-rows, DW(row0,row0,row1) -> tmp, release 1 in-row
    #              acquire tmp, 1x1(tmp) -> out_row, release tmp, out_row
    #   middle (outH-2 iters): acquire 3 in-rows, DW(r0,r1,r2) -> tmp, release 2 in-rows
    #                           acquire tmp, 1x1(tmp) -> out_row, release tmp, out_row
    #   last:      acquire 3 in-rows, DW(r0,r1,r2) -> tmp, release 3 in-rows
    #              acquire tmp, 1x1(tmp) -> out_row, release tmp, out_row
    def bn12_l23_fn(of_12, act_out, wts_l2, wts_l3, k_dw, k_1x1,
                    inW, outW, DWC, OutC, sf2, sf3):
        # local fifo for DW -> 1x1 intermediate (single row at a time)
        act_l2_l3 = ObjectFifo(
            np.ndarray[(outW, 1, DWC), np.dtype[np.uint8]],
            depth=1,
            name="bn12_act_l2_l3",
        )

        # preamble: top row (zero-pad above)
        rows = of_12.acquire(2)
        row_tmp = act_l2_l3.acquire(1)
        k_dw(rows[0], rows[0], rows[1], wts_l2, row_tmp, inW, 1, DWC, 3, 3, 0, sf2, 0)
        of_12.release(1)
        act_l2_l3.release(1)

        row_tmp = act_l2_l3.acquire(1)
        row_out = act_out.acquire(1)
        k_1x1(row_tmp, wts_l3, row_out, outW, DWC, OutC, sf3)
        act_l2_l3.release(1)
        act_out.release(1)

        # middle rows (outH - 2 iterations)
        for _ in range_(_BN12_OUT_H - 2):
            rows = of_12.acquire(3)
            row_tmp = act_l2_l3.acquire(1)
            k_dw(rows[0], rows[1], rows[2], wts_l2, row_tmp, inW, 1, DWC, 3, 3, 1, sf2, 0)
            of_12.release(2)
            act_l2_l3.release(1)

            row_tmp = act_l2_l3.acquire(1)
            row_out = act_out.acquire(1)
            k_1x1(row_tmp, wts_l3, row_out, outW, DWC, OutC, sf3)
            act_l2_l3.release(1)
            act_out.release(1)

        # last row (zero-pad below)
        rows = of_12.acquire(3)
        row_tmp = act_l2_l3.acquire(1)
        k_dw(rows[0], rows[1], rows[2], wts_l2, row_tmp, inW, 1, DWC, 3, 3, 1, sf2, 0)
        of_12.release(3)
        act_l2_l3.release(1)

        row_tmp = act_l2_l3.acquire(1)
        row_out = act_out.acquire(1)
        k_1x1(row_tmp, wts_l3, row_out, outW, DWC, OutC, sf3)
        act_l2_l3.release(1)
        act_out.release(1)

    bn12_l23_worker = Worker(
        bn12_l23_fn,
        fn_args=[
            act_bn12_l1_l2.cons(),
            act_bn12_out.prod(),
            bn12_l23_wts,   # wts_l2 (caller slices; here we pass combined buf)
            bn12_l23_wts,   # wts_l3 (same buf, kernel indexes into correct offset)
            bn12_conv2dk3_dw_stride2,
            bn12_conv2dk1_ui8,
            _BN12_IN_W,
            _BN12_OUT_W,
            _BN12_DW_CH,
            _BN12_OUT_C,
            bn12_s2,
            bn12_s3,
        ],
        name="bn12_l23_worker",
    )
    workers.append(bn12_l23_worker)

    return workers, act_bn12_out
