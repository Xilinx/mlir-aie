#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.

import numpy as np
import os
from aie.iron import Buffer, Kernel, ObjectFifo, Worker
from aie.iron.controlflow import range_


def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def _u8(shape):
    return np.ndarray[shape, np.dtype[np.uint8]]


def _i32():
    return np.int32


def _load_weights(data_dir, filename):
    path = os.path.join(data_dir, filename)
    if os.path.exists(path):
        return np.fromfile(path, dtype=np.int8)
    return None


def pipeline_bottlenecks(
    act_in: ObjectFifo,
    *scale_factors: int,
    data_dir: str,
) -> tuple:
    """Returns (workers, act_bn12_out)

    scale_factors order:
        bn10_s1, bn10_s2, bn10_s3,
        bn11_s1, bn11_s2, bn11_s3, bn11_sAdd,
        bn12_s1, bn12_s2, bn12_s3
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

    # ---- bn10 ----
    # Dimensions (from aie2_bottleneckBStatic.py):
    #   b10_InW1=14, b10_InH1=14, b10_InC1=80, b10_OutC1=480, b10_OutC3=112
    #   L1: in=(14,1,80) int8,   out=(14,1,480) uint8  wts=80*480=38400
    #   L2: in=(14,1,480) uint8, out=(14,1,480) uint8  wts=3*3*480=4320  (DW stride-1)
    #   L3: in=(14,1,480) uint8, out=(14,1,112) int8   wts=480*112=53760
    bn10_l1_wts_sz = 80 * 480    # 38400
    bn10_l2_wts_sz = 3 * 3 * 480  # 4320
    bn10_l3_wts_sz = 480 * 112   # 53760

    bn10_l1_data = _load_weights(data_dir, "bn10_1_chain.txt")
    bn10_l2_data = _load_weights(data_dir, "bn10_2_chain.txt")
    bn10_l3_data = _load_weights(data_dir, "bn10_3_chain.txt")

    bn10_l1_wts = Buffer(
        _i8((bn10_l1_wts_sz,)),
        initial_value=(
            bn10_l1_data
            if bn10_l1_data is not None
            else np.zeros(bn10_l1_wts_sz, dtype=np.int8)
        )
    )
    bn10_l2_wts = Buffer(
        _i8((bn10_l2_wts_sz,)),
        initial_value=(
            bn10_l2_data
            if bn10_l2_data is not None
            else np.zeros(bn10_l2_wts_sz, dtype=np.int8)
        )
    )
    bn10_l3_wts = Buffer(
        _i8((bn10_l3_wts_sz,)),
        initial_value=(
            bn10_l3_data
            if bn10_l3_data is not None
            else np.zeros(bn10_l3_wts_sz, dtype=np.int8)
        )
    )

    # Kernel declarations matching aie2_bottleneckBStatic.py external_func signatures.
    # bn10 L1: (in(14,1,80)i8, wts(38400)i8, out(14,1,480)u8, W, InC, OutC, scale)
    k_bn10_l1 = Kernel(
        "bn10_conv2dk1_relu_i8_ui8",
        "bn10_conv2dk1_fused_relu.o",
        [
            _i8((14, 1, 80)),
            _i8((38400,)),
            _u8((14, 1, 480)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn10 L2 DW stride-1:
    #   (top, mid, bot: u8(14,1,480), wts(4320)i8, out(14,1,480)u8,
    #    W, 1, C, kH, kW, border, scale, 0)  -- 13 args
    k_bn10_l2 = Kernel(
        "bn10_conv2dk3_dw_stride1_relu_ui8_ui8",
        "bn10_conv2dk3_dw.o",
        [
            _u8((14, 1, 480)),
            _u8((14, 1, 480)),
            _u8((14, 1, 480)),
            _i8((4320,)),
            _u8((14, 1, 480)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn10 L3: (in(14,1,480)u8, wts(53760)i8, out(14,1,112)i8, W, InC, OutC, scale)
    k_bn10_l3 = Kernel(
        "bn10_conv2dk1_ui8_i8",
        "bn10_conv2dk1_ui8.o",
        [
            _u8((14, 1, 480)),
            _i8((53760,)),
            _i8((14, 1, 112)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )

    # Internal fifos (depth=4 for L1->L2 matching source OF_b10_act_layer1_layer2)
    bn10_of_12 = ObjectFifo(_u8((14, 1, 480)), depth=4, name="bn10_of_12")
    bn10_of_23 = ObjectFifo(_u8((14, 1, 480)), depth=2, name="bn10_of_23")
    act_bn10_out = ObjectFifo(_i8((14, 1, 112)), depth=2, name="act_bn10_out")

    def bn10_l1_fn(act_in, of_12, wts_buf, k_l1, sf1):
        for _ in range_(14):
            row_in = act_in.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, 14, 80, 480, sf1)
            act_in.release(1)
            of_12.release(1)

    def bn10_l2_fn(of_12, of_23, wts_buf, k_l2, sf2):
        # preamble: top row (border=0, replicate row 0 above)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[0], rows[1], wts_buf, row_out, 14, 1, 480, 3, 3, 0, sf2, 0)
        of_23.release(1)
        # middle rows (border=1): b10_InH2 - 2 = 14 - 2 = 12
        for _ in range_(12):
            rows = of_12.acquire(3)
            row_out = of_23.acquire(1)
            k_l2(rows[0], rows[1], rows[2], wts_buf, row_out, 14, 1, 480, 3, 3, 1, sf2, 0)
            of_12.release(1)
            of_23.release(1)
        # postamble: bottom row (border=2, replicate last row below)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[1], rows[1], wts_buf, row_out, 14, 1, 480, 3, 3, 2, sf2, 0)
        of_12.release(2)
        of_23.release(1)

    def bn10_l3_fn(of_23, act_out, wts_buf, k_l3, sf3):
        for _ in range_(14):
            row_in = of_23.acquire(1)
            row_out = act_out.acquire(1)
            k_l3(row_in, wts_buf, row_out, 14, 480, 112, sf3)
            of_23.release(1)
            act_out.release(1)

    workers += [
        Worker(
            bn10_l1_fn,
            [act_in.cons(), bn10_of_12.prod(), bn10_l1_wts, k_bn10_l1, bn10_s1],
        ),
        Worker(
            bn10_l2_fn,
            [bn10_of_12.cons(), bn10_of_23.prod(), bn10_l2_wts, k_bn10_l2, bn10_s2],
        ),
        Worker(
            bn10_l3_fn,
            [bn10_of_23.cons(), act_bn10_out.prod(), bn10_l3_wts, k_bn10_l3, bn10_s3],
        ),
    ]

    # ---- bn11 (with skip) ----
    # bn11 skip: bn10 output forwarded via MemTile DMA to bn11 L3.
    # forward() creates the MemTile-side copy, no extra compute tile needed.
    bn11_skip_of = act_bn10_out.cons(depth=2).forward(name="bn11_skip_of", depth=2)

    # b11_OutC1=336, b11_OutC2=336, b11_OutC3=112
    bn11_l1_wts_sz = 112 * 336   # 37632
    bn11_l2_wts_sz = 3 * 3 * 336  # 3024
    bn11_l3_wts_sz = 336 * 112   # 37632

    bn11_l1_data = _load_weights(data_dir, "bn11_1_chain.txt")
    bn11_l2_data = _load_weights(data_dir, "bn11_2_chain.txt")
    bn11_l3_data = _load_weights(data_dir, "bn11_3_chain.txt")

    bn11_l1_wts = Buffer(
        _i8((bn11_l1_wts_sz,)),
        initial_value=(
            bn11_l1_data
            if bn11_l1_data is not None
            else np.zeros(bn11_l1_wts_sz, dtype=np.int8)
        )
    )
    bn11_l2_wts = Buffer(
        _i8((bn11_l2_wts_sz,)),
        initial_value=(
            bn11_l2_data
            if bn11_l2_data is not None
            else np.zeros(bn11_l2_wts_sz, dtype=np.int8)
        )
    )
    bn11_l3_wts = Buffer(
        _i8((bn11_l3_wts_sz,)),
        initial_value=(
            bn11_l3_data
            if bn11_l3_data is not None
            else np.zeros(bn11_l3_wts_sz, dtype=np.int8)
        )
    )

    # bn11 L1: (in(14,1,112)i8, wts(37632)i8, out(14,1,336)u8, W, InC, OutC, scale)
    k_bn11_l1 = Kernel(
        "bn11_conv2dk1_relu_i8_ui8",
        "bn11_conv2dk1_fused_relu.o",
        [
            _i8((14, 1, 112)),
            _i8((37632,)),
            _u8((14, 1, 336)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn11 L2 DW stride-1:
    #   (top, mid, bot: u8(14,1,336), wts(3024)i8, out(14,1,336)u8,
    #    W, 1, C, kH, kW, border, scale, 0)  -- 13 args
    k_bn11_l2 = Kernel(
        "bn11_conv2dk3_dw_stride1_relu_ui8_ui8",
        "bn11_conv2dk3_dw.o",
        [
            _u8((14, 1, 336)),
            _u8((14, 1, 336)),
            _u8((14, 1, 336)),
            _i8((3024,)),
            _u8((14, 1, 336)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn11 L3 skip: actual kernel name from source is "bn11_conv2dk1_skip_ui8_i8_i8"
    # Signature from source external_func (9 args):
    #   (in(14,1,336)u8, wts(37632)i8, out(14,1,112)i8, skip(14,1,112)i8,
    #    W, InC, OutC, scale, scaleAdd)
    k_bn11_l3 = Kernel(
        "bn11_conv2dk1_skip_ui8_i8_i8",
        "bn11_conv2dk1_skip.o",
        [
            _u8((14, 1, 336)),
            _i8((37632,)),
            _i8((14, 1, 112)),
            _i8((14, 1, 112)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )

    bn11_of_12 = ObjectFifo(_u8((14, 1, 336)), depth=4, name="bn11_of_12")
    bn11_of_23 = ObjectFifo(_u8((14, 1, 336)), depth=2, name="bn11_of_23")
    act_bn11_out = ObjectFifo(_i8((14, 1, 112)), depth=2, name="act_bn11_out")

    def bn11_l1_fn(act_in, of_12, wts_buf, k_l1, sf1):
        for _ in range_(14):
            row_in = act_in.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, 14, 112, 336, sf1)
            act_in.release(1)
            of_12.release(1)

    def bn11_l2_fn(of_12, of_23, wts_buf, k_l2, sf2):
        # preamble: top row (border=0)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[0], rows[1], wts_buf, row_out, 14, 1, 336, 3, 3, 0, sf2, 0)
        of_23.release(1)
        # middle rows (border=1): b10_InH2 - 2 = 12
        for _ in range_(12):
            rows = of_12.acquire(3)
            row_out = of_23.acquire(1)
            k_l2(rows[0], rows[1], rows[2], wts_buf, row_out, 14, 1, 336, 3, 3, 1, sf2, 0)
            of_12.release(1)
            of_23.release(1)
        # postamble: bottom row (border=2)
        rows = of_12.acquire(2)
        row_out = of_23.acquire(1)
        k_l2(rows[0], rows[1], rows[1], wts_buf, row_out, 14, 1, 336, 3, 3, 2, sf2, 0)
        of_12.release(2)
        of_23.release(1)

    # bn11 L3 call from source:
    #   call(bn11_conv2dk1_skip,
    #        [elemIn, wts, elemOut, elementSkipsIn, W, C2, C3, scale, skipScale])
    def bn11_l3_fn(of_23, skip_in, act_out, wts_buf, k_l3, sf3, sfAdd):
        for _ in range_(14):
            row_23 = of_23.acquire(1)
            skip_row = skip_in.acquire(1)
            row_out = act_out.acquire(1)
            k_l3(row_23, wts_buf, row_out, skip_row, 14, 336, 112, sf3, sfAdd)
            of_23.release(1)
            skip_in.release(1)
            act_out.release(1)

    workers += [
        Worker(
            bn11_l1_fn,
            [
                act_bn10_out.cons(),
                bn11_of_12.prod(),
                bn11_l1_wts,
                k_bn11_l1,
                bn11_s1,
            ],
        ),
        Worker(
            bn11_l2_fn,
            [bn11_of_12.cons(), bn11_of_23.prod(), bn11_l2_wts, k_bn11_l2, bn11_s2],
        ),
        Worker(
            bn11_l3_fn,
            [
                bn11_of_23.cons(),
                bn11_skip_of.cons(),
                act_bn11_out.prod(),
                bn11_l3_wts,
                k_bn11_l3,
                bn11_s3,
                bn11_sAdd,
            ],
        ),
    ]

    # ---- bn12 (2-tile: L1 on tile1, interleaved DW-stride2 + 1x1 on tile2) ----
    # Source (aie2_bottleneckBStatic.py lines 959-1146) uses two separate kernels:
    #   bn12_conv2dk3_dw_stride2_relu_ui8_ui8  (DW stride-2, 13 args)
    #   bn12_conv2dk1_ui8_i8                   (1x1, 7 args)
    # interleaved per output row via of_act_bn12_2_3 (self-loop fifo, depth=1).
    # Weights concatenated: [dw_wts(3024) | pw_wts(26880)] = 29904 total.
    # DW output: (14,1,336) -> (7,1,336) u8 (stride-2 halves spatial).
    # Source link_with: DW="bn12_conv2dk3_dw_stride2.o", PW="bn12_conv2dk1_ui8.o"
    bn12_l1_wts_sz = 112 * 336    # 37632
    bn12_dw_wts_sz = 3 * 3 * 336  # 3024
    bn12_pw_wts_sz = 336 * 80     # 26880
    bn12_l23_wts_sz = bn12_dw_wts_sz + bn12_pw_wts_sz  # 29904

    bn12_l1_data = _load_weights(data_dir, "bn12_1_chain.txt")
    bn12_l23_data = _load_weights(data_dir, "bn12_2_3_chain.txt")

    bn12_l1_wts = Buffer(
        _i8((bn12_l1_wts_sz,)),
        initial_value=(
            bn12_l1_data
            if bn12_l1_data is not None
            else np.zeros(bn12_l1_wts_sz, dtype=np.int8)
        )
    )
    bn12_l23_wts = Buffer(
        _i8((bn12_l23_wts_sz,)),
        initial_value=(
            bn12_l23_data
            if bn12_l23_data is not None
            else np.zeros(bn12_l23_wts_sz, dtype=np.int8)
        )
    )

    # bn12 L1: (in(14,1,112)i8, wts(37632)i8, out(14,1,336)u8, W, InC, OutC, scale)
    k_bn12_l1 = Kernel(
        "bn12_conv2dk1_relu_i8_ui8",
        "bn12_conv2dk1_fused_relu.o",
        [
            _i8((14, 1, 112)),
            _i8((37632,)),
            _u8((14, 1, 336)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn12 DW stride-2 -- kernel name from source: "bn12_conv2dk3_dw_stride2_relu_ui8_ui8"
    # link_with from source: "bn12_conv2dk3_dw_stride2.o"
    # Output is (7,1,336) u8 (spatial halved by stride-2).
    # Args: (top, mid, bot: u8(14,1,336), wts(3024)i8, out(7,1,336)u8,
    #         inW, 1, C, kH, kW, border, scale, 0)  -- 13 args
    k_bn12_dw = Kernel(
        "bn12_conv2dk3_dw_stride2_relu_ui8_ui8",
        "bn12_conv2dk3_dw_stride2.o",
        [
            _u8((14, 1, 336)),
            _u8((14, 1, 336)),
            _u8((14, 1, 336)),
            _i8((3024,)),
            _u8((7, 1, 336)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )
    # bn12 PW 1x1 -- kernel name from source: "bn12_conv2dk1_ui8_i8"
    # link_with from source: "bn12_conv2dk1_ui8.o"
    # Args: (in(7,1,336)u8, wts(26880)i8, out(7,1,80)i8, W, InC, OutC, scale)
    k_bn12_pw = Kernel(
        "bn12_conv2dk1_ui8_i8",
        "bn12_conv2dk1_ui8.o",
        [
            _u8((7, 1, 336)),
            _i8((26880,)),
            _i8((7, 1, 80)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )

    bn12_of_12 = ObjectFifo(_u8((14, 1, 336)), depth=4, name="bn12_of_12")
    # Local intermediate fifo for DW -> PW handoff on tile2 (depth=1, mirrors self-loop)
    # Intermediate local buffer for DW->PW on the same tile.
    # In the placed dialect this is object_fifo(computeTile, computeTile, 1, ty) — a
    # self-loop. In IRON we use a Buffer on the compute tile instead, which is simpler.
    bn12_dw_tmp = Buffer(
        _u8((7, 1, 336)),
        name="bn12_dw_tmp",
    )
    act_bn12_out = ObjectFifo(_i8((7, 1, 80)), depth=2, name="act_bn12_out")

    def bn12_l1_fn(act_in, of_12, wts_buf, k_l1, sf1):
        for _ in range_(14):
            row_in = act_in.acquire(1)
            row_out = of_12.acquire(1)
            k_l1(row_in, wts_buf, row_out, 14, 112, 336, sf1)
            act_in.release(1)
            of_12.release(1)

    # bn12 tile2: interleave DW-stride2 and PW per output row.
    # dw_tmp is a tile-local Buffer (not an ObjectFifo) for DW->PW handoff.
    def bn12_l23_fn(of_12, dw_tmp, act_out, wts_buf, k_dw, k_pw, sf2, sf3):
        # preamble: top output row (border=0)
        rows = of_12.acquire(2)
        pw_out = act_out.acquire(1)
        k_dw(rows[0], rows[0], rows[1], wts_buf, dw_tmp, 14, 1, 336, 3, 3, 0, sf2, 0)
        of_12.release(1)
        k_pw(dw_tmp, wts_buf, pw_out, 7, 336, 80, sf3)
        act_out.release(1)
        # middle output rows (border=1): 5 iters
        for _ in range_(5):
            rows = of_12.acquire(3)
            pw_out = act_out.acquire(1)
            k_dw(rows[0], rows[1], rows[2], wts_buf, dw_tmp, 14, 1, 336, 3, 3, 1, sf2, 0)
            of_12.release(2)
            k_pw(dw_tmp, wts_buf, pw_out, 7, 336, 80, sf3)
            act_out.release(1)
        # postamble: last output row (border=1, release 3)
        rows = of_12.acquire(3)
        pw_out = act_out.acquire(1)
        k_dw(rows[0], rows[1], rows[2], wts_buf, dw_tmp, 14, 1, 336, 3, 3, 1, sf2, 0)
        of_12.release(3)
        k_pw(dw_tmp, wts_buf, pw_out, 7, 336, 80, sf3)
        act_out.release(1)

    workers += [
        Worker(
            bn12_l1_fn,
            [act_bn11_out.cons(), bn12_of_12.prod(), bn12_l1_wts, k_bn12_l1, bn12_s1],
        ),
        Worker(
            bn12_l23_fn,
            [
                bn12_of_12.cons(),
                bn12_dw_tmp,           # tile-local Buffer for DW->PW handoff
                act_bn12_out.prod(),
                bn12_l23_wts,
                k_bn12_dw,
                k_bn12_pw,
                bn12_s2,
                bn12_s3,
            ],
        ),
    ]

    return workers, act_bn12_out
