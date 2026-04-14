#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""MobileNet V3 — IRON API rewrite.

Replaces the placed-dialect implementation in aie2_mobilenet.py with the
high-level IRON API.  Computation is organized by bottleneck family rather
than by hardware tile column layout:

  regular.py   — bn0–bn9  (single compute tile per block)
  pipeline.py  — bn10–bn12 (one tile per layer)
  cascade.py   — bn13–bn14 (split-channel cascade stream, 5 tiles per block)

Scale factors are compile-time Python int constants loaded from
scale_factors_final.json and passed directly in Worker fn_args — no RTP
buffers or NpuWriteRTPOp calls are needed.

Usage:
    python3 aie2_mobilenet_iron.py > mobilenet_iron.mlir
"""

import json
import os
import sys
import numpy as np

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker, PersistentBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_

# Import bottleneck modules (new IRON organization)
import importlib.util, pathlib

_here = pathlib.Path(__file__).parent
sys.path.insert(0, str(_here))

from bottleneck.regular import regular_bottlenecks
from bottleneck.pipeline import pipeline_bottlenecks
from bottleneck.cascade import cascade_bottlenecks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
data_dir = os.path.join(os.path.dirname(__file__), "data") + "/"
scale_factor_file = "scale_factors_final.json"

with open(data_dir + scale_factor_file) as f:
    sf = json.load(f)

# Init conv
init_scaleFactor = sf.get("init_scaleFactor", 8)

# bn0–bn9 scale factors (36 total)
bn0_sf2      = sf.get("bn0_scaleFactor2",     9)
bn0_sf3      = sf.get("bn0_scaleFactor3",     8)
bn0_sfAdd    = sf.get("bn0_scaleFactorAdd",   2)
bn1_sf1      = sf.get("bn1_scaleFactor1",     8)
bn1_sf2      = sf.get("bn1_scaleFactor2",     8)
bn1_sf3      = sf.get("bn1_scaleFactor3",    11)
bn2_sf1      = sf.get("bn2_scaleFactor1",     8)
bn2_sf2      = sf.get("bn2_scaleFactor2",     8)
bn2_sf3      = sf.get("bn2_scaleFactor3",    11)
bn2_sfAdd    = sf.get("bn2_scaleFactorAdd",   0)
bn3_sf1      = sf.get("bn3_scaleFactor1",     8)
bn3_sf2      = sf.get("bn3_scaleFactor2",     8)
bn3_sf3      = sf.get("bn3_scaleFactor3",    11)
bn4_sf1      = sf.get("bn4_scaleFactor1",     8)
bn4_sf2      = sf.get("bn4_scaleFactor2",     8)
bn4_sf3      = sf.get("bn4_scaleFactor3",    11)
bn4_sfAdd    = sf.get("bn4_scaleFactorAdd",   0)
bn5_sf1      = sf.get("bn5_scaleFactor1",     8)
bn5_sf2      = sf.get("bn5_scaleFactor2",     8)
bn5_sf3      = sf.get("bn5_scaleFactor3",    11)
bn5_sfAdd    = sf.get("bn5_scaleFactorAdd",   0)
bn6_sf1      = sf.get("bn6_scaleFactor1",     8)
bn6_sf2      = sf.get("bn6_scaleFactor2",     8)
bn6_sf3      = sf.get("bn6_scaleFactor3",    11)
bn6_sfAdd    = sf.get("bn6_scaleFactorAdd",   0)
bn7_sf1      = sf.get("bn7_scaleFactor1",     9)
bn7_sf2      = sf.get("bn7_scaleFactor2",     8)
bn7_sf3      = sf.get("bn7_scaleFactor3",    11)
bn8_sf1      = sf.get("bn8_scaleFactor1",     9)
bn8_sf2      = sf.get("bn8_scaleFactor2",     8)
bn8_sf3      = sf.get("bn8_scaleFactor3",    11)
bn8_sfAdd    = sf.get("bn8_scaleFactorAdd",   0)
bn9_sf1      = sf.get("bn9_scaleFactor1",     9)
bn9_sf2      = sf.get("bn9_scaleFactor2",     8)
bn9_sf3      = sf.get("bn9_scaleFactor3",    11)
bn9_sfAdd    = sf.get("bn9_scaleFactorAdd",   0)

# bn10–bn12 scale factors (10 total)
bn10_sf1     = sf.get("bn10_scaleFactor1",   10)
bn10_sf2     = sf.get("bn10_scaleFactor2",    7)
bn10_sf3     = sf.get("bn10_scaleFactor3",    9)
bn11_sf1     = sf.get("bn11_scaleFactor1",    9)
bn11_sf2     = sf.get("bn11_scaleFactor2",    8)
bn11_sf3     = sf.get("bn11_scaleFactor3",   12)
bn11_sfAdd   = sf.get("bn11_scaleFactorAdd",  1)
bn12_sf1     = sf.get("bn12_scaleFactor1",    8)
bn12_sf2     = sf.get("bn12_scaleFactor2",    8)
bn12_sf3     = sf.get("bn12_scaleFactor3",    9)

# bn13–bn14 scale factors (8 total)
bn13_sf1     = sf.get("bn13_scaleFactor1",   10)
bn13_sf2     = sf.get("bn13_scaleFactor2",    7)
bn13_sf3     = sf.get("bn13_scaleFactor3",    9)
bn13_sfAdd   = sf.get("bn13_scaleFactorAdd",  1)
bn14_sf1     = sf.get("bn14_scaleFactor1",    9)
bn14_sf2     = sf.get("bn14_scaleFactor2",    8)
bn14_sf3     = sf.get("bn14_scaleFactor3",   12)
bn14_sfAdd   = sf.get("bn14_scaleFactorAdd",  1)

# Post-processing scale factors
post_sf      = sf.get("post_scaleFactor",     8)
post_fc1_sf  = sf.get("post_FC1_scaleFactor", 9)
post_fc2_sf  = sf.get("post_FC2_scaleFactor", 9)

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------
tensorInW, tensorInH, tensorInC = 224, 224, 8
init_OutC = 16          # init conv output channels
init_OutW = 112         # stride-2 output width
init_OutH = 112

# Post-processing dimensions
post_L1_InW   = 7
post_L1_InH   = 7
post_L1_InC   = 80
post_L1_OutC  = 960     # expand 1x1
post_L1_OutW  = 1       # global avg pool -> 1x1
post_L1_OutH  = 1

post_L2_InC   = 1280    # after avg pool + 1x1
post_L2_OutC  = 1280
post_wts_per_tile = post_L2_OutC * (post_L1_OutC // 4)  # split across 4 tiles

# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
def _i8(shape):  return np.ndarray[shape, np.dtype[np.int8]]
def _u8(shape):  return np.ndarray[shape, np.dtype[np.uint8]]
def _i32():      return np.int32


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
def mobilenet_iron():
    # ------------------------------------------------------------------
    # Input / output tensor types (used in runtime sequence)
    # ------------------------------------------------------------------
    in_ty  = np.ndarray[(tensorInW * tensorInH * tensorInC,), np.dtype[np.int8]]
    out_ty = np.ndarray[(post_L1_OutW * post_L1_OutH * post_L2_OutC,), np.dtype[np.uint8]]

    # ------------------------------------------------------------------
    # Init conv boundary fifos (declared here; owned by orchestrator)
    # act_in:      input activations (224,1,8) int8
    # act_init_out: init conv output (112,1,16) uint8
    # ------------------------------------------------------------------
    act_in = ObjectFifo(
        np.ndarray[(tensorInW, 1, tensorInC), np.dtype[np.int8]],
        depth=5,   # Consumer depth=5 to allow 3-row sliding window in init conv
        name="act_in",
    )
    act_init_out = ObjectFifo(
        np.ndarray[(init_OutW, 1, init_OutC), np.dtype[np.uint8]],
        depth=5,
        name="act_init_out",
    )

    # ------------------------------------------------------------------
    # Init conv weights (static buffer on compute tile)
    # 3x3 stride-2 conv: InC=8, OutC=16 -> wts = 3*3*8*16 = 1152
    # ------------------------------------------------------------------
    init_wts_sz = 3 * 3 * tensorInC * init_OutC  # 1152
    init_wts_data = None
    init_wts_path = data_dir + "init_chain.txt"
    if os.path.exists(init_wts_path):
        init_wts_data = np.fromfile(init_wts_path, sep=",", dtype=np.int8)
    if init_wts_data is None:
        init_wts_data = np.zeros(init_wts_sz, dtype=np.int8)

    init_wts = Buffer(
        _i8((init_wts_sz,)),
        initial_value=init_wts_data,
        name="init_wts",
    )

    # ------------------------------------------------------------------
    # Init conv kernel: 3x3 stride-2, int8 in, uint8 out
    # fn signature from source: (in0, in0, in1, wts, out, W, InC, OutC,
    #                            kW, kH, border_top, scale, border_bottom, padding)
    # ------------------------------------------------------------------
    k_init = Kernel(
        "conv2dk3_stride2_i8",
        "init_conv2dk3_stride2.o",
        [_i8((tensorInW, 1, tensorInC)),
         _i8((tensorInW, 1, tensorInC)),
         _i8((tensorInW, 1, tensorInC)),
         _i8((init_wts_sz,)),
         _u8((init_OutW, 1, init_OutC)),
         _i32(), _i32(), _i32(), _i32(), _i32(), _i32(), _i32(), _i32(), _i32()],
    )

    def init_fn(act_in, act_out, wts, k, inW, inH, inC, outW, outH, outC, sf):
        # Preamble: top row (pad above = row 0)
        rows = act_in.acquire(2)
        row_out = act_out.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out, inW, inC, outC, 3, 3, 0, sf, 0, 0)
        act_in.release(1)
        act_out.release(1)
        # Middle rows (stride-2: 2 input rows per output row)
        for _ in range_(outH - 2):
            rows = act_in.acquire(3)
            row_out = act_out.acquire(1)
            k(rows[0], rows[1], rows[2], wts, row_out, inW, inC, outC, 3, 3, 1, sf, 0, 0)
            act_in.release(2)
            act_out.release(1)
        # Postamble
        rows = act_in.acquire(2)
        row_out = act_out.acquire(1)
        k(rows[0], rows[1], rows[1], wts, row_out, inW, inC, outC, 3, 3, 2, sf, 0, 0)
        act_in.release(2)
        act_out.release(1)

    w_init = Worker(
        init_fn,
        fn_args=[
            act_in.cons(), act_init_out.prod(), init_wts, k_init,
            tensorInW, tensorInH, tensorInC,
            init_OutW, init_OutH, init_OutC,
            init_scaleFactor,
        ],
    )

    # ------------------------------------------------------------------
    # Bottleneck blocks
    # ------------------------------------------------------------------
    # Regular family: bn0–bn9
    a_workers, act_bn9_out = regular_bottlenecks(
        act_init_out,
        bn0_sf2, bn0_sf3,      # bn0: 2-layer block, no sf1 (no expand 1x1)
        bn1_sf1, bn1_sf2, bn1_sf3,
        bn2_sf1, bn2_sf2, bn2_sf3, bn2_sfAdd,
        bn3_sf1, bn3_sf2, bn3_sf3,
        bn4_sf1, bn4_sf2, bn4_sf3, bn4_sfAdd,
        bn5_sf1, bn5_sf2, bn5_sf3, bn5_sfAdd,
        bn6_sf1, bn6_sf2, bn6_sf3, bn6_sfAdd,
        bn7_sf1, bn7_sf2, bn7_sf3,
        bn8_sf1, bn8_sf2, bn8_sf3, bn8_sfAdd,
        bn9_sf1, bn9_sf2, bn9_sf3, bn9_sfAdd,
        data_dir=data_dir,
    )

    # Pipeline family: bn10–bn12
    b_workers, act_bn12_out = pipeline_bottlenecks(
        act_bn9_out,
        bn10_sf1, bn10_sf2, bn10_sf3,
        bn11_sf1, bn11_sf2, bn11_sf3, bn11_sfAdd,
        bn12_sf1, bn12_sf2, bn12_sf3,
        data_dir=data_dir,
    )

    # Cascade family: bn13–bn14
    c_workers, act_bn14_out, wts_fifos, cascade_pairs = cascade_bottlenecks(
        act_bn12_out,
        bn13_sf1, bn13_sf2, bn13_sf3, bn13_sfAdd,
        bn14_sf1, bn14_sf2, bn14_sf3, bn14_sfAdd,
        data_dir=data_dir,
    )

    # ------------------------------------------------------------------
    # Post-processing L1: avg pool + expand 1x1 conv
    # Input:  (7,1,80) int8   Output: (1,1,960) int8
    # ------------------------------------------------------------------
    # Post-L1 weights: 960*80 = 76800 bytes — too large for compute tile.
    # Original: stored on MemTile(4,1), streamed 1/8 at a time to PostL1Tile(6,4).
    # PersistentBuffer replicates this: MemTile holds full weights, DMA streams
    # post_l1_wts_chunk bytes at a time to the small recv buffer on compute tile.
    PostOutputSplit = 8                             # split output channels (original: 8)
    PostRepeatChannels = post_L1_InH               # = 7 (original: math.floor(post_L1_InH))
    post_l1_wts_full_sz = post_L1_OutC * post_L1_InC        # 76800 bytes on MemTile
    post_l1_wts_chunk = post_l1_wts_full_sz // PostOutputSplit  # 9600 bytes per chunk on compute

    post_l1_wts_path = data_dir + "post_conv_chain.txt"
    post_l1_wts_data = None
    if os.path.exists(post_l1_wts_path):
        post_l1_wts_data = np.fromfile(post_l1_wts_path, sep=",", dtype=np.int8)
    if post_l1_wts_data is None:
        post_l1_wts_data = np.zeros(post_l1_wts_full_sz, dtype=np.int8)

    # Original: PostL1Tile = tile(6,4), MemTile41 = tile(4,1), flow: MemTile41→PostL1Tile
    post_l1_pb = PersistentBuffer(
        obj_type=_i8((post_l1_wts_full_sz,)),
        initial_value=post_l1_wts_data,
        name="post_l1_wts",
        recv_type=_i8((post_l1_wts_chunk,)),
        repeat_count=PostRepeatChannels,
        memtile_placement=Tile(4, 1),    # MemTile41 in original
        compute_placement=Tile(6, 4),    # PostL1Tile in original
        mm2s_channel=0,
        s2mm_channel=0,
    )

    act_post_l1_out = ObjectFifo(
        np.ndarray[(post_L1_OutW, 1, post_L2_InC), np.dtype[np.int8]],
        depth=2,
        name="act_post_l1_out",
    )

    k_post_l1 = Kernel(
        "post_fused_conv2dk1_i8_avg_pool",
        "post_conv2dk1_avg_pool.o",
        [_i8((post_L1_InW, 1, post_L1_InC)),
         _i8((post_l1_wts_chunk,)),   # 9600 bytes = 1/PostOutputSplit of full weights
         _i8((post_L1_OutW, 1, post_L2_InC)),
         _i32(), _i32(), _i32(), _i32(), _i32(), _i32()],
    )

    # Post-L1 worker: placed on Tile(6,4) = PostL1Tile (matches original)
    def post_l1_fn(act_in, act_out, wts_pb, k, inW, inH, inC, outC, outC_padd, sf):
        # One full output frame: acquire output, loop over rows
        elem_out = act_out.acquire(1)
        for _ in range_(inH):
            elem_in = act_in.acquire(1)
            for wi in range_(PostOutputSplit):
                wts_chunk = wts_pb.acquire(1)  # wait for MemTile DMA delivery
                k(elem_in, wts_chunk, elem_out, inW, inC, outC, outC_padd, sf, 0)
                wts_pb.release(1)              # signal DMA to send next chunk
            act_in.release(1)
        act_out.release(1)

    w_post_l1 = Worker(
        post_l1_fn,
        fn_args=[
            act_bn14_out.cons(), act_post_l1_out.prod(), post_l1_pb.cons(), k_post_l1,
            post_L1_InW, post_L1_InH, post_L1_InC, post_L2_InC, post_L2_InC, post_sf,
        ],
        placement=Tile(6, 4),    # PostL1Tile = tile(6,4) in original
    )

    # ------------------------------------------------------------------
    # Post-processing L2: 4-tile FC (split output channels)
    # Input:  (1,1,1280) int8   Output: (1,1,1280) uint8 (4 tiles, joined)
    # ------------------------------------------------------------------
    # Weight files for 4 tiles (FC1 + FC2 interleaved)
    fc_wts_filenames = [
        ("FC1_0_chain.txt", "FC2_0_chain.txt"),
        ("FC1_1_chain.txt", "FC2_1_chain.txt"),
        ("FC1_2_chain.txt", "FC2_2_chain.txt"),
        ("FC1_3_chain.txt", "FC2_3_chain.txt"),
    ]
    n_fc_tiles = 4
    fc_out_per_tile = post_L2_OutC // n_fc_tiles   # 1280/4 = 320

    # Output fifo: all 4 FC tiles join their results here via MemTile
    act_out_of = ObjectFifo(
        np.ndarray[(post_L2_OutC,), np.dtype[np.uint8]],
        depth=2,
        name="act_post_out",
    )
    # Split the output fifo into 4 segments, one per FC tile
    act_post_l2_tiles = act_out_of.prod().join(
        offsets=[i * fc_out_per_tile for i in range(n_fc_tiles)],
        depths=[2] * n_fc_tiles,
        obj_types=[np.ndarray[(fc_out_per_tile,), np.dtype[np.uint8]]] * n_fc_tiles,
        names=[f"act_post_l2_tile{i}" for i in range(n_fc_tiles)],
    )

    # FC weights: ping-pong design matching original.
    # Each PostL2Tile gets FC1 (409,600 bytes) and FC2 (409,600 bytes) on separate MemTiles.
    # Compute tile receive buffer: 10,240 bytes = 1/40 of one FC half at a time.
    # Original assignments:
    #   PostL2Tile_1(6,3): FC1 on MemTile(0,1), FC2 on MemTile(1,1), flow from MemTile(1,1)
    #   PostL2Tile_2(7,4): FC1 on MemTile(2,1), FC2 on MemTile(3,1), flow from MemTile(3,1)
    #   PostL2Tile_3(7,3): FC1 on MemTile(4,1), FC2 on MemTile(5,1), flow from MemTile(5,1)
    #   PostL2Tile_4(7,2): FC1 on MemTile(6,1), FC2 on MemTile(7,1), flow from MemTile(7,1)
    PostOutputSplitL2 = 40
    fc_full_per_tile = post_L2_InC * fc_out_per_tile          # 409,600 bytes per FC half
    fc_recv_per_tile = fc_full_per_tile // PostOutputSplitL2   # 10,240 bytes on compute tile

    fc1_memtiles = [Tile(0,1), Tile(2,1), Tile(4,1), Tile(6,1)]   # FC1 MemTiles
    fc2_memtiles = [Tile(1,1), Tile(3,1), Tile(5,1), Tile(7,1)]   # FC2 MemTiles (serve DMA)
    fc_comptiles = [Tile(6,3), Tile(7,4), Tile(7,3), Tile(7,2)]   # PostL2 compute tiles

    k_post_l2 = Kernel(
        "post_conv2dk1_i8_ui8",
        "post_conv2dk1.o",
        [_i8((1, 1, post_L2_InC)),
         _i8((fc_recv_per_tile,)),
         _u8((fc_out_per_tile,)),
         _i32(), _i32(), _i32(), _i32(), _i32()],
    )

    post_l2_workers = []
    for i, (fc1_f, fc2_f) in enumerate(fc_wts_filenames):
        fc1_data = np.zeros(fc_full_per_tile, dtype=np.int8)
        fc2_data = np.zeros(fc_full_per_tile, dtype=np.int8)
        if os.path.exists(data_dir + fc1_f):
            fc1_data = np.fromfile(data_dir + fc1_f, sep=",", dtype=np.int8)
        if os.path.exists(data_dir + fc2_f):
            fc2_data = np.fromfile(data_dir + fc2_f, sep=",", dtype=np.int8)

        # Single PersistentBuffer with ping-pong: FC1 on fc2_memtile, FC2 as ping-pong
        # on fc1_memtile (adjacent MemTile). DMA BD chain alternates: FC2→FC1→FC2→...
        # This matches original's single DMA flow with two-BD chain from MemTile(N,1).
        fc_pb = PersistentBuffer(
            obj_type=_i8((fc_full_per_tile,)), initial_value=fc2_data,
            name=f"post_l2_fc2_wts_{i}",
            recv_type=_i8((fc_recv_per_tile,)), repeat_count=PostOutputSplitL2,
            memtile_placement=fc2_memtiles[i], compute_placement=fc_comptiles[i],
            mm2s_channel=0, s2mm_channel=1,
            ping_pong_buf=(_i8((fc_full_per_tile,)), fc1_data, f"post_l2_fc1_wts_{i}"),
            ping_pong_memtile=fc1_memtiles[i],  # FC1 on adjacent MemTile
        )

        def post_l2_fn(act_in, act_out, wts_h, k, inC, outC, sf1, sf2,
                       n_splits=PostOutputSplitL2):
            elem_in = act_in.acquire(1)
            elem_out = act_out.acquire(1)
            # FC2 pass (first in DMA chain)
            for _ in range_(n_splits):
                wts = wts_h.acquire(1)
                k(elem_in, wts, elem_out, 1, inC, outC, outC, sf2)
                wts_h.release(1)
            # FC1 pass (ping-pong second)
            for _ in range_(n_splits):
                wts = wts_h.acquire(1)
                k(elem_in, wts, elem_out, 1, inC, outC, outC, sf1)
                wts_h.release(1)
            act_in.release(1)
            act_out.release(1)

        w = Worker(
            post_l2_fn,
            fn_args=[
                act_post_l1_out.cons(), act_post_l2_tiles[i].prod(),
                fc_pb.cons(), k_post_l2,
                post_L2_InC, fc_out_per_tile, post_fc1_sf, post_fc2_sf,
            ],
            placement=fc_comptiles[i],
        )
        post_l2_workers.append(w)

    # ------------------------------------------------------------------
    # Collect all workers
    # ------------------------------------------------------------------
    all_workers = (
        [w_init]
        + a_workers
        + b_workers
        + c_workers
        + [w_post_l1]
        + post_l2_workers
    )

    # ------------------------------------------------------------------
    # Cascade weight types (for runtime sequence DMA)
    # ------------------------------------------------------------------
    _l1_full_wts_sz = 80 * 960       # 76800
    _l3_full_wts_sz = 960 * 80       # 76800
    bn13_l1_wts_ty = np.ndarray[(_l1_full_wts_sz,), np.dtype[np.int8]]
    bn13_l3_wts_ty = np.ndarray[(_l3_full_wts_sz,), np.dtype[np.int8]]
    bn14_l1_wts_ty = np.ndarray[(_l1_full_wts_sz,), np.dtype[np.int8]]
    bn14_l3_wts_ty = np.ndarray[(_l3_full_wts_sz,), np.dtype[np.int8]]

    # ------------------------------------------------------------------
    # Runtime sequence
    # ------------------------------------------------------------------
    rt = Runtime()
    with rt.sequence(in_ty,
                     bn13_l1_wts_ty, bn13_l3_wts_ty,
                     bn14_l1_wts_ty, bn14_l3_wts_ty,
                     out_ty) as (inp,
                                  wts_bn13_l1, wts_bn13_l3,
                                  wts_bn14_l1, wts_bn14_l3,
                                  out):
        rt.start(*all_workers)

        # Register cascade stream connections
        for src, dst in cascade_pairs:
            rt.cascade_flow(src, dst)

        # Data movement — activations in, output out
        # Shim tile placement matches original design to distribute DMA BD load:
        #   act_in:       ShimTile00 (col 0)
        #   bn13 L1 wts:  ShimTile40 (col 4)
        #   bn13 L3 wts:  ShimTile50 (col 5)
        #   bn14 L1 wts:  ShimTile60 (col 6)
        #   bn14 L3 wts:  ShimTile70 (col 7)
        #   act_out:      ShimTile70 (col 7, shared with bn14 L3)
        rt.fill(act_in.prod(), inp, placement=Tile(0, 0))

        # Cascade weight DMA: host fills the full-weight fifos,
        # MemTile splits them to put/get tiles via cons().split()
        rt.fill(wts_fifos[0].prod(), wts_bn13_l1, placement=Tile(4, 0))  # bn13 L1 → ShimTile40
        rt.fill(wts_fifos[1].prod(), wts_bn13_l3, placement=Tile(5, 0))  # bn13 L3 → ShimTile50
        rt.fill(wts_fifos[2].prod(), wts_bn14_l1, placement=Tile(6, 0))  # bn14 L1 → ShimTile60
        rt.fill(wts_fifos[3].prod(), wts_bn14_l3, placement=Tile(7, 0))  # bn14 L3 → ShimTile70

        rt.drain(act_out_of.cons(), out, wait=True, placement=Tile(7, 0))

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":
    module = mobilenet_iron()
    print(module)
