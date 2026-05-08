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

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from lowlevel_dma import StaticWeightStream
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


# JSON layout: {"BN<n>": {"conv1x1_1": int, "conv3x3": int, "conv1x1_2": int, "skip_add": int}, ...}
# Mapping to legacy iron names: _sf1 -> conv1x1_1, _sf2 -> conv3x3, _sf3 -> conv1x1_2, _sfAdd -> skip_add.
# Use direct dict access (no .get fallbacks) so a missing key fails loud rather than silently
# substituting a wrong default — that bug just cost us a multi-hour debug.
def _bn(n, key):
    return sf[f"BN{n}"][key]


init_scaleFactor = sf["INIT"]["conv3x3"]

# bn0 (no conv1x1_1: it's the first stage with no 1x1 reduction before the dw)
bn0_sf2 = _bn(0, "conv3x3")
bn0_sf3 = _bn(0, "conv1x1_2")
bn0_sfAdd = _bn(0, "skip_add")

bn1_sf1 = _bn(1, "conv1x1_1")
bn1_sf2 = _bn(1, "conv3x3")
bn1_sf3 = _bn(1, "conv1x1_2")
bn2_sf1 = _bn(2, "conv1x1_1")
bn2_sf2 = _bn(2, "conv3x3")
bn2_sf3 = _bn(2, "conv1x1_2")
bn2_sfAdd = _bn(2, "skip_add")
bn3_sf1 = _bn(3, "conv1x1_1")
bn3_sf2 = _bn(3, "conv3x3")
bn3_sf3 = _bn(3, "conv1x1_2")
bn4_sf1 = _bn(4, "conv1x1_1")
bn4_sf2 = _bn(4, "conv3x3")
bn4_sf3 = _bn(4, "conv1x1_2")
bn4_sfAdd = _bn(4, "skip_add")
bn5_sf1 = _bn(5, "conv1x1_1")
bn5_sf2 = _bn(5, "conv3x3")
bn5_sf3 = _bn(5, "conv1x1_2")
bn5_sfAdd = _bn(5, "skip_add")
bn6_sf1 = _bn(6, "conv1x1_1")
bn6_sf2 = _bn(6, "conv3x3")
bn6_sf3 = _bn(6, "conv1x1_2")
bn6_sfAdd = _bn(6, "skip_add")
bn7_sf1 = _bn(7, "conv1x1_1")
bn7_sf2 = _bn(7, "conv3x3")
bn7_sf3 = _bn(7, "conv1x1_2")
bn7_sfAdd = _bn(7, "skip_add")
bn8_sf1 = _bn(8, "conv1x1_1")
bn8_sf2 = _bn(8, "conv3x3")
bn8_sf3 = _bn(8, "conv1x1_2")
bn8_sfAdd = _bn(8, "skip_add")
bn9_sf1 = _bn(9, "conv1x1_1")
bn9_sf2 = _bn(9, "conv3x3")
bn9_sf3 = _bn(9, "conv1x1_2")
bn9_sfAdd = _bn(9, "skip_add")

bn10_sf1 = _bn(10, "conv1x1_1")
bn10_sf2 = _bn(10, "conv3x3")
bn10_sf3 = _bn(10, "conv1x1_2")
bn11_sf1 = _bn(11, "conv1x1_1")
bn11_sf2 = _bn(11, "conv3x3")
bn11_sf3 = _bn(11, "conv1x1_2")
bn11_sfAdd = _bn(11, "skip_add")
bn12_sf1 = _bn(12, "conv1x1_1")
bn12_sf2 = _bn(12, "conv3x3")
bn12_sf3 = _bn(12, "conv1x1_2")

bn13_sf1 = _bn(13, "conv1x1_1")
bn13_sf2 = _bn(13, "conv3x3")
bn13_sf3 = _bn(13, "conv1x1_2")
bn13_sfAdd = _bn(13, "skip_add")
bn14_sf1 = _bn(14, "conv1x1_1")
bn14_sf2 = _bn(14, "conv3x3")
bn14_sf3 = _bn(14, "conv1x1_2")
bn14_sfAdd = _bn(14, "skip_add")

post_sf = sf["POST"]["conv1x1_1"]
post_fc1_sf = sf["POST"]["FC1"]
post_fc2_sf = sf["POST"]["FC2"]

# ---------------------------------------------------------------------------
# Network dimensions
# ---------------------------------------------------------------------------
tensorInW, tensorInH, tensorInC = 224, 224, 8
init_OutC = 16  # init conv output channels
init_OutW = 112  # stride-2 output width
init_OutH = 112

# Post-processing dimensions
post_L1_InW = 7
post_L1_InH = 7
post_L1_InC = 80
post_L1_OutC = 960  # expand 1x1
post_L1_OutW = 1  # global avg pool -> 1x1
post_L1_OutH = 1

post_L2_InC = 1280  # after avg pool + 1x1
post_L2_OutC = 1280
post_wts_per_tile = post_L2_OutC * (post_L1_OutC // 4)  # split across 4 tiles


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------
def _i8(shape):
    return np.ndarray[shape, np.dtype[np.int8]]


def _u8(shape):
    return np.ndarray[shape, np.dtype[np.uint8]]


def _i32():
    return np.int32


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
def mobilenet_iron():
    # ------------------------------------------------------------------
    # Input / output tensor types (used in runtime sequence)
    # ------------------------------------------------------------------
    in_ty = np.ndarray[(tensorInW * tensorInH * tensorInC,), np.dtype[np.int8]]
    out_ty = np.ndarray[
        (post_L1_OutW * post_L1_OutH * post_L2_OutC,), np.dtype[np.uint16]
    ]

    # ------------------------------------------------------------------
    # Init conv boundary fifos (declared here; owned by orchestrator)
    # act_in:      input activations (224,1,8) int8
    # act_init_out: init conv output (112,1,16) uint8
    # ------------------------------------------------------------------
    act_in = ObjectFifo(
        np.ndarray[(tensorInW, 1, tensorInC), np.dtype[np.int8]],
        depth=5,  # Consumer depth=5 to allow 3-row sliding window in init conv
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
        "init_conv2dk3.o",
        [
            _i8((tensorInW, 1, tensorInC)),
            _i8((tensorInW, 1, tensorInC)),
            _i8((tensorInW, 1, tensorInC)),
            _i8((init_wts_sz,)),
            _u8((init_OutW, 1, init_OutC)),
            _i32(),
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

    def init_fn(act_in, act_out, wts, k, inW, inH, inC, outW, outH, outC, sf):
        # Match lowlevel initConv exactly: preamble + (outH - 1) middle iters,
        # then a trailing release(1) to drain the final input row. The earlier
        # iron postamble call (index=2) would leave one shim BD un-released and
        # caused ERT_CMD_STATE_TIMEOUT.
        rows = act_in.acquire(2)
        row_out = act_out.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out, inW, inC, outC, 3, 3, 0, sf, 0, 0)
        act_in.release(1)
        act_out.release(1)
        for _ in range_(outH - 1):
            rows = act_in.acquire(3)
            row_out = act_out.acquire(1)
            k(
                rows[0],
                rows[1],
                rows[2],
                wts,
                row_out,
                inW,
                inC,
                outC,
                3,
                3,
                1,
                sf,
                0,
                0,
            )
            act_in.release(2)
            act_out.release(1)
        act_in.release(1)

    w_init = Worker(
        init_fn,
        fn_args=[
            act_in.cons(depth=5),
            act_init_out.prod(depth=5),
            init_wts,
            k_init,
            tensorInW,
            tensorInH,
            tensorInC,
            init_OutW,
            init_OutH,
            init_OutC,
            init_scaleFactor,
        ],
        tile=Tile(0, 2),  # original: init_tile = tile(0, 2)
    )

    # ------------------------------------------------------------------
    # Bottleneck blocks
    # ------------------------------------------------------------------
    # Regular family: bn0–bn9
    a_workers, act_bn9_out = regular_bottlenecks(
        act_init_out,
        bn0_sf2,
        bn0_sf3,
        bn0_sfAdd,  # bn0: 2-layer block, no sf1 (no expand 1x1)
        bn1_sf1,
        bn1_sf2,
        bn1_sf3,
        bn2_sf1,
        bn2_sf2,
        bn2_sf3,
        bn2_sfAdd,
        bn3_sf1,
        bn3_sf2,
        bn3_sf3,
        bn4_sf1,
        bn4_sf2,
        bn4_sf3,
        bn4_sfAdd,
        bn5_sf1,
        bn5_sf2,
        bn5_sf3,
        bn5_sfAdd,
        bn6_sf1,
        bn6_sf2,
        bn6_sf3,
        # bn6 is stride-2 (no residual skip) — regular.py signature does not
        # take bn6_scaleAdd. Including it here would shift every subsequent
        # scale-factor arg by one (caused bn7 relu/dw/skip to read wrong sf).
        bn7_sf1,
        bn7_sf2,
        bn7_sf3,
        bn7_sfAdd,
        bn8_sf1,
        bn8_sf2,
        bn8_sf3,
        bn8_sfAdd,
        bn9_sf1,
        bn9_sf2,
        bn9_sf3,
        bn9_sfAdd,
        data_dir=data_dir,
    )

    # Pipeline family: bn10–bn12
    b_workers, act_bn12_out = pipeline_bottlenecks(
        act_bn9_out,
        bn10_sf1,
        bn10_sf2,
        bn10_sf3,
        bn11_sf1,
        bn11_sf2,
        bn11_sf3,
        bn11_sfAdd,
        bn12_sf1,
        bn12_sf2,
        bn12_sf3,
        data_dir=data_dir,
    )

    # Cascade family: bn13–bn14
    c_workers, act_bn14_out, wts_fifos = cascade_bottlenecks(
        act_bn12_out,
        bn13_sf1,
        bn13_sf2,
        bn13_sf3,
        bn13_sfAdd,
        bn14_sf1,
        bn14_sf2,
        bn14_sf3,
        bn14_sfAdd,
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
    PostOutputSplit = 8  # split output channels (original: 8)
    PostRepeatChannels = post_L1_InH  # = 7 (original: math.floor(post_L1_InH))
    post_l1_wts_full_sz = post_L1_OutC * post_L1_InC  # 76800 bytes on MemTile
    post_l1_wts_chunk = (
        post_l1_wts_full_sz // PostOutputSplit
    )  # 9600 bytes per chunk on compute

    post_l1_wts_path = data_dir + "post_conv_chain.txt"
    post_l1_wts_data = None
    if os.path.exists(post_l1_wts_path):
        post_l1_wts_data = np.fromfile(post_l1_wts_path, sep=",", dtype=np.int8)
    if post_l1_wts_data is None:
        post_l1_wts_data = np.zeros(post_l1_wts_full_sz, dtype=np.int8)

    # Original: PostL1Tile = tile(6,4), MemTile41 = tile(4,1), flow: MemTile41→PostL1Tile
    # Lock IDs matching original: MemTile41 uses lock_id=2,3; PostL1Tile uses lock_id=0,1
    post_l1_pb = StaticWeightStream(
        obj_type=_i8((post_l1_wts_full_sz,)),
        initial_value=post_l1_wts_data,
        name="post_l1_wts",
        recv_type=_i8((post_l1_wts_chunk,)),
        repeat_count=PostRepeatChannels,
        memtile_placement=Tile(4, 1),
        compute_placement=Tile(6, 4),
        mm2s_channel=0,
        s2mm_channel=0,
        mem_lock_id=2,  # MemTile41: lock_id=2 (prod), lock_id=3 (cons)
        comp_lock_id=0,  # PostL1Tile: lock_id=0 (prod), lock_id=1 (cons)
    )

    # Match original: round-trip avgpool output through L3 (DDR) so it can be
    # re-broadcast to all 4 PostL2 FC tiles. A single compute->4-compute fan-out
    # exceeds stream-switch routing capacity from tile(6,4); the original design
    # uses shim DMAs to drain to DDR via shim(3,0) and re-fill from shim(4,0).
    # Element type is uint16 — kernel `post_conv2dk1_relu_xy_pool_padded_i8_ui8.o`
    # writes 2 bytes per output channel; declaring i8 here would halve the DMA
    # transfer size and deadlock the consumer kernel which expects 2560 B.
    _post_l1_out_ty = np.ndarray[(post_L2_InC,), np.dtype[np.uint16]]
    act_out_post_avgpool_shim = ObjectFifo(
        _post_l1_out_ty,
        depth=2,
        name="act_out_post_avgpool_shim",
    )
    act_out_post_shim_FC = ObjectFifo(
        _post_l1_out_ty,
        depth=2,
        name="act_out_post_shim_FC",
    )

    k_post_l1 = Kernel(
        "conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8",
        "post_conv2dk1_relu_xy_pool_padded_i8_ui8.o",
        [
            _i8((post_L1_InW, 1, post_L1_InC)),
            _i8((post_l1_wts_chunk,)),
            np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
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

    # Post-L1 worker: placed on Tile(6,4) = PostL1Tile (matches original)
    def post_l1_fn(
        act_in,
        act_out,
        wts_pb,
        k,
        inW,
        inH,
        inC,
        outC,
        outC_padd,
        sf,
        n_splits=PostOutputSplit,
    ):
        # One full output frame: acquire output, loop over rows then weight splits
        elem_out = act_out.acquire(1)
        for yi in range_(inH):
            elem_in = act_in.acquire(1)
            for wi in range_(n_splits):
                wts_chunk = wts_pb.acquire(1)
                k(
                    elem_in,
                    wts_chunk,
                    elem_out,
                    inW,
                    inC,
                    outC,
                    outC_padd,
                    sf,
                    yi,  # yIndex (was hardcoded 0 — kernel kept overwriting row 0)
                    n_splits,
                    wi,  # WeightIndex (was hardcoded 0 — kernel kept the same weight chunk)
                )
                wts_pb.release(1)
            act_in.release(1)
        act_out.release(1)

    w_post_l1 = Worker(
        post_l1_fn,
        fn_args=[
            act_bn14_out.cons(),
            act_out_post_avgpool_shim.prod(),
            post_l1_pb,
            k_post_l1,
            post_L1_InW,
            post_L1_InH,
            post_L1_InC,
            post_L1_OutC,  # outC = 960 (pre-pad). Lowlevel passes 960 here, iron
            post_L2_InC,  # outC_padd = 1280 (padded to next layer's input)
            post_sf,
        ],
        tile=Tile(6, 4),
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
    fc_out_per_tile = post_L2_OutC // n_fc_tiles  # 1280/4 = 320

    # Output fifo: all 4 FC tiles join their results here via MemTile
    act_out_of = ObjectFifo(
        np.ndarray[(post_L2_OutC,), np.dtype[np.uint16]],
        depth=2,
        name="act_post_out",
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
    fc_full_per_tile = post_L2_InC * fc_out_per_tile  # 409,600 bytes per FC half
    fc_recv_per_tile = (
        fc_full_per_tile // PostOutputSplitL2
    )  # 10,240 bytes on compute tile
    # co: channels per output element (8 channels per objectfifo element, 40 per inference)
    # Matches original ty_post_Layer2_out_split = memref<8 x uint16>
    co = post_L2_OutC // (PostOutputSplitL2 * n_fc_tiles)  # = 8

    # Split the output fifo into 4 segments, one per FC tile.
    # Each element = 8 channels (co), depth=1 — matches original design.
    # unrollForLoops sees the objectfifo.acquire inside WeightIndex loop → no unrolling.
    act_post_l2_tiles = act_out_of.prod().join(
        offsets=[i * fc_out_per_tile for i in range(n_fc_tiles)],
        depths=[2] * n_fc_tiles,
        obj_types=[np.ndarray[(co,), np.dtype[np.uint16]]] * n_fc_tiles,
        names=[f"act_post_l2_tile{i}" for i in range(n_fc_tiles)],
        tile=Tile(6, 1),  # mem_tile_6_1 in placed (matches @act_out)
    )

    fc1_memtiles = [Tile(0, 1), Tile(2, 1), Tile(4, 1), Tile(6, 1)]  # FC1 MemTiles
    fc2_memtiles = [
        Tile(1, 1),
        Tile(3, 1),
        Tile(5, 1),
        Tile(7, 1),
    ]  # FC2 MemTiles (serve DMA)
    fc_comptiles = [
        Tile(6, 3),
        Tile(7, 4),
        Tile(7, 3),
        Tile(7, 2),
    ]  # PostL2 compute tiles

    def _u16(shape):
        return np.ndarray[shape, np.dtype[np.uint16]]

    # Post-L2 FC: int8 input → uint16 output (matches original ty_post_Layer2_out_split)
    # Output element = co=8 channels (not fc_out_per_tile=320): each element is one
    # WeightIndex-loop iteration's output slice.
    k_post_l2 = Kernel(
        "post_L2_conv2dk1_relu_i16_ui16_pad",
        "post_L2_conv2dk1_relu_ui16_ui16_pad.o",
        [
            np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
            _i8((fc_recv_per_tile,)),
            _u16((co,)),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
            _i32(),
        ],
    )

    post_l2_workers = []
    for i, (fc1_f, fc2_f) in enumerate(fc_wts_filenames):
        fc1_data = np.zeros(fc_full_per_tile, dtype=np.int8)
        fc2_data = np.zeros(fc_full_per_tile, dtype=np.int8)
        if os.path.exists(data_dir + fc1_f):
            fc1_data = np.fromfile(data_dir + fc1_f, sep=",", dtype=np.int8)
        if os.path.exists(data_dir + fc2_f):
            fc2_data = np.fromfile(data_dir + fc2_f, sep=",", dtype=np.int8)

        # PersistentBuffer ping-pong, FC1-first to match lowlevel chronologically.
        # Primary memtile = fc2_memtiles[i] (the *odd* memtile column in
        # lowlevel — that's where the MM2S DMA flow lives) so we don't collide
        # with post_L1's MM2S 0 on MemTile(4,1). The data layout differs from
        # lowlevel (lowlevel puts FC1_x on the even tile, FC2_x on the odd
        # tile) but the on-wire sequence is what matters: BD1 sends the primary
        # buffer first, so we put FC1 data in the primary buffer regardless of
        # which tile holds it.
        fc_pb = StaticWeightStream(
            obj_type=_i8((fc_full_per_tile,)),
            initial_value=fc1_data,
            name=f"post_l2_fc1_wts_{i}",
            recv_type=_i8((fc_recv_per_tile,)),
            repeat_count=PostOutputSplitL2,
            memtile_placement=fc2_memtiles[i],
            compute_placement=fc_comptiles[i],
            mm2s_channel=0,
            s2mm_channel=1,
            ping_pong_buf=(_i8((fc_full_per_tile,)), fc2_data, f"post_l2_fc2_wts_{i}"),
            ping_pong_memtile=fc1_memtiles[i],
            mem_lock_id=0,
            comp_lock_id=2,
            pp_lock_id=0,
        )

        def post_l2_fn(
            act_in,
            act_out,
            wts_h,
            k,
            inC_fc1,
            inC_fc2,
            outC,
            n_co,
            sf1,
            sf2,
            n_splits=PostOutputSplitL2,
        ):
            # Match lowlevel postBlockL2 exactly: FC1 first, then FC2, with
            # one acquire/release of act_in per pass. Runtime pings FC input
            # through L3 (avgpool -> FC1 in, FC1 out -> FC2 in).
            elem_in = act_in.acquire(1)
            for _ in range_(n_splits):
                elem_out = act_out.acquire(1)
                wts = wts_h.acquire(1)
                k(elem_in, wts, elem_out, 1, inC_fc1, outC, n_co, sf1)
                wts_h.release(1)
                act_out.release(1)
            act_in.release(1)
            elem_in = act_in.acquire(1)
            for _ in range_(n_splits):
                elem_out = act_out.acquire(1)
                wts = wts_h.acquire(1)
                k(elem_in, wts, elem_out, 1, inC_fc2, outC, n_co, sf2)
                wts_h.release(1)
                act_out.release(1)
            act_in.release(1)

        w = Worker(
            post_l2_fn,
            fn_args=[
                act_out_post_shim_FC.cons(),
                act_post_l2_tiles[i].prod(),
                fc_pb,
                k_post_l2,
                post_L1_OutC,  # inC for FC1 = 960 (pre-pad)
                post_L2_InC,  # inC for FC2 = 1280 (padded)
                post_L2_OutC,  # outC = 1280 (full output channels)
                co,  # n_co = 8 (chunk size per kernel call)
                post_fc1_sf,
                post_fc2_sf,
            ],
            tile=fc_comptiles[i],
        )
        post_l2_workers.append(w)

    # ------------------------------------------------------------------
    # Collect all workers
    # ------------------------------------------------------------------
    all_workers = (
        [w_init] + a_workers + b_workers + c_workers + [w_post_l1] + post_l2_workers
    )

    # Combined cascade weight tensor — matches test_mobilenet.py API (3 buffers).
    # Layout: bn13_L1(76800) | bn13_L3_put(38400) | bn13_L3_get(38400) |
    #         bn14_L1(76800) | bn14_L3_put(38400) | bn14_L3_get(38400) = 307200
    from aie.helpers.taplib import TensorAccessPattern

    _L1 = 80 * 960  # 76800
    _L3h = 480 * 80  # 38400 (half)
    _L3f = _L3h * 2  # 76800 (full = put+get combined)
    _cascade_wts_sz = (_L1 + _L3h * 2) * 2  # 307200
    cascade_wts_ty = np.ndarray[(_cascade_wts_sz,), np.dtype[np.int8]]

    # TAP helpers: slice a sub-tensor from the combined buffer (offset in int8 elements)
    def _wts_tap(offset, size):
        return TensorAccessPattern(
            (_cascade_wts_sz,),
            offset=offset,
            sizes=[1, 1, 1, size],
            strides=[0, 0, 0, 1],
        )

    # strict_task_groups=False so we can mix the default group (fills/final
    # drain) with an explicit group used only to force a wait between the
    # avgpool drain and its re-fill in the post-block round-trip.
    rt = Runtime(strict_task_groups=False)
    with rt.sequence(in_ty, cascade_wts_ty, out_ty) as (inp, cascade_wts, out):
        rt.start(*all_workers)
        rt.fill(act_in.prod(depth=1), inp, tile=Tile(0, 0))
        # Slice each sub-weight from the combined buffer (offsets in int8 elements):
        #   0..76799:   bn13 L1 (76800)
        #   76800..153599: bn13 L3 (38400 put + 38400 get = 76800 combined)
        #   153600..230399: bn14 L1 (76800)
        #   230400..307199: bn14 L3 (38400 put + 38400 get = 76800 combined)
        rt.fill(wts_fifos[0].prod(), cascade_wts, _wts_tap(0, _L1), tile=Tile(4, 0))
        rt.fill(wts_fifos[1].prod(), cascade_wts, _wts_tap(_L1, _L3f), tile=Tile(5, 0))
        rt.fill(
            wts_fifos[2].prod(),
            cascade_wts,
            _wts_tap(_L1 + _L3f, _L1),
            tile=Tile(6, 0),
        )
        rt.fill(
            wts_fifos[3].prod(),
            cascade_wts,
            _wts_tap(_L1 * 2 + _L3f, _L3f),
            tile=Tile(7, 0),
        )
        # Round-trip the avgpool output through L3 (DDR) — matches original
        # design's shim 30/40 hop. Reuses inp as scratch (input has been fully
        # consumed by the time PostL1 produces output). drain waits so that the
        # subsequent fill sees valid data.
        # 1280 ui16 = 2560 bytes; tap is in i8 (inp dtype) units.
        _post_l1_out_sz = post_L1_OutW * post_L1_OutH * post_L2_InC * 2
        _post_l1_scratch_tap = TensorAccessPattern(
            (tensorInW * tensorInH * tensorInC,),
            offset=0,
            sizes=[1, 1, 1, _post_l1_out_sz],
            strides=[0, 0, 0, 1],
        )
        # Drain into its own task_group + finish so the await is emitted BEFORE
        # the fill (matches original's `dma_wait("act_out_post_avgpool_shim")`).
        # iron's default task group defers all awaits to the end of the sequence,
        # which would let the fill's BD start before the DDR scratch is valid.
        _tg_drain = rt.task_group()
        rt.drain(
            act_out_post_avgpool_shim.cons(),
            inp,
            tap=_post_l1_scratch_tap,
            wait=True,
            task_group=_tg_drain,
            tile=Tile(3, 0),
        )
        rt.finish_task_group(_tg_drain)
        # First FC pass: avgpool output (in inp scratch at offset 0) -> FC1 input.
        # post_L2 worker now computes FC1 first (matches lowlevel chronologically).
        # Round-trip the FC1 output through L3 (offset = _post_l1_out_sz) and
        # re-fill as FC2 input.
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_l1_scratch_tap,
            tile=Tile(4, 0),
        )
        _post_fc_out_tap = TensorAccessPattern(
            (tensorInW * tensorInH * tensorInC,),
            offset=_post_l1_out_sz,
            sizes=[1, 1, 1, _post_l1_out_sz],
            strides=[0, 0, 0, 1],
        )
        # Second FC pass: drain first FC output to L3 then re-feed as input.
        # Wait so the subsequent fill sees the L3 buffer fully written.
        _tg_fc1 = rt.task_group()
        rt.drain(
            act_out_of.cons(),
            inp,
            tap=_post_fc_out_tap,
            wait=True,
            task_group=_tg_fc1,
            tile=Tile(7, 0),
        )
        rt.finish_task_group(_tg_fc1)
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_fc_out_tap,
            tile=Tile(4, 0),
        )
        # Final drain: second FC output -> host buffer.
        rt.drain(act_out_of.cons(), out, wait=True, tile=Tile(7, 0))

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    module = mobilenet_iron()
    print(module)
