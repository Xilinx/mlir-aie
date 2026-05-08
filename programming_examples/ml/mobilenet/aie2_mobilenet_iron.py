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
# Direct dict access (no .get with default) — a missing key fails loud, never silent.
init_scaleFactor = sf["INIT"]["conv3x3"]
post_sf     = sf["POST"]["conv1x1_1"]
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
    # Match lowlevel runtime arg types EXACTLY: lowlevel declares all three
    # runtime args as memref<...xi32> (i32 element view of the underlying byte
    # buffers). Iron previously used i8/ui16 native element types -- same
    # physical bytes but different MLIR element-level interpretation. Switch
    # to i32 to match lowlevel.
    #   arg0 (act_in / scratch): 401408 bytes = 100352 i32
    #   arg2 (final FC2 output): 2560 bytes  = 640   i32
    in_ty = np.ndarray[(tensorInW * tensorInH * tensorInC // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[
        (post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4,),  # 1*1*1280*2/4 = 640 i32
        np.dtype[np.int32],
    ]

    # ------------------------------------------------------------------
    # Init conv boundary fifos (declared here; owned by orchestrator)
    # act_in:      input activations (224,1,8) int8
    # act_init_out: init conv output (112,1,16) uint8
    # ------------------------------------------------------------------
    act_in = ObjectFifo(
        np.ndarray[(tensorInW, 1, tensorInC), np.dtype[np.int8]],
        depth=5,  # Consumer depth=5 to allow 3-row sliding window in init conv
    )
    act_init_out = ObjectFifo(
        np.ndarray[(init_OutW, 1, init_OutC), np.dtype[np.uint8]],
        depth=5,
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
        # Preamble release order: lowlevel releases output BEFORE input
        # (aie2_mobilenet.py:108-109). Middle iter does the opposite (in then out).
        act_out.release(1)
        act_in.release(1)
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
    a_workers, act_bn9_out  = regular_bottlenecks(act_init_out, sf, data_dir=data_dir)
    b_workers, act_bn12_out = pipeline_bottlenecks(act_bn9_out, sf, data_dir=data_dir)
    c_workers, act_bn14_out, wts_fifos = cascade_bottlenecks(
        act_bn12_out, sf, data_dir=data_dir
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

    post_l1_pb = StaticWeightStream(
        obj_type=_i8((post_l1_wts_full_sz,)),
        initial_value=post_l1_wts_data,
        name="post_L1_wts",
        recv_type=_i8((post_l1_wts_chunk,)),
        repeat_count=PostRepeatChannels,
        memtile_placement=Tile(4, 1),
        compute_placement=Tile(6, 4),
        mem_lock_id=2,
        comp_lock_id=0,
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
    )
    act_out_post_shim_FC = ObjectFifo(
        _post_l1_out_ty,
        depth=2,
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
        # Match lowlevel's `@core(PostL1Tile, dynamic_objfifo_lowering=True)`.
        # Without this attribute, the static objfifo lowering UNROLLS the
        # inner loops to handle ping-pong buffer alternation explicitly,
        # producing 15 func.call ops in 9 basic blocks (lowlevel keeps the
        # loop intact with 1 call site). The dynamic lowering uses runtime
        # modulo indexing, preserving the loop structure.
        dynamic_objfifo_lowering=True,
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
        names=[f"post_L2_out_core{i+1}" for i in range(n_fc_tiles)],  # match lowlevel @post_L2_out_core1..4
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

        # FC weights: ping-pong across two adjacent memtiles. Primary = FC1
        # (consumed first), ping-pong = FC2. DMA + primary share fc2_memtiles[i]
        # (odd column) to avoid colliding with post_L1's MM2S on (4,1).
        fc_pb = StaticWeightStream(
            obj_type=_i8((fc_full_per_tile,)),
            initial_value=fc1_data,
            name=f"fc1_wts_{i}",
            recv_type=_i8((fc_recv_per_tile,)),
            repeat_count=PostOutputSplitL2,
            memtile_placement=fc2_memtiles[i],
            compute_placement=fc_comptiles[i],
            s2mm_channel=1,
            ping_pong_buf=(_i8((fc_full_per_tile,)), fc2_data, f"fc2_wts_{i}"),
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
                post_L1_OutC,
                post_L2_InC,
                post_L2_OutC,
                co,
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

    # Combined cascade weight tensor — test_mobilenet.py concatenates 4 chunks
    # into a single buffer in this exact order:
    #   bn13_L1(76800) | bn13_L3(76800) | bn14_L1(76800) | bn14_L3(76800)
    from aie.helpers.taplib import TensorAccessPattern

    _BN_L1_SZ = 80 * 960          # 76800 bytes per L1 weight chunk
    _BN_L3_SZ = 480 * 80 * 2      # 76800 bytes per L3 weight chunk (put+get)
    _CASCADE_OFFSETS = [0, _BN_L1_SZ, 2 * _BN_L1_SZ, 3 * _BN_L1_SZ]
    _CASCADE_SIZES = [_BN_L1_SZ, _BN_L3_SZ, _BN_L1_SZ, _BN_L3_SZ]
    _cascade_wts_sz_i32 = sum(_CASCADE_SIZES) // 4  # 76800 i32 = 307200 bytes
    cascade_wts_ty = np.ndarray[(_cascade_wts_sz_i32,), np.dtype[np.int32]]

    def _wts_tap(byte_offset, byte_size):
        return TensorAccessPattern(
            (_cascade_wts_sz_i32,),
            offset=byte_offset // 4,
            sizes=[1, 1, 1, byte_size // 4],
            strides=[0, 0, 0, 1],
        )

    # Use the gemm-style "one task_group at a time" pattern. Each task_group
    # holds a batch of fills + a wait=True drain, and finish_task_group()
    # awaits the drain AND frees every task in the group atomically. This
    # avoids the previous bug of `dma_free_task` being emitted for tasks
    # that were never awaited (act_in, weights, FC fills) — which deallocated
    # their BD IDs while their DMAs were potentially still in flight.
    rt = Runtime()
    with rt.sequence(in_ty, cascade_wts_ty, out_ty) as (inp, cascade_wts, out):
        rt.start(*all_workers)

        # ---- Group 1: input + weights + avgpool drain ----
        # All upstream fills + the first sync drain in the same group. By the
        # time the avgpool drain completes, init_conv has consumed all of
        # act_in and bn13/14 have consumed their weights, so freeing all of
        # those at finish_task_group() is safe (causal closure).
        tg1 = rt.task_group()
        rt.fill(act_in.prod(depth=1), inp, tile=Tile(0, 0), task_group=tg1)
        # bn13/14 L1+L3 weight chunks from the combined cascade buffer
        for fifo, off, sz, shim in zip(
            wts_fifos, _CASCADE_OFFSETS, _CASCADE_SIZES, [Tile(c, 0) for c in (4, 5, 6, 7)]
        ):
            rt.fill(fifo.prod(), cascade_wts, _wts_tap(off, sz), tile=shim, task_group=tg1)
        # Round-trip the avgpool output through L3 (DDR) — matches original
        # design's shim 30/40 hop. Reuses inp as scratch (input has been fully
        # consumed by the time PostL1 produces output).
        # All offsets/sizes are in i32 elements (4 bytes), matching lowlevel:
        #   avgpool/FC1-input scratch: i32 offset 640 = byte 2560 (1280 ui16)
        #   FC1-output/FC2-input scratch: i32 offset 1280 = byte 5120
        #   transfer length: i32 640 = bytes 2560 = 1280 ui16
        _post_l1_out_sz_i32 = post_L1_OutW * post_L1_OutH * post_L2_InC * 2 // 4  # 640
        _inp_sz_i32 = tensorInW * tensorInH * tensorInC // 4  # 100352
        _post_l1_scratch_tap = TensorAccessPattern(
            (_inp_sz_i32,),
            offset=_post_l1_out_sz_i32,
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        rt.drain(
            act_out_post_avgpool_shim.cons(),
            inp,
            tap=_post_l1_scratch_tap,
            wait=True,
            task_group=tg1,
            tile=Tile(3, 0),
        )
        rt.finish_task_group(tg1)

        # ---- Group 2: FC1 fill + FC1 drain ----
        # FC1 fill reads the avgpool scratch (drained above). FC1 drain
        # waits for FC compute to consume the fill, then drains FC1 output
        # to L3. By finish_task_group, both FC1 fill and FC1 drain have
        # completed.
        tg2 = rt.task_group()
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_l1_scratch_tap,
            tile=Tile(4, 0),
            task_group=tg2,
        )
        _post_fc_out_tap = TensorAccessPattern(
            (_inp_sz_i32,),
            offset=_post_l1_out_sz_i32 * 2,  # i32 offset 1280
            sizes=[1, 1, 1, _post_l1_out_sz_i32],
            strides=[0, 0, 0, 1],
        )
        rt.drain(
            act_out_of.cons(),
            inp,
            tap=_post_fc_out_tap,
            wait=True,
            task_group=tg2,
            tile=Tile(7, 0),
        )
        rt.finish_task_group(tg2)

        # ---- Group 3: FC2 fill + FC2 final drain to host ----
        tg3 = rt.task_group()
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_fc_out_tap,
            tile=Tile(4, 0),
            task_group=tg3,
        )
        rt.drain(
            act_out_of.cons(),
            out,
            wait=True,
            tile=Tile(7, 0),
            task_group=tg3,
        )
        rt.finish_task_group(tg3)

    # ------------------------------------------------------------------
    # Generate MLIR
    # ------------------------------------------------------------------
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    module = mobilenet_iron()
    print(module)
