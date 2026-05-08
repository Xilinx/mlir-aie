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

# Allow importing the local bottleneck/ package when running this file directly.
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from bottleneck.regular import regular_bottlenecks
from bottleneck.pipeline import pipeline_bottlenecks
from bottleneck.cascade import cascade_bottlenecks
from network_spec import block as nsblock

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
post_sf = sf["POST"]["conv1x1_1"]
post_fc1_sf = sf["POST"]["FC1"]
post_fc2_sf = sf["POST"]["FC2"]

# ---------------------------------------------------------------------------
# Network dimensions — derived from network_spec.NETWORK (single source of truth)
# ---------------------------------------------------------------------------
tensorInW, tensorInH, tensorInC = nsblock("init").layers[0].in_shape
init_OutW, init_OutH, init_OutC = nsblock("init").layers[0].out_shape

# Post-processing dimensions
post_L1_InW, post_L1_InH, post_L1_InC = nsblock("post_l1").layers[0].in_shape
post_L1_OutW, post_L1_OutH, _ = nsblock("post_l1").layers[0].out_shape
post_L1_OutC = 960  # expand-1x1 width before padding to L2_InC (kernel-internal)

post_L2_InC = nsblock("post_l2").layers[0].in_shape[2]
post_L2_OutC = nsblock("post_l2").layers[-1].out_shape[2]
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


def _require_wts(path, expected_size):
    """Load int8 weights from `path`; raise if missing or wrong size.

    Silent zero-fill on missing files used to compile a numerically-broken
    design that ran fine but produced wrong output — fail loud instead.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"weight file not found: {path}")
    arr = np.fromfile(path, sep=",", dtype=np.int8)
    if arr.size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} int8 elements, got {arr.size}"
        )
    return arr


# ---------------------------------------------------------------------------
# Tile placement — single source of truth.
#
# Compute tiles use rows 2-5 of columns 0-7 (Strix layout). MemTiles live on
# row 1; shim DMA endpoints on row 0. Adjacent rows in the same column share
# memory, which we exploit for fused-pair self-loop fifos and L2→L3 handoffs.
#
# Re-targeting the design = editing this dict (and only this dict).
# ---------------------------------------------------------------------------
PLACEMENT = {
    "init": Tile(0, 2),
    "post_l1": {"compute": Tile(6, 4), "memtile": Tile(4, 1)},
    "post_l2": {
        "fc1_memtiles": [Tile(0, 1), Tile(2, 1), Tile(4, 1), Tile(6, 1)],
        "fc2_memtiles": [Tile(1, 1), Tile(3, 1), Tile(5, 1), Tile(7, 1)],
        "compute": [Tile(6, 3), Tile(7, 4), Tile(7, 3), Tile(7, 2)],
        "join_memtile": Tile(6, 1),
    },
    "shim": {
        "input": Tile(0, 0),
        "wts": [Tile(c, 0) for c in (4, 5, 6, 7)],
        "scratch_drain": Tile(3, 0),
        "fc_fill": Tile(4, 0),
        "fc_drain": Tile(7, 0),
    },
    "regular": {
        "bn0": Tile(0, 3),
        "bn1": Tile(0, 4),
        "bn2": Tile(0, 5),
        "bn3": Tile(1, 3),
        "bn4_5": {"compute": Tile(1, 2), "alloc": Tile(0, 2)},  # alloc on init tile
        "bn6": Tile(1, 4),
        "bn7": Tile(2, 3),
        "bn8_9": {"compute": Tile(3, 3), "alloc": Tile(3, 4)},  # alloc on bn11 L1 tile
    },
    "pipeline": {
        "bn10": {"l1": Tile(1, 5), "l2": Tile(2, 4), "l3": Tile(2, 5)},
        "bn11": {
            "l1": Tile(3, 2),
            "l2": Tile(3, 4),
            "l3": Tile(2, 2),
            "mem_skip": Tile(2, 1),
        },
        "bn12": {"l1": Tile(3, 5), "l23": Tile(4, 4)},
    },
    "cascade": {
        "bn13": {
            "l1_put": Tile(4, 5),
            "l1_get": Tile(5, 5),
            "l2": Tile(5, 4),
            "l3_put": Tile(4, 3),
            "l3_get": Tile(5, 3),
            "mem_l1": Tile(0, 1),
            "mem_l3": Tile(1, 1),
            "mem_skip": Tile(5, 1),
        },
        "bn14": {
            "l1_put": Tile(6, 5),
            "l1_get": Tile(7, 5),
            "l2": Tile(6, 2),
            "l3_put": Tile(4, 2),
            "l3_get": Tile(5, 2),
            "mem_l1": Tile(2, 1),
            "mem_l3": Tile(3, 1),
            "mem_skip": Tile(7, 1),
        },
    },
}


# ---------------------------------------------------------------------------
# Design top-level function
# ---------------------------------------------------------------------------
def mobilenet_iron(collect_only: bool = False):
    """Build the full mobilenet IRON design.

    By default returns the resolved Program (MLIR module). If `collect_only`
    is True, instead returns the list of all Workers BEFORE Program resolution
    — used by dataflow_dot.py to walk the design without lowering it.
    """
    # Runtime arg types: i32 element view over the underlying byte buffers.
    #   arg0 (act_in / scratch):  100352 i32 = 401408 bytes
    #   arg2 (final FC2 output):    640 i32 =   2560 bytes
    in_ty = np.ndarray[(tensorInW * tensorInH * tensorInC // 4,), np.dtype[np.int32]]
    out_ty = np.ndarray[
        (post_L1_OutW * post_L1_OutH * post_L2_OutC * 2 // 4,),
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
    init_wts_data = _require_wts(data_dir + "init_chain.txt", init_wts_sz)

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
        # 2-phase: preamble (border=0) + (outH-1) middle iters; trailing
        # release(1) drains the last input row. Preamble releases output BEFORE
        # input; middle iter does the opposite (in then out).
        rows = act_in.acquire(2)
        row_out = act_out.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out, inW, inC, outC, 3, 3, 0, sf, 0, 0)
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
        tile=PLACEMENT["init"],
    )

    # ------------------------------------------------------------------
    # Bottleneck blocks
    # ------------------------------------------------------------------
    a_workers, act_bn9_out = regular_bottlenecks(
        act_init_out, sf, placement=PLACEMENT["regular"], data_dir=data_dir
    )
    b_workers, act_bn12_out = pipeline_bottlenecks(
        act_bn9_out, sf, placement=PLACEMENT["pipeline"], data_dir=data_dir
    )
    c_workers, act_bn14_out, wts_fifos = cascade_bottlenecks(
        act_bn12_out, sf, placement=PLACEMENT["cascade"], data_dir=data_dir
    )

    # ------------------------------------------------------------------
    # Post-processing L1: avg pool + expand 1x1 conv
    # Input:  (7,1,80) int8   Output: (1,1,960) int8
    # ------------------------------------------------------------------
    # 76800 B of L1 weights are too large for the compute tile. Stage them on
    # MemTile(4,1) and stream 9600 B chunks via StaticWeightStream.
    PostOutputSplit = 8
    PostRepeatChannels = post_L1_InH  # = 7
    post_l1_wts_full_sz = post_L1_OutC * post_L1_InC  # 76800 B on MemTile
    post_l1_wts_chunk = post_l1_wts_full_sz // PostOutputSplit  # 9600 B per chunk

    post_l1_wts_data = _require_wts(
        data_dir + "post_conv_chain.txt", post_l1_wts_full_sz
    )

    post_l1_pb = StaticWeightStream(
        obj_type=_i8((post_l1_wts_full_sz,)),
        initial_value=post_l1_wts_data,
        name="post_L1_wts",
        recv_type=_i8((post_l1_wts_chunk,)),
        repeat_count=PostRepeatChannels,
        memtile_placement=PLACEMENT["post_l1"]["memtile"],
        compute_placement=PLACEMENT["post_l1"]["compute"],
        mem_lock_id=2,
        comp_lock_id=0,
    )

    # Round-trip avgpool output through L3 (DDR) so it can be re-broadcast to
    # all 4 PostL2 FC tiles — a direct compute→4-compute fan-out exceeds
    # stream-switch routing capacity from tile(6,4). Element type is uint16:
    # the kernel writes 2 bytes per output channel; declaring i8 here would
    # halve the DMA transfer size and deadlock the consumer.
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
        # One full output frame: acquire output, loop over rows then weight splits.
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
                    yi,
                    n_splits,
                    wi,
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
            post_L1_OutC,  # outC=960 (pre-pad)
            post_L2_InC,  # outC_padd=1280 (next layer's input width)
            post_sf,
        ],
        tile=PLACEMENT["post_l1"]["compute"],
        # dynamic_objfifo_lowering keeps the inner loop intact instead of
        # unrolling for ping-pong; kernel uses runtime modulo indexing.
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

    # FC weights: ping-pong across two adjacent MemTiles per compute tile (FC1
    # on the even-column MemTile, FC2 on the odd-column MemTile that hosts the
    # MM2S DMA — keeps DMA off (4,1) which post_L1 owns).
    PostOutputSplitL2 = 40
    fc_full_per_tile = post_L2_InC * fc_out_per_tile  # 409600 B per FC half
    fc_recv_per_tile = fc_full_per_tile // PostOutputSplitL2  # 10240 B on compute
    # `co` = channels per ObjectFifo element (one WeightIndex iteration's output slice).
    co = post_L2_OutC // (PostOutputSplitL2 * n_fc_tiles)  # = 8

    # Split the output fifo into 4 channel-segments, one per FC tile.
    act_post_l2_tiles = act_out_of.prod().join(
        offsets=[i * fc_out_per_tile for i in range(n_fc_tiles)],
        depths=[2] * n_fc_tiles,
        obj_types=[np.ndarray[(co,), np.dtype[np.uint16]]] * n_fc_tiles,
        tile=PLACEMENT["post_l2"]["join_memtile"],
    )

    fc1_memtiles = PLACEMENT["post_l2"]["fc1_memtiles"]
    fc2_memtiles = PLACEMENT["post_l2"]["fc2_memtiles"]
    fc_comptiles = PLACEMENT["post_l2"]["compute"]

    def _u16(shape):
        return np.ndarray[shape, np.dtype[np.uint16]]

    # Post-L2 FC: uint16 input (avgpool output) → uint16 output, in `co`-element slices.
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
        fc1_data = _require_wts(data_dir + fc1_f, fc_full_per_tile)
        fc2_data = _require_wts(data_dir + fc2_f, fc_full_per_tile)

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
            # Two FC passes (FC1 then FC2) sharing the same inner ping-pong loop.
            # The outer Python for-loop is inlined at codegen time → emits two
            # copies of the inner range_ loop, same MLIR as a manual unroll.
            for inC, sf in ((inC_fc1, sf1), (inC_fc2, sf2)):
                elem_in = act_in.acquire(1)
                for _ in range_(n_splits):
                    elem_out = act_out.acquire(1)
                    wts = wts_h.acquire(1)
                    k(elem_in, wts, elem_out, 1, inC, outC, n_co, sf)
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

    _BN_L1_SZ = 80 * 960  # 76800 bytes per L1 weight chunk
    _BN_L3_SZ = 480 * 80 * 2  # 76800 bytes per L3 weight chunk (put+get)
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
        rt.fill(
            act_in.prod(depth=1), inp, tile=PLACEMENT["shim"]["input"], task_group=tg1
        )
        # bn13/14 L1+L3 weight chunks from the combined cascade buffer
        for fifo, off, sz, shim in zip(
            wts_fifos, _CASCADE_OFFSETS, _CASCADE_SIZES, PLACEMENT["shim"]["wts"]
        ):
            rt.fill(
                fifo.prod(), cascade_wts, _wts_tap(off, sz), tile=shim, task_group=tg1
            )
        # Round-trip avgpool output through L3 (shim 30/40 hop). Reuse `inp`
        # as scratch — input is fully consumed by the time PostL1 emits output.
        # Offsets/sizes are i32 elements (4 bytes each):
        #   avgpool / FC1-input scratch: i32 offset 640  (byte 2560)
        #   FC1-output / FC2-input scratch: i32 offset 1280 (byte 5120)
        #   transfer length: 640 i32 = 2560 B = 1280 ui16
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
            tile=PLACEMENT["shim"]["scratch_drain"],
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
            tile=PLACEMENT["shim"]["fc_fill"],
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
            tile=PLACEMENT["shim"]["fc_drain"],
        )
        rt.finish_task_group(tg2)

        # ---- Group 3: FC2 fill + FC2 final drain to host ----
        tg3 = rt.task_group()
        rt.fill(
            act_out_post_shim_FC.prod(),
            inp,
            tap=_post_fc_out_tap,
            tile=PLACEMENT["shim"]["fc_fill"],
            task_group=tg3,
        )
        rt.drain(
            act_out_of.cons(),
            out,
            wait=True,
            tile=PLACEMENT["shim"]["fc_drain"],
            task_group=tg3,
        )
        rt.finish_task_group(tg3)

    # ------------------------------------------------------------------
    # Generate MLIR (or return workers for graphviz emission)
    # ------------------------------------------------------------------
    if collect_only:
        return all_workers
    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    if "--dot" in sys.argv:
        from dataflow_dot import emit_dot

        emit_dot(mobilenet_iron(collect_only=True))
        sys.exit(0)
    print(mobilenet_iron())
