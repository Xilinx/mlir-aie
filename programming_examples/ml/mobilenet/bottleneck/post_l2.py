#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Post-processing L2: 4-tile FC1 + FC2 (split output channels).

Mirror of the sibling bottleneck builders. The 4 FC tiles each consume
the broadcast uint16 avgpool output (`act_in`, fed from host scratch by
the runtime sequence) and join their per-tile output slices into a
single uint16 output fifo via MemTile.

Input:  (1,1,1280) uint16  Output: (1,1,1280) uint16 (4 tiles, joined)
"""

import numpy as np

from aie.iron import Kernel, ObjectFifo, Worker
from aie.iron.controlflow import range_

from bottleneck._common import i8, load_wts
from lowlevel_dma import StaticWeightStream
from network_spec import block as nsblock


def post_l2(act_in, sf, *, placement, data_dir):
    """Build the post-L2 (4-tile FC1+FC2) block.

    Args:
        act_in: ObjectFifo  — host-scratch fill of the avgpool output (uint16).
        sf: dict            — full scale-factor mapping; uses sf["POST"]["FC1"], ["FC2"].
        placement: dict     — PLACEMENT["post_l2"] with keys "fc1_memtiles",
                              "fc2_memtiles", "compute", "join_memtile".
        data_dir: str       — directory holding FC{1,2}_{0..3}_chain.txt.

    Returns:
        workers: list[Worker]  — the 4 per-tile FC workers.
        act_out_of: ObjectFifo — joined output (uint16, post_L2_OutC wide).
    """
    post_L2_InC = nsblock("post_l2").layers[0].in_shape[2]
    post_L2_OutC = nsblock("post_l2").layers[-1].out_shape[2]
    # post_L1_OutC = FC1 input channel count = post_L1 expand-1x1 output width
    # (pre-pad). See bottleneck/post_l1.py for the matching constant.
    post_L1_OutC = 960
    post_fc1_sf = sf["POST"]["FC1"]
    post_fc2_sf = sf["POST"]["FC2"]

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
        tile=placement["join_memtile"],
    )

    fc1_memtiles = placement["fc1_memtiles"]
    fc2_memtiles = placement["fc2_memtiles"]
    fc_comptiles = placement["compute"]

    def _u16(shape):
        return np.ndarray[shape, np.dtype[np.uint16]]

    # Post-L2 FC: uint16 input (avgpool output) → uint16 output, in `co`-element slices.
    k_post_l2 = Kernel(
        "post_L2_conv2dk1_relu_i16_ui16_pad",
        "post_L2_conv2dk1_relu_ui16_ui16_pad.o",
        [
            np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
            i8((fc_recv_per_tile,)),
            _u16((co,)),
            np.int32,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    post_l2_workers = []
    for i, (fc1_f, fc2_f) in enumerate(fc_wts_filenames):
        fc1_data = load_wts(data_dir, fc1_f, fc_full_per_tile)
        fc2_data = load_wts(data_dir, fc2_f, fc_full_per_tile)

        # FC weights: ping-pong across two adjacent memtiles. Primary = FC1
        # (consumed first), ping-pong = FC2. DMA + primary share fc2_memtiles[i]
        # (odd column) to avoid colliding with post_L1's MM2S on (4,1).
        fc_pb = StaticWeightStream(
            obj_type=i8((fc_full_per_tile,)),
            initial_value=fc1_data,
            name=f"fc1_wts_{i}",
            recv_type=i8((fc_recv_per_tile,)),
            repeat_count=PostOutputSplitL2,
            memtile_placement=fc2_memtiles[i],
            compute_placement=fc_comptiles[i],
            s2mm_channel=1,
            ping_pong_buf=(i8((fc_full_per_tile,)), fc2_data, f"fc2_wts_{i}"),
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
                act_in.cons(),
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

    return post_l2_workers, act_out_of
