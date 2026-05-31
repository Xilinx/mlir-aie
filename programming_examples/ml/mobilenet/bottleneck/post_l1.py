#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Post-processing L1: avg pool + expand 1x1 conv.

Mirror of the sibling bottleneck builders: takes the upstream activation
fifo, owns its weight stream / kernel / worker, returns the workers list
plus the output shim fifo. The output fifo drains to host scratch (round-
trip through L3) before re-broadcasting to PostL2 — see the comment near
`act_out_post_avgpool_shim` for why.

Input:  (7,1,80) int8   Output: (1,1,1280) uint16 (post_L2_InC wide)
"""

import numpy as np

from aie.iron import Kernel, ObjectFifo, Worker, kernels
from aie.iron.controlflow import range_

from bottleneck._common import i8, load_wts
from lowlevel_dma import StaticWeightStream
from network_spec import block as nsblock


def post_l1(act_in, sf, *, placement, data_dir):
    """Build the post-L1 (avg-pool + 1x1 expand) block.

    Args:
        act_in: ObjectFifo  — handoff from the cascade bottleneck (bn14 out).
        sf: dict            — full scale-factor mapping (uses sf["POST"]["conv1x1_1"]).
        placement: dict     — PLACEMENT["post_l1"] with keys "compute", "memtile".
        data_dir: str       — directory holding `post_conv_chain.txt`.

    Returns:
        workers: list[Worker]
        act_out_post_avgpool_shim: ObjectFifo  — drained to L3 (uint16, post_L2_InC wide).
    """
    blk = nsblock("post_l1")
    post_L1_InW, post_L1_InH, post_L1_InC = blk.layers[0].in_shape
    post_L1_OutC = 960  # expand-1x1 width before padding to L2_InC (kernel-internal)
    post_L2_InC = nsblock("post_l2").layers[0].in_shape[2]
    post_sf = sf["POST"]["conv1x1_1"]

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

    post_l1_wts_data = load_wts(data_dir, "post_conv_chain.txt", post_l1_wts_full_sz)

    post_l1_pb = StaticWeightStream(
        obj_type=i8((post_l1_wts_full_sz,)),
        initial_value=post_l1_wts_data,
        name="post_L1_wts",
        recv_type=i8((post_l1_wts_chunk,)),
        repeat_count=PostRepeatChannels,
        memtile_placement=placement["memtile"],
        compute_placement=placement["compute"],
        mem_lock_id=2,
        comp_lock_id=0,
    )

    # Round-trip avgpool output through L3 (DDR) so it can be re-broadcast to
    # all 4 PostL2 FC tiles — a direct compute→4-compute fan-out exceeds
    # stream-switch routing capacity from tile(6,4). Element type is uint16:
    # the kernel writes 2 bytes per output channel; declaring i8 here would
    # halve the DMA transfer size and deadlock the consumer.
    act_out_post_avgpool_shim = ObjectFifo(
        np.ndarray[(post_L2_InC,), np.dtype[np.uint16]],
        depth=2,
    )

    k_post_l1 = kernels.bn_conv2dk1_relu_xy_pool_padded(
        input_width=post_L1_InW,
        input_channels=post_L1_InC,
        output_channels=post_L2_InC,
        weight_chunk_count=post_l1_wts_chunk,
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
            act_in.cons(),
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
        tile=placement["compute"],
        # dynamic_objfifo_lowering keeps the inner loop intact instead of
        # unrolling for ping-pong; kernel uses runtime modulo indexing.
        # Without this attribute, the static objfifo lowering UNROLLS the
        # inner loops to handle ping-pong buffer alternation explicitly,
        # producing 15 func.call ops in 9 basic blocks (lowlevel keeps the
        # loop intact with 1 call site). The dynamic lowering uses runtime
        # modulo indexing, preserving the loop structure.
        dynamic_objfifo_lowering=True,
    )

    return [w_post_l1], act_out_post_avgpool_shim
