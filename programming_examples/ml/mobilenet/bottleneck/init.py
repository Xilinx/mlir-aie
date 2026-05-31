#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
"""Init conv block (3x3 stride-2) for MobileNet V3 IRON API.

Single builder `init_conv` mirrors the sibling-bottleneck signature shape:
takes the block's scale factors and placement, owns its own activations
fifos / weights buffer / kernel / worker, returns `(workers, act_in,
act_init_out)`. `act_in` is the shim-DMA destination for the design's
input activations and is exposed so the runtime sequence's `rt.fill`
can address it.
"""

import numpy as np

from aie.iron import ObjectFifo, Worker, kernels
from aie.iron.controlflow import range_

from ._common import wts_buffer
from ..network_spec import block as nsblock


def init_conv(sf, *, tile, data_dir):
    """Build the init 3x3 stride-2 conv block.

    Returns:
        workers: list[Worker]  — single-element [w_init] for collection.
        act_in: ObjectFifo     — shim input fifo (i8 activations).
        act_init_out: ObjectFifo — handoff to the first bottleneck (u8).
    """
    blk = nsblock("init")
    tensorInW, tensorInH, tensorInC = blk.layers[0].in_shape
    init_OutW, init_OutH, init_OutC = blk.layers[0].out_shape
    init_scaleFactor = sf["INIT"]["conv3x3"]

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
    init_wts = wts_buffer(data_dir, "init_chain.txt", init_wts_sz)

    # Init conv kernel: 3x3 stride-2, int8 in, uint8 out.
    # C++ signature (aie_kernels/aie2/bottleneck/bn_conv2dk3.cc):
    #   (in0, in0, in1, wts, out, W, InC, OutC, kW, kH, check, scale, channel_offset)
    k_init = kernels.bn_conv2dk3(
        input_width=tensorInW,
        input_channels=tensorInC,
        output_channels=init_OutC,
    )

    def init_fn(act_in, act_out, wts, k, inW, inH, inC, outW, outH, outC, sf):
        # 2-phase: preamble (border=0) + (outH-1) middle iters; trailing
        # release(1) drains the last input row. Preamble releases output BEFORE
        # input; middle iter does the opposite (in then out).
        rows = act_in.acquire(2)
        row_out = act_out.acquire(1)
        k(rows[0], rows[0], rows[1], wts, row_out, inW, inC, outC, 3, 3, 0, sf, 0)
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
        tile=tile,
    )

    return [w_init], act_in, act_init_out
