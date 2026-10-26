#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Post-processing L2: 4-tile FC1 + FC2 (split output channels).

Mirror of the sibling bottleneck builders. The 4 FC tiles each consume
the broadcast uint16 avgpool output (`act_in`, fed from host scratch by
the runtime sequence) and join their per-tile output slices into a
single uint16 output fifo via MemTile.

Input:  (1,1,1280) uint16  Output: (1,1,1280) uint16 (4 tiles, joined)
"""

import numpy as np
from aie.iron import ObjectFifo, Worker, kernels
from aie.iron.controlflow import range_
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint
from aie.iron.device import AnyMemTile

from ..network_spec import block as nsblock
from ._common import i8, load_wts


def post_l2(act_in, sf, *, data_dir):
    """Build the post-L2 (4-tile FC1+FC2) block.

    Args:
        act_in: ObjectFifo  — host-scratch fill of the avgpool output (uint16).
        sf: dict            — full scale-factor mapping; uses sf["POST"]["FC1"], ["FC2"].
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

    # Each FC tile streams 32-output blocks of its weights in chunks of 320
    # inputs, so FC1 (960 real inputs of the 1280 its weights are padded to)
    # streams three chunks a block and FC2 four.
    chunk_ic, co = 320, 32
    chunk_bytes = chunk_ic * co
    n_blocks = fc_out_per_tile // co
    fc1_chunks, fc2_chunks = post_L1_OutC // chunk_ic, post_L2_InC // chunk_ic
    fc_full_per_tile = post_L2_InC * fc_out_per_tile
    fc_stream_per_tile = (fc1_chunks + fc2_chunks) * n_blocks * chunk_bytes

    # Split the output fifo into 4 channel-segments, one per FC tile.
    act_post_l2_tiles = act_out_of.prod().join(
        offsets=[i * fc_out_per_tile for i in range(n_fc_tiles)],
        depths=[2] * n_fc_tiles,
        obj_types=[np.ndarray[(co,), np.dtype[np.uint16]]] * n_fc_tiles,
    )

    k_post_l2 = kernels.bn_fc_relu_ui16_ic_chunk(
        input_channels=post_L2_InC,
        chunk_input_channels=chunk_ic,
        chunk_output_channels=co,
    )

    post_l2_workers = []
    for i, (fc1_f, fc2_f) in enumerate(fc_wts_filenames):
        # [OC/8][1280][8] weights, reordered into the stream's chunk order.
        stream = np.concatenate(
            [
                load_wts(data_dir, f, fc_full_per_tile)
                .reshape(n_blocks, co // 8, post_L2_InC // chunk_ic, -1)[:, :, :n]
                .transpose(0, 2, 1, 3)
                .reshape(-1)
                for f, n in ((fc1_f, fc1_chunks), (fc2_f, fc2_chunks))
            ]
        )

        # The tile's whole weight stream rests on a MemTile, as two halves
        # (each must fit one MemTile); the consumer walks it a chunk at a time.
        fc_wts_of = ObjectFifo(
            i8((fc_stream_per_tile // 2,)),
            depth=2,
            name=f"post_L2_wts_{i + 1}",
            consumer_obj_type=i8((chunk_bytes,)),
            init_values=np.split(stream, 2),
        )
        # Put the producer on a MemTile. Normally a Worker sets its fifo
        # endpoint implicitly, but this fifo has no producing Worker.
        fc_wts_of.prod().endpoint = ObjectFifoEndpoint(AnyMemTile.copy())

        def post_l2_fn(act_in, act_out, wts_h, k, sf1, sf2):
            for n_chunks, sf in ((fc1_chunks, sf1), (fc2_chunks, sf2)):
                elem_in = act_in.acquire(1)
                for _ in range_(n_blocks):
                    elem_out = act_out.acquire(1)
                    for chunk in range(n_chunks):
                        wts = wts_h.acquire(1)
                        k(elem_in, wts, elem_out, chunk, 1, sf)
                        wts_h.release(1)
                    act_out.release(1)
                act_in.release(1)

        w = Worker(
            post_l2_fn,
            fn_args=[
                act_in.cons(),
                act_post_l2_tiles[i].prod(),
                fc_wts_of.cons(),
                k_post_l2,
                post_fc1_sf,
                post_fc2_sf,
            ],
        )
        post_l2_workers.append(w)

    return post_l2_workers, act_out_of
