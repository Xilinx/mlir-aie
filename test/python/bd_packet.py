# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test that Bd.packet emits a distinct aie.dma_bd_packet op BEFORE the
aie.dma_bd, rather than a `packet` attribute on the dma_bd. The CDO/xclbin
backends (AIERT, AIETargetXAIEV2) read the packet header only from the
dma_bd_packet op, so the header must be emitted as that op to reach hardware."""

import numpy as np

from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit_packet_bd():
    n = 256
    vector_ty = np.ndarray[(n,), np.dtype[np.int32]]

    compute_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=compute_tile, type=vector_ty, name="pkt_buf")

    tile_dma = TileDma(
        tile=compute_tile,
        channels=[
            DmaChannel(
                direction=DMAChannelDir.MM2S,
                channel=0,
                bds=[
                    Bd(
                        buffer=buf,
                        offset=0,
                        length=n,
                        packet=(0, 5),
                        next="self",
                    ),
                ],
            ),
        ],
    )

    rt = Runtime()
    rt.add_tile_dma(tile_dma)
    with rt.sequence(vector_ty) as _:
        pass

    return Program(NPU2Col1(), rt).resolve_program()


# The packet header is a distinct op emitted immediately before the dma_bd,
# and the dma_bd itself carries no packet attribute.
# CHECK: aie.dma_bd_packet(0, 5)
# CHECK-NEXT: aie.dma_bd({{.*}} : memref<256xi32>, 0, 256)
# CHECK-NOT: aie.dma_bd({{.*}}packet
print(emit_packet_bd())
