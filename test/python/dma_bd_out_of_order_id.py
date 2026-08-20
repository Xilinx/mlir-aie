# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test the IRON sender-side out-of-order id: Bd(out_of_order_id=slot) forwards
the id through the real producer (aie.iron.Bd -> aie.dma_bd's out_of_order_id
attribute), so a sending BD can be expressed entirely in dataflow instead of
dropping to the runtime aiex.npu.writebd descriptor. Each BD is packet-enabled
(the id rides the packet header), emitted as a sibling aie.dma_bd_packet op."""

import numpy as np

from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir

tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
buf = Buffer(tile=tile, type=np.ndarray[(16,), np.dtype[np.int32]], name="buf")

# Two sending BDs, each stamping a distinct out-of-order id (its target slot).
bds = [
    Bd(buffer=buf, offset=4 * i, length=4, packet=(0, 0), out_of_order_id=i)
    for i in range(2)
]
tile_dma = TileDma(
    tile=tile,
    channels=[DmaChannel(direction=DMAChannelDir.MM2S, channel=0, bds=bds)],
)


def sequence(_):
    pass


rt = Runtime(sequence, [np.ndarray[(16,), np.dtype[np.int32]]])
rt.add_tile_dma(tile_dma)

# Each sending BD carries a packet header (dma_bd_packet) and the matching
# out_of_order_id attribute on the dma_bd.
# CHECK: aie.dma_bd_packet(0, 0)
# CHECK: aie.dma_bd({{.*}}out_of_order_id = 0 : i32
# CHECK: aie.dma_bd_packet(0, 0)
# CHECK: aie.dma_bd({{.*}}out_of_order_id = 1 : i32
print(Program(NPU2Col1(), rt).resolve_program())
