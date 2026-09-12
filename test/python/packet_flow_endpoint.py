# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""A PacketFlow names its ends as a Flow does: a DMA program, static or a
runtime task, runs on PacketFlow.endpoint, whose channel is the one the
PacketFlow gives."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Bd, Buffer, DmaChannel, PacketFlow, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile

N = 256
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def build():
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    into = PacketFlow(3, shim, mem, dst_channel=1, name="into")
    onward = PacketFlow(4, mem, core, src_channel=2, dst_channel=1, name="onward")
    staged = Buffer(tile=mem, type=vec_ty, name="staged")
    landed = Buffer(tile=core, type=vec_ty, name="landed")

    mem_dma = TileDma(
        mem, [DmaChannel(DMAChannelDir.S2MM, into.endpoint(mem), [Bd(staged)])]
    )
    core_dma = TileDma(
        core, [DmaChannel(DMAChannelDir.S2MM, onward.endpoint(core), [Bd(landed)])]
    )

    def sequence(a):
        into.fill(a)
        onward.endpoint(mem).task(Bd(staged, packet=(0, 4))).start().free()

    rt = Runtime(sequence, [vec_ty])
    for fl in (into, onward):
        rt.add_flow(fl)
    for td in (mem_dma, core_dma):
        rt.add_tile_dma(td)
    return Program(NPU2Col1(), rt).resolve_program()


# CHECK: aie.memtile_dma
# CHECK: aie.dma_start(S2MM, 1,
# CHECK: aie.mem(
# CHECK: aie.dma_start(S2MM, 1,
# CHECK-LABEL: aie.runtime_sequence
# CHECK: aiex.dma_configure_task(%{{.*}}, MM2S, 2) {
# CHECK: aie.dma_bd(%staged : memref<256xi32>) {packet = #aie.packet_info<pkt_id = 4>}
# CHECK: aie.packet_flow(3)
# CHECK: aie.packet_flow(4)
print(build())


def not_an_end():
    shim = Tile(tile_type=AIETileType.ShimNOCTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    PacketFlow(0, shim, mem).endpoint(Tile(tile_type=AIETileType.CoreTile))


try:
    not_an_end()
except ValueError as e:
    print(f"// not_an_end: {e}")
# CHECK: // not_an_end: Tile{{.*}} is not an end of this PacketFlow.
