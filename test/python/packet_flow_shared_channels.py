# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-objectfifo-allocate | FileCheck %s --check-prefix=ALLOC

"""A PacketFlow channel left as None is assigned by the compiler, and an end
given another PacketFlow's end shares its channel: two cores merge into one
MemTile channel, and the MemTile sends both ids back out of one channel."""

import numpy as np

from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.iron import Bd, Buffer, DmaChannel, PacketFlow, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile

N = 64
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def build():
    mem = Tile(0, 1)
    cores = [Tile(0, 2), Tile(0, 3)]

    gather = PacketFlow(
        1, cores[0], mem, src_channel=None, dst_channel=None, keep_pkt_header=True
    )
    gather_b = PacketFlow(
        2,
        cores[1],
        mem,
        src_channel=None,
        dst_channel=gather.endpoint(mem),
        keep_pkt_header=True,
    )
    scatter = PacketFlow(1, mem, cores[0], src_channel=None, dst_channel=None)
    scatter_b = PacketFlow(
        2, mem, cores[1], src_channel=scatter.endpoint(mem), dst_channel=None
    )

    staged = Buffer(tile=mem, type=vec_ty, name="staged")
    l1 = [Buffer(tile=c, type=vec_ty, name=f"l1_{i}") for i, c in enumerate(cores)]
    mem_dma = TileDma(
        mem,
        [
            DmaChannel(DMAChannelDir.S2MM, gather.endpoint(mem), [Bd(staged)]),
            DmaChannel(
                DMAChannelDir.MM2S,
                scatter_b.endpoint(mem),
                [Bd(staged, packet=(0, 1)), Bd(staged, packet=(0, 2))],
            ),
        ],
    )
    core_dmas = [
        TileDma(
            core,
            [
                DmaChannel(DMAChannelDir.MM2S, send.endpoint(core), [Bd(buf, packet=(0, pid))]),
                DmaChannel(DMAChannelDir.S2MM, recv.endpoint(core), [Bd(buf)]),
            ],
        )
        for core, buf, send, recv, pid in (
            (cores[0], l1[0], gather, scatter, 1),
            (cores[1], l1[1], gather_b, scatter_b, 2),
        )
    ]

    rt = Runtime(lambda: None, [])
    for fl in (gather, gather_b, scatter, scatter_b):
        rt.add_flow(fl)
    for td in (mem_dma, *core_dmas):
        rt.add_tile_dma(td)
    return Program(NPU2Col1(), rt).resolve_program()


# The sharing ends name the first flow's endpoint and emit none of their own;
# allocation would refuse one no route names.

# CHECK-DAG: aie.route_endpoint @packetflow0_dst(%{{.*}}) DMA
# CHECK-DAG: aie.route_endpoint @packetflow2_src(%{{.*}}) DMA
# CHECK-DAG: aie.route from @packetflow0_src to [@packetflow0_dst] {keep_pkt_header, packet, packet_id = 1 : i8}
# CHECK-DAG: aie.route from @packetflow1_src to [@packetflow0_dst] {keep_pkt_header, packet, packet_id = 2 : i8}
# CHECK-DAG: aie.route from @packetflow2_src to [@packetflow2_dst] {packet, packet_id = 1 : i8}
# CHECK-DAG: aie.route from @packetflow2_src to [@packetflow3_dst] {packet, packet_id = 2 : i8}

# ALLOC-LABEL: aie.memtile_dma
# ALLOC-DAG: aie.dma_start(S2MM, 0,
# ALLOC-DAG: aie.dma_start(MM2S, 0,

print(build())
