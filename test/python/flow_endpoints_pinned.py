# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectfifo-allocate \
# RUN:   | FileCheck %s --check-prefix=ALLOC

"""A Flow that pins one end's channel still lowers to route endpoints when
another end is left to the compiler, and a DMA program on the pinned end names
that endpoint, as on an unpinned one. Allocation then keeps the pinned index;
had the program named the index itself, it would claim the channel the
endpoint pins and allocation would refuse the pin."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Bd, Buffer, DmaChannel, Flow, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile

N = 256
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def build():
    shim = Tile(tile_type=AIETileType.ShimNOCTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    cores = [Tile(tile_type=AIETileType.CoreTile) for _ in range(2)]

    into = Flow(shim, mem, dst_channel=5, name="into")
    spread = Flow(mem, cores, src_channel=2, name="spread")

    staged = Buffer(tile=mem, type=vec_ty, name="staged")
    landed = [
        Buffer(tile=c, type=vec_ty, name=f"landed{i}") for i, c in enumerate(cores)
    ]
    mem_dma = TileDma(
        mem, [DmaChannel(DMAChannelDir.S2MM, into.endpoint(mem), [Bd(staged)])]
    )
    core_dmas = [
        TileDma(
            c, [DmaChannel(DMAChannelDir.S2MM, spread.endpoint(c), [Bd(landed[i])])]
        )
        for i, c in enumerate(cores)
    ]

    def sequence(a):
        into.fill(a)
        spread.endpoint(mem).task(staged).start().free()

    rt = Runtime(sequence, [vec_ty])
    for fl in (into, spread):
        rt.add_flow(fl)
    for td in (mem_dma, *core_dmas):
        rt.add_tile_dma(td)
    return Program(NPU2Col1(), rt).resolve_program()


# CHECK-DAG: aie.route_endpoint @into_dst(%{{.*}}) DMA {channelIndex = 5 : i32}
# CHECK-DAG: aie.route_endpoint @spread_src(%{{.*}}) DMA {channelIndex = 2 : i32}
# CHECK-DAG: aie.dma_start(S2MM, @into_dst,
# CHECK-DAG: aiex.dma_configure_task_for @spread_src {

# ALLOC-LABEL: aie.memtile_dma
# ALLOC: aie.dma_start(S2MM, 5,
# ALLOC-LABEL: aie.runtime_sequence
# ALLOC: aiex.dma_configure_task(%{{.*}}, MM2S, 2)
print(build())
