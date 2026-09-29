# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-objectfifo-allocate \
# RUN:   | FileCheck %s --check-prefix=ALLOC

"""A Flow whose channels are left out lowers to route endpoints the compiler
assigns, and a DMA program -- static (TileDma) or a runtime task -- names an
end through Flow.endpoint instead of an index. No tile is pinned
either: nothing in the design depends on where the ends land or which
channels they get."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir, WireBundle
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DmaChannel,
    Flow,
    Lock,
    Program,
    Release,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1, Tile

N = 256
vec_ty = np.ndarray[(N,), np.dtype[np.int32]]


def build(bad=None):
    shim = Tile(tile_type=AIETileType.ShimNOCTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    cores = [Tile(tile_type=AIETileType.CoreTile) for _ in range(2)]

    into = Flow(shim, mem)
    spread = Flow(mem, cores)
    out = Flow(cores[0], shim)
    # Both channels given: still a plain aie.flow, and endpoint() lowers to the
    # index.
    side = Flow(cores[1], mem, src_channel=0, dst_channel=3)

    staged = Buffer(tile=mem, type=vec_ty, name="staged")
    landed = [
        Buffer(tile=c, type=vec_ty, name=f"landed{i}") for i, c in enumerate(cores)
    ]
    side_in = Buffer(tile=mem, type=vec_ty, name="side_in")
    # Found from the Bd that uses it; no rt.add_lock.
    side_free = Lock(mem, init=1, name="side_free")
    side_full = Lock(mem, init=0, name="side_full")

    mem_dma = TileDma(
        mem,
        [
            DmaChannel(DMAChannelDir.S2MM, into.endpoint(mem), [Bd(staged)]),
            DmaChannel(
                DMAChannelDir.S2MM,
                side.endpoint(mem),
                [
                    Bd(
                        side_in,
                        acquires=[Acquire(side_free)],
                        releases=[Release(side_full)],
                    )
                ],
            ),
        ],
    )
    core_dmas = [
        TileDma(
            cores[0],
            [
                DmaChannel(
                    DMAChannelDir.S2MM, spread.endpoint(cores[0]), [Bd(landed[0])]
                ),
                DmaChannel(DMAChannelDir.MM2S, out.endpoint(cores[0]), [Bd(landed[0])]),
            ],
        ),
        TileDma(
            cores[1],
            [
                DmaChannel(
                    DMAChannelDir.S2MM, spread.endpoint(cores[1]), [Bd(landed[1])]
                ),
                DmaChannel(
                    DMAChannelDir.MM2S, side.endpoint(cores[1]), [Bd(landed[1])]
                ),
            ],
        ),
    ]

    def sequence(a, c):
        into.fill(a)
        # The broadcast's source is programmed from the sequence.
        spread.endpoint(mem).task(staged).start().free()
        out.drain(c, wait=True)

    rt = Runtime(sequence, [vec_ty, vec_ty])
    for fl in (into, spread, out, side):
        rt.add_flow(fl)
    for td in (mem_dma, *core_dmas):
        rt.add_tile_dma(td)
    return Program(NPU2Col1(), rt).resolve_program()


# Ends are named after the Flow (flow{n} by default). The shim end carries
# fifoName, which is what later gives the sequence a shim DMA allocation.

# CHECK-DAG: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 3)
# CHECK-DAG: %side_free = aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "side_free"}
# CHECK-DAG: %side_full = aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "side_full"}
# CHECK-DAG: aie.route_endpoint @flow0_src(%{{.*}}) DMA {fifoName = "flow0_src"}
# CHECK-DAG: aie.route_endpoint @flow0_dst(%{{.*}}) DMA
# CHECK-DAG: aie.route from @flow0_src to [@flow0_dst]
# CHECK-DAG: aie.route_endpoint @flow1_src(%{{.*}}) DMA
# CHECK-DAG: aie.route_endpoint @flow1_dst0(%{{.*}}) DMA
# CHECK-DAG: aie.route_endpoint @flow1_dst1(%{{.*}}) DMA
# CHECK-DAG: aie.route from @flow1_src to [@flow1_dst0, @flow1_dst1]
# CHECK-DAG: aie.route_endpoint @flow2_dst(%{{.*}}) DMA {fifoName = "flow2_dst"}
# CHECK-DAG: aie.route from @flow2_src to [@flow2_dst]

# CHECK-DAG: aie.dma_start(S2MM, @flow0_dst,
# CHECK-DAG: aie.dma_start(S2MM, 3,
# CHECK-DAG: aie.dma_start(S2MM, @flow1_dst0,
# CHECK-DAG: aie.dma_start(MM2S, @flow2_src,
# CHECK-DAG: aie.dma_start(S2MM, @flow1_dst1,
# CHECK-DAG: aie.dma_start(MM2S, 0,
# CHECK-DAG: aiex.dma_configure_task_for @flow0_src
# CHECK-DAG: aiex.dma_configure_task_for @flow1_src {
# CHECK-DAG: aiex.dma_configure_task_for @flow2_dst

# After allocation every program names an index, the runtime chain is
# configured on the memtile channel the broadcast's flows leave from, and the
# sequence reaches the shim ends through allocations.
# ALLOC-LABEL: aie.memtile_dma
# ALLOC: aie.dma_start(S2MM, {{[0-9]}},
# ALLOC: aie.dma_start(S2MM, 3,
# ALLOC: aie.mem(
# ALLOC: aie.dma_start(S2MM, {{[0-9]}},
# ALLOC: aie.dma_start(MM2S, {{[0-9]}},
# ALLOC: aie.mem(
# ALLOC: aie.dma_start(S2MM, {{[0-9]}},
# ALLOC: aie.dma_start(MM2S, 0,

# ALLOC-LABEL: aie.runtime_sequence
# ALLOC: aiex.dma_configure_task_for @flow0_src_shim_alloc
# ALLOC: aiex.dma_configure_task(%[[MEM:.*]], MM2S, [[B:[0-9]]])
# ALLOC: aiex.dma_configure_task_for @flow2_dst_shim_alloc
# ALLOC: aie.route_endpoint @flow1_src(%[[MEM]]) DMA {channelIndex = [[B]] : i32}
# ALLOC-COUNT-2: aie.flow(%[[MEM]], DMA : [[B]], %{{.*}}, DMA : {{[0-9]}})
# ALLOC-DAG: aie.shim_dma_allocation @flow0_src_shim_alloc(%{{.*}}, MM2S, {{[0-9]}})
# ALLOC-DAG: aie.shim_dma_allocation @flow2_dst_shim_alloc(%{{.*}}, S2MM, {{[0-9]}})
print(build())


# Printed as MLIR comments so the ALLOC run still parses the module above.
def expect_error(label, fn):
    try:
        fn()
    except (ValueError, RuntimeError) as e:
        print(f"// {label}: {e}")


def outside_sequence():
    mem, core = Tile(tile_type=AIETileType.MemTile), Tile(
        tile_type=AIETileType.CoreTile
    )
    Flow(mem, core).endpoint(mem).task(Buffer(tile=mem, type=vec_ty))


def shim_task():
    shim, mem = Tile(tile_type=AIETileType.ShimNOCTile), Tile(
        tile_type=AIETileType.MemTile
    )
    fl = Flow(shim, mem, name="feed")

    def sequence(a):
        fl.endpoint(shim).task(Buffer(tile=shim, type=vec_ty))

    rt = Runtime(sequence, [vec_ty])
    rt.add_flow(fl)
    Program(NPU2Col1(), rt).resolve_program()


def wrong_tile():
    mem, core = Tile(tile_type=AIETileType.MemTile), Tile(
        tile_type=AIETileType.CoreTile
    )
    fl = Flow(mem, core, name="misplaced")
    TileDma(core, [DmaChannel(DMAChannelDir.MM2S, fl.endpoint(mem), [])])


def wrong_direction():
    mem, core = Tile(tile_type=AIETileType.MemTile), Tile(
        tile_type=AIETileType.CoreTile
    )
    fl = Flow(mem, core, name="backwards")
    TileDma(core, [DmaChannel(DMAChannelDir.MM2S, fl.endpoint(core), [])])


def not_an_end():
    mem, core = Tile(tile_type=AIETileType.MemTile), Tile(
        tile_type=AIETileType.CoreTile
    )
    Flow(mem, core).endpoint(Tile(tile_type=AIETileType.CoreTile))


def core_port():
    Flow(Tile(0, 2), Tile(0, 3), src_port=WireBundle.Core)


expect_error("outside_sequence", outside_sequence)
expect_error("shim_task", shim_task)
expect_error("wrong_tile", wrong_tile)
expect_error("wrong_direction", wrong_direction)
expect_error("not_an_end", not_an_end)
expect_error("core_port", core_port)
# CHECK: // outside_sequence: No active runtime sequence: {{.*}}DmaEndpoint.task(){{.*}}
# CHECK: // shim_task: @feed_src is a shim channel, which moves host memory; use the Flow's fill()/drain() for it.
# CHECK: // wrong_tile: DMA endpoint @misplaced_src is on {{.*}}, not {{.*}}; a DMA program can only run its own tile's channels.
# CHECK: // wrong_direction: DMA endpoint @backwards_dst is S2MM, not MM2S.
# CHECK: // not_an_end: Tile{{.*}} is not an end of this Flow.
# CHECK: // core_port: Flow src_port=Core needs an explicit src_channel; the compiler only assigns DMA channels.
