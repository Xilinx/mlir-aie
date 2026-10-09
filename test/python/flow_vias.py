# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.dialects._aie_enum_gen import AIETileType, WireBundle
from aie.iron import Flow, Program, Runtime
from aie.iron.device import NPU2Col1, Tile


def build():
    shim = Tile(0, 0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(0, 1, tile_type=AIETileType.MemTile)
    core = Tile(0, 2, tile_type=AIETileType.CoreTile)
    flow = Flow(
        core,
        shim,
        src_channel=0,
        dst_channel=0,
        vias=[
            (core, (WireBundle.DMA, 0), (WireBundle.South, 0)),
            (mem, (WireBundle.North, 0), (WireBundle.South, 0)),
            (shim, (WireBundle.North, 0), (WireBundle.DMA, 0)),
        ],
        shim_symbol="output",
    )
    assert flow.endpoint(core).channel == 0
    assert flow.endpoint(shim).channel == 0

    runtime = Runtime(lambda: None, [])
    runtime.add_flow(flow)
    return Program(NPU2Col1(), runtime).resolve_program()


# CHECK-DAG: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(0, 0)
# CHECK-DAG: %[[CORE:.*]] = aie.logical_tile<CoreTile>(0, 2)
# CHECK-DAG: %[[MEM:.*]] = aie.logical_tile<MemTile>(0, 1)
# CHECK: aie.flow(%[[CORE]], DMA : 0, %[[SHIM]], DMA : 0) via (%[[CORE]] : DMA : 0 -> South : 0, %[[MEM]] : North : 0 -> South : 0, %[[SHIM]] : North : 0 -> DMA : 0)
# CHECK: aie.shim_dma_allocation @output(%[[SHIM]], S2MM, 0)
print(build())


def expect_error(label, fn):
    try:
        fn()
    except ValueError as error:
        print(f"// {label}: {error}")


def assigned_channels():
    Flow(
        Tile(tile_type=AIETileType.CoreTile),
        Tile(tile_type=AIETileType.MemTile),
        vias=[],
    )


def assigned_channel_with_via():
    core = Tile(tile_type=AIETileType.CoreTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    Flow(core, mem, vias=[(core, (WireBundle.DMA, 0), (WireBundle.South, 0))])


def broadcast_with_via():
    core = Tile(tile_type=AIETileType.CoreTile)
    mem = Tile(tile_type=AIETileType.MemTile)
    Flow(
        core,
        [mem, Tile(tile_type=AIETileType.MemTile)],
        src_channel=0,
        dst_channel=0,
        vias=[(core, (WireBundle.DMA, 0), (WireBundle.South, 0))],
    )


assigned_channels()
expect_error("assigned_channel_with_via", assigned_channel_with_via)
expect_error("broadcast_with_via", broadcast_with_via)
# CHECK: // assigned_channel_with_via: Flow vias require one destination and explicit source and destination channels.
# CHECK: // broadcast_with_via: Flow vias require one destination and explicit source and destination channels.
