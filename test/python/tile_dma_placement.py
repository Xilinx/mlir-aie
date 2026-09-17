# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test how a TileDma finds its tile.

Which region op it opens, and that a tile ends up with exactly one DMA program.

Both are failures that pass verification. A TileDma that reads an unset
tile_type hint at face value opens an aie.mem for a shim tile, and two TileDma
programs on one tile emit two regions for it -- neither is rejected downstream,
and both produce hardware that quietly does nothing.
"""

import numpy as np
from aie.dialects._aie_enum_gen import DMAChannelDir
from aie.iron import (
    Bd,
    Buffer,
    DmaChannel,
    ExternalBuffer,
    Program,
    Runtime,
    TileDma,
)
from aie.iron.device import NPU2Col1, Tile

vector_ty = np.ndarray[(64,), np.dtype[np.int32]]


def channel(buf, direction=DMAChannelDir.MM2S, index=0):
    return DmaChannel(direction=direction, channel=index, bds=[Bd(buffer=buf)])


# A Tile carries no tile_type here: the Device infers each from its
# coordinates, and the region op has to follow that rather than the unset hint.
# Note AIETileType.CoreTile is 0, so a truthiness test on the hint reads an
# explicitly-typed core tile as unset too.
# CHECK-LABEL: region_op_follows_the_resolved_tile
def region_op_follows_the_resolved_tile():
    print("\nTEST: region_op_follows_the_resolved_tile")
    rt = Runtime(lambda: None, [])
    for row in (0, 1, 2):  # shim, memtile, core on NPU2
        tile = Tile(0, row)
        # A shim tile has no local memory, so its BD names off-chip memory.
        if row == 0:
            buf = ExternalBuffer(vector_ty, address=0x8000_0000, name="off_chip")
            rt.add_external_buffer(buf)
        else:
            buf = Buffer(tile=tile, type=vector_ty, name=f"buf_row{row}")
        rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf)]))
    print(Program(NPU2Col1(), rt).resolve_program())


# CHECK: aie.shim_dma(
# CHECK: aie.memtile_dma(
# CHECK: aie.mem(


# CHECK-LABEL: one_dma_program_per_tile
def one_dma_program_per_tile():
    print("\nTEST: one_dma_program_per_tile")
    rt = Runtime(lambda: None, [])
    tile = Tile(0, 2)
    buf = Buffer(tile=tile, type=vector_ty, name="shared")
    # Registered separately, as a helper wiring one transfer at a time would.
    rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf, index=0)]))
    rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf, index=1)]))
    print(Program(NPU2Col1(), rt).resolve_program())


# One region, holding both channels.
# CHECK:      aie.mem(
# CHECK:        aie.dma_start(MM2S, 0
# CHECK:        aie.dma_start(MM2S, 1
# CHECK-NOT:  aie.mem(


# CHECK-LABEL: distinct_tiles_at_one_coordinate_are_rejected
def distinct_tiles_at_one_coordinate_are_rejected():
    print("\nTEST: distinct_tiles_at_one_coordinate_are_rejected")
    rt = Runtime(lambda: None, [])
    for i in range(2):
        tile = Tile(0, 2)  # same place, different object -- cannot be merged
        buf = Buffer(tile=tile, type=vector_ty, name=f"b{i}")
        try:
            rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf, index=i)]))
        except Exception as e:  # noqa: BLE001 - the message is the assertion
            print(f"{type(e).__name__}: {e}")


# CHECK: IronRuntimeError: Two TileDma programs name Tile(0, 2)


region_op_follows_the_resolved_tile()
one_dma_program_per_tile()
distinct_tiles_at_one_coordinate_are_rejected()
