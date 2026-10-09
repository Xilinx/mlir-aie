# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test how a TileDma finds its tile.

Which region op it opens, and that a tile ends up with exactly one DMA program.

Both are failures that pass verification. A TileDma that reads an unset
tile_type hint at face value opens an aie.mem for a memtile, and two TileDma
programs on one tile emit two regions for it -- neither is rejected downstream,
and both produce hardware that quietly does nothing.

Shim rows are not covered here: a BD on a shim names off-chip memory, which
an aie.buffer cannot be ("op exists in a tile with local memory"). An untyped
Tile(0, 0) is covered through Flow instead, in flow_fill_drain.py.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Bd,
    Buffer,
    DmaChannel,
    Program,
    Runtime,
    TileDma,
    Worker,
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
    for row in (1, 2):  # memtile, core on NPU2
        tile = Tile(0, row)
        buf = Buffer(tile=tile, type=vector_ty, name=f"buf_row{row}")
        rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf)]))
    print(Program(NPU2Col1(), rt).resolve_program())


# The memtile is the one that used to come out as an aie.mem.
# CHECK: aie.memtile_dma(
# CHECK: aie.mem(


# CHECK-LABEL: one_dma_program_per_tile
def one_dma_program_per_tile():
    print("\nTEST: one_dma_program_per_tile")
    rt = Runtime(lambda: rt.resolve_tile_dmas(), [])
    tile = Tile(0, 2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=tile, type=vector_ty, name="shared")
    # Registered separately, as a helper wiring one transfer at a time would.
    first = TileDma(tile=tile, channels=[channel(buf, index=0)])
    second = TileDma(tile=tile, channels=[channel(buf, index=1)])
    rt.add_tile_dma(first)
    rt.add_tile_dma(second)
    worker = Worker(lambda *_: None, [first, second], tile=tile, while_true=False)
    module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program()
    assert len(first.channels) == len(second.channels) == 1
    assert str(module).count("aie.mem(") == 1
    print(module)


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
        rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf, index=i)]))
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except Exception as e:  # noqa: BLE001 - the message is the assertion
        print(f"{type(e).__name__}: {e}")
    else:
        raise AssertionError("Expected conflicting TileDma registrations to fail")


# CHECK: IronRuntimeError: Two TileDma programs name Tile(0, 2)


def duplicate_channels_are_rejected():
    print("\nTEST: duplicate_channels_are_rejected")
    tile = Tile(0, 2)
    buf = Buffer(tile=tile, type=vector_ty, name="duplicate")
    try:
        TileDma(tile=tile, channels=[channel(buf), channel(buf)])
    except ValueError as e:
        print(f"initial: {e}")
    else:
        raise AssertionError("Expected duplicate initial channels to fail")

    rt = Runtime(lambda: None, [])
    registered = TileDma(tile=tile, channels=[channel(buf)])
    rt.add_tile_dma(registered)
    rt.add_tile_dma(TileDma(tile=tile, channels=[channel(buf, index=1), channel(buf)]))
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except ValueError as e:
        print(f"merged: {e}")
    else:
        raise AssertionError("Expected duplicate registered channels to fail")
    print(f"channels after rejection: {len(registered.channels)}")


# CHECK-LABEL: duplicate_channels_are_rejected
# CHECK: initial: TileDma for Tile(0, 2) already has MM2S channel 0.
# CHECK: merged: TileDma for Tile(0, 2) already has MM2S channel 0.
# CHECK: channels after rejection: 1


def repeated_registration_is_rejected():
    print("\nTEST: repeated_registration_is_rejected")
    tile = Tile(0, 2)
    buf = Buffer(tile=tile, type=vector_ty, name="repeated")
    registered = TileDma(tile=tile, channels=[channel(buf)])
    rt = Runtime(lambda: None, [])
    rt.add_tile_dma(registered)
    rt.add_tile_dma(registered)
    try:
        Program(NPU2Col1(), rt).resolve_program()
    except ValueError as e:
        print(e)
    else:
        raise AssertionError("Expected repeated registration to fail")


# CHECK-LABEL: repeated_registration_is_rejected
# CHECK: TileDma for Tile(0, 2) already has MM2S channel 0.


region_op_follows_the_resolved_tile()
one_dma_program_per_tile()
distinct_tiles_at_one_coordinate_are_rejected()
duplicate_channels_are_rejected()
repeated_registration_is_rejected()
