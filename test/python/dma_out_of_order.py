# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test the IRON out-of-order S2MM channel API: DmaChannel(out_of_order=True)
lowers to an aie.dma_start with the out_of_order attribute, pins each receive
BD's id to its position (so it equals the header out-of-order id that selects
it), forwards repeat_count, and auto-chains the receive BDs (the BDs below set
no `next`). Also checks: out_of_order rejected on a non-S2MM channel; a receive
BD that is not packet-enabled; a bd_id COLLISION between two out-of-order
channels on a tile; and that two out-of-order channels DO share a tile when
their bd_ids are disjoint (default ids on one, explicit Bd.bd_id on the other)."""

import numpy as np

from aie.iron import Bd, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir


def emit(channels):
    tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=tile, type=np.ndarray[(16,), np.dtype[np.int32]], name="buf")
    chans = []
    for spec in channels:
        direction, channel, nbds, ooo, repeat, pkt = spec[:6]
        # Optional 7th element: explicit bd_ids for this channel's BDs.
        bd_ids = spec[6] if len(spec) > 6 else [None] * nbds
        # No `next` set: an out-of-order channel auto-chains its receive BDs.
        bds = [
            Bd(
                buffer=buf,
                offset=4 * i,
                length=4,
                packet=(0, 0) if pkt else None,
                bd_id=bd_ids[i],
            )
            for i in range(nbds)
        ]
        chans.append(
            DmaChannel(
                direction=direction,
                channel=channel,
                bds=bds,
                repeat_count=repeat,
                out_of_order=ooo,
            )
        )
    tile_dma = TileDma(tile=tile, channels=chans)

    def sequence(_):
        pass

    rt = Runtime(sequence, [np.ndarray[(16,), np.dtype[np.int32]]])
    rt.add_tile_dma(tile_dma)
    return Program(NPU2Col1(), rt).resolve_program()


# A single out-of-order S2MM channel: the attribute is on the dma_start, the
# receive BDs are pinned to bd_id 0 and 1, and repeat_count is forwarded.
# CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 3) {out_of_order}
# CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32
# CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32
print(emit([(DMAChannelDir.S2MM, 0, 2, True, 3, True)]))

# out_of_order is rejected on an MM2S channel.
# CHECK: REJECT MM2S: out_of_order is only valid for an S2MM DmaChannel
try:
    emit([(DMAChannelDir.MM2S, 0, 2, True, 0, True)])
    print("MM2S out_of_order was NOT rejected")
except ValueError as e:
    print(f"REJECT MM2S: {e}")

# A receive BD that is not packet-enabled is rejected (placement needs the
# header, which only exists on a packet-enabled BD).
# CHECK: REJECT NOPKT: out_of_order channel 0 BD at slot 0 must be packet-enabled
try:
    emit([(DMAChannelDir.S2MM, 0, 2, True, 0, False)])
    print("non-packet out_of_order BD was NOT rejected")
except ValueError as e:
    print(f"REJECT NOPKT: {e}")

# Two out-of-order channels on one tile whose bd_ids collide are rejected (they
# share the tile's bd_id space; both default to ids 0,1 here).
# CHECK: REJECT COLLISION: out_of_order bd_id 0 is used by more than one BD
try:
    emit(
        [
            (DMAChannelDir.S2MM, 0, 2, True, 0, True),
            (DMAChannelDir.S2MM, 1, 2, True, 0, True),
        ]
    )
    print("out_of_order channel collision was NOT rejected")
except ValueError as e:
    print(f"REJECT COLLISION: {e}")

# But two out-of-order channels DO share a tile when their bd_ids are disjoint:
# channel 0 keeps the default ids 0,1; channel 1 pins its slots to ids 2,3.
# CHECK: TWO OOO OK
# CHECK: aie.dma_start(S2MM, 0, {{.*}} {out_of_order}
# CHECK: aie.dma_start(S2MM, 1, {{.*}} {out_of_order}
print("TWO OOO OK")
print(
    emit(
        [
            (DMAChannelDir.S2MM, 0, 2, True, 2, True),
            (DMAChannelDir.S2MM, 1, 2, True, 2, True, [2, 3]),
        ]
    )
)

# Bd.bd_id pins the descriptor slot on an ORDINARY (in-order) channel too, not
# only out-of-order: assign-bd-ids reserves a pinned id regardless of ordering.
# Here an in-order S2MM pins id 7 (auto-assign would give 0).
# CHECK: IN-ORDER BD_ID OK
# CHECK: aie.dma_bd({{.*}}) {bd_id = 7 : i32
print("IN-ORDER BD_ID OK")
print(emit([(DMAChannelDir.S2MM, 0, 1, False, 0, False, [7])]))
