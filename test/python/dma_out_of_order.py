# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Test the IRON out-of-order S2MM channel API.

DmaChannel(out_of_order=True) lowers to an aie.dma_start with the out_of_order
attribute, pins each receive BD's id to its position (so it equals the header
out-of-order id that selects it), auto-chains the receive BDs (the BDs below set
no `next`), and derives the merge's packets-per-round from the receive BDs (the
sum of each BdIteration size) repeated `repeat_count + 1` times. Also checks:
out_of_order rejected on a non-S2MM channel; a receive BD that is not
packet-enabled; a bd_id COLLISION between two out-of-order channels on a tile;
and that two out-of-order channels DO share a tile when their bd_ids are
disjoint.
"""

import numpy as np
from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import Bd, BdIteration, Buffer, DmaChannel, Program, Runtime, TileDma
from aie.iron.device import NPU2Col1, Tile


def emit(channels):
    tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    buf = Buffer(tile=tile, type=np.ndarray[(2,), np.dtype[np.int32]], name="buf")
    chans = []
    for spec in channels:
        direction, channel, nbds, ooo, repeat, pkt = spec[:6]
        bd_ids = spec[6] if len(spec) > 6 else [None] * nbds
        it = spec[7] if len(spec) > 7 else 1
        its = it if isinstance(it, list) else [it] * nbds
        # No `next` set: an out-of-order channel auto-chains its receive BDs.
        bds = [
            Bd(
                buffer=buf,
                offset=4 * i,
                length=4,
                packet=(0, 0) if pkt else None,
                bd_id=bd_ids[i],
                iteration=BdIteration(size=its[i], stride=1) if its[i] > 1 else None,
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

    rt = Runtime(sequence, [np.ndarray[(2,), np.dtype[np.int32]]])
    rt.add_tile_dma(tile_dma)
    return Program(NPU2Col1(), rt).resolve_program()


# A single out-of-order S2MM channel: the attribute is on the dma_start and the
# receive BDs are pinned to bd_id 0 and 1. The packet count is DERIVED from the
# receive BDs: 2 BDs, no iteration -> 2 packets -> repeat_count 1.
# CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 1) {out_of_order}
# CHECK: aie.dma_bd({{.*}}) {bd_id = 0 : i32
# CHECK: aie.dma_bd({{.*}}) {bd_id = 1 : i32
print(emit([(DMAChannelDir.S2MM, 0, 2, True, 0, True)]))

# The derived count folds in each BD's iteration: 2 BDs x BdIteration(size=3) =
# 6 packets -> repeat_count 5.
# CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 5) {out_of_order}
print(emit([(DMAChannelDir.S2MM, 0, 2, True, 0, True, [None, None], 3)]))

# Iteration sizes can differ: sizes 2 and 3 sum to 5 packets -> repeat_count 4.
# CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 4) {out_of_order}
print(emit([(DMAChannelDir.S2MM, 0, 2, True, 0, True, [None, None], [2, 3])]))

# repeat_count repeats the whole round: 2 BDs (2 packets/round) x (1+1) rounds
# = 4 packets -> repeat_count 3.
# CHECK: aie.dma_start(S2MM, 0, {{.*}}, repeat_count = 3) {out_of_order}
print(emit([(DMAChannelDir.S2MM, 0, 2, True, 1, True)]))

# out_of_order is rejected on an MM2S channel.
# CHECK: REJECT MM2S: out_of_order is only valid for an S2MM DmaChannel
try:
    emit([(DMAChannelDir.MM2S, 0, 2, True, 0, True)])
    print("MM2S out_of_order was NOT rejected")
except ValueError as e:
    print(f"REJECT MM2S: {e}")

# A receive BD must be packet-enabled.
# CHECK: REJECT NOPKT: out_of_order channel 0 BD at slot 0 must be packet-enabled
try:
    emit([(DMAChannelDir.S2MM, 0, 2, True, 0, False)])
    print("non-packet out_of_order BD was NOT rejected")
except ValueError as e:
    print(f"REJECT NOPKT: {e}")

# Two out-of-order channels on one tile whose bd_ids collide are rejected.
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

# But two out-of-order channels DO share a tile when their bd_ids are disjoint.
# CHECK: TWO OOO OK
# CHECK: aie.dma_start(S2MM, 0, {{.*}} {out_of_order}
# CHECK: aie.dma_start(S2MM, 1, {{.*}} {out_of_order}
print("TWO OOO OK")
print(
    emit(
        [
            (DMAChannelDir.S2MM, 0, 2, True, 0, True),
            (DMAChannelDir.S2MM, 1, 2, True, 0, True, [2, 3]),
        ]
    )
)

# Bd.bd_id pins the descriptor slot on an ORDINARY (in-order) channel too, not
# only out-of-order.
# CHECK: IN-ORDER BD_ID OK
# CHECK: aie.dma_bd({{.*}}) {bd_id = 7 : i32
print("IN-ORDER BD_ID OK")
print(emit([(DMAChannelDir.S2MM, 0, 1, False, 0, False, [7])]))
