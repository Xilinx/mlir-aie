# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-assign-lock-ids \
# RUN:   --aie-assign-buffer-addresses --aie-assign-runtime-sequence-bd-ids \
# RUN:   --aie-dma-tasks-to-npu | FileCheck %s --check-prefix=LOWER

"""tile_dma_chain(out_of_order=True) arms an out-of-order S2MM merge from the
runtime sequence: each Bd pins the bd_id that senders name in their packet
headers, and repeat_count counts the packets the channel accepts."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.iron import (
    Bd,
    Buffer,
    Flow,
    Lock,
    Program,
    Release,
    Runtime,
    tile_dma_chain,
)
from aie.iron.device import NPU2Col1, Tile

SLOTS = 4
SLOT = 16


def emit_merge(bad=None):
    mem_tile = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    core_tile = Tile(col=0, row=2, tile_type=AIETileType.CoreTile)
    merged = Buffer(
        tile=mem_tile, type=np.ndarray[(SLOTS * SLOT,), np.dtype[np.int32]], name="m"
    )
    done = Lock(mem_tile, init=0, name="done")
    into = Flow(core_tile, mem_tile)

    def slot_bds():
        bds = [
            Bd(
                merged,
                offset=i * SLOT,
                length=SLOT,
                bd_id=3 + 2 * i,
                packet=(0, 0),
                releases=[Release(done, value=1)],
            )
            for i in range(SLOTS)
        ]
        if bad == "bd_id":
            bds[1].bd_id = None
        elif bad == "packet":
            bds[2].packet = None
        return bds

    def sequence(_host):
        direction = DMAChannelDir.MM2S if bad == "direction" else DMAChannelDir.S2MM
        channel = into.endpoint(mem_tile) if bad == "endpoint" else 0
        tile_dma_chain(
            mem_tile,
            direction,
            channel,
            slot_bds(),
            repeat_count=2 * SLOTS - 1,
            out_of_order=True,
        )

    rt = Runtime(sequence, [np.ndarray[(SLOT,), np.dtype[np.int32]]])
    rt.add_flow(into)
    rt.add_buffer(merged)
    rt.add_lock(done)
    return Program(NPU2Col1(), rt).resolve_program()


# One pinned-id, packet-enabled BD per slot; out of order, the repeat count is
# the packet count minus one. The header comes from the dma_bd's packet
# attribute, not an aie.dma_bd_packet op.
# CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 0) {
# CHECK-NOT: aie.dma_bd_packet
# CHECK:   aie.dma_bd(%{{.*}} : memref<64xi32> len = 16) {bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 0>}
# CHECK:   aie.use_lock(%done, Release, %{{.*}})
# CHECK:   aie.next_bd ^bb1
# CHECK:   aie.dma_bd(%{{.*}} : memref<64xi32> offset = 48 len = 16) {bd_id = 9 : i32, packet = {{.*}}}
# CHECK:   aie.end
# CHECK: } {out_of_order, repeat_count = 7 : i32}
# CHECK: aiex.dma_start_task(%[[T]])

# LOWER: aiex.npu.writebd {bd_id = 3 : i32, {{.*}}enable_packet = 1
# LOWER: aiex.npu.writebd {bd_id = 9 : i32, {{.*}}enable_packet = 1
print(emit_merge())

for bad in ("bd_id", "packet", "direction", "endpoint"):
    try:
        emit_merge(bad)
    except ValueError as e:
        print(f"// {bad}: {e}")
# CHECK: // bd_id: tile_dma_chain out_of_order Bd 1 must set bd_id and packet; senders address it by its bd_id.
# CHECK: // packet: tile_dma_chain out_of_order Bd 2 must set bd_id and packet; senders address it by its bd_id.
# CHECK: // direction: tile_dma_chain out_of_order is only valid for S2MM, not {{.*}}MM2S
# CHECK: // endpoint: tile_dma_chain out_of_order needs an integer channel, not the Flow endpoint @flow0_dst.
