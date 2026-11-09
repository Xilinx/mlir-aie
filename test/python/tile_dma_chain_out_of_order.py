# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s
# RUN: %python %s | aie-opt --aie-place-tiles --aie-assign-lock-ids \
# RUN:   --aie-assign-buffer-addresses --aie-assign-runtime-sequence-bd-ids \
# RUN:   --aie-dma-tasks-to-npu | FileCheck %s --check-prefix=LOWER

"""DmaEndpoint.task(out_of_order=True) arms an out-of-order S2MM merge from the
runtime sequence: each Bd pins the bd_id that senders name in their packet
headers, and runs counts the packets the channel accepts."""

import numpy as np

from aie.dialects._aie_enum_gen import AIETileType, DMAChannelDir
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Bd,
    Buffer,
    DmaEndpoint,
    Flow,
    Lock,
    Program,
    Release,
    Runtime,
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
    into = Flow(core_tile, mem_tile, name="into")

    def slot_bds():
        bds = [
            Bd(
                merged,
                tap=slot,
                bd_id=3 + 2 * i,
                packet=(0, 0),
                releases=[Release(done, value=1)],
            )
            for i, slot in enumerate(
                TensorAccessPattern.full((SLOTS * SLOT,)).partition(SLOTS)
            )
        ]
        if bad == "bd_id":
            bds[1].bd_id = None
        elif bad == "packet":
            bds[2].packet = None
        return bds

    def sequence(_host):
        direction = DMAChannelDir.MM2S if bad == "direction" else DMAChannelDir.S2MM
        if bad == "endpoint":
            end = into.endpoint(mem_tile)
        else:
            end = DmaEndpoint(mem_tile, direction, 0)
        end.task(*slot_bds(), runs=2 * SLOTS, out_of_order=True).start()

    rt = Runtime(sequence, [np.ndarray[(SLOT,), np.dtype[np.int32]]])
    rt.add_flow(into)
    module = Program(NPU2Col1(), rt).resolve_program()
    module.operation.verify()
    return module


# One pinned-id, packet-enabled BD per slot; out of order, the repeat count is
# the packet count minus one. The header comes from the dma_bd's packet
# attribute, not an aie.dma_bd_packet op.
# CHECK: %[[T:.*]] = aiex.dma_configure_task(%{{.*}}, S2MM, 0) {
# CHECK-NOT: aie.dma_bd_packet
# CHECK:   aie.dma_bd(%{{.*}} : memref<64xi32> len = 16) {bd_id = 3 : i32, packet = #aie.packet_info<pkt_id = 0>}
# CHECK:   aie.use_lock(%done, Release, %{{.*}})
# CHECK:   aie.next_bd ^bb1
# CHECK:   aie.dma_bd(%{{.*}} : memref<64xi32> offset = 48 len = 16) {bd_id = 9 : i32, packet = {{.*}}}
# CHECK:   aie.end
# CHECK: } {out_of_order, repeat_count = 7 : i32}
# CHECK: aiex.dma_start_task(%[[T]])

# LOWER: aiex.npu.writebd {bd_id = 3 : i32, {{.*}}enable_packet = 1
# LOWER: aiex.npu.writebd {bd_id = 9 : i32, {{.*}}enable_packet = 1
print(emit_merge())

# Printed as MLIR comments so the LOWER run still parses the module above.
unpinned = str(emit_merge("bd_id")).splitlines()
print("// bd_id:", next(line for line in unpinned if "offset = 16 " in line))
for bad in ("packet", "direction", "endpoint"):
    try:
        emit_merge(bad)
    except Exception as e:
        print(f"// {bad}: {type(e).__name__}: {e}".replace("\n", " "))
# A Bd left without a bd_id takes its position in the task, as on a DmaChannel.
# CHECK: // bd_id: aie.dma_bd({{.*}} offset = 16 len = 16) {bd_id = 1 : i32
# CHECK: // packet: MLIRError: {{.*}}out-of-order S2MM receive buffer descriptor must be packet-enabled
# CHECK: // direction: MLIRError: {{.*}}out_of_order is only valid on an S2MM channel
# CHECK: // endpoint: ValueError: An out-of-order task needs a channel given by index, not the compiler-assigned @into_dst.
