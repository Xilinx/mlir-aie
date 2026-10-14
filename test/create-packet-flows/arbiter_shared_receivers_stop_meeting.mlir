//===- arbiter_shared_receivers_stop_meeting.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=ERR
// RUN: aie-opt --split-input-file --aie-create-pathfinder-flows="allow-deadlock-prone=true" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN

// Trees that cannot meet route without meeting, leaving a hold cycle through
// the receivers they share: an error, or a warning with allow-deadlock-prone.

// Meeting put ids 28 and 29 on one arbiter at (0, 4), in a hold cycle.
// Reduced from router_mutation.py npu1 seed 207360280 (move-endpoint).

// ERR: error: Unable to find a legal routing: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 3) Core:0 -> (0, 0) DMA:1 (id 28)
// WARN: warning: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 3) Core:0 -> (0, 0) DMA:1 (id 28)

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.packet_flow(29) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(28) { aie.packet_source<%t_0_3, Core : 0> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(23) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 1> }
    aie.packet_flow(11) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_5, DMA : 1> }
  }
}

// -----

// Joining (0, 0)'s id 3 tree at (0, 3) ties (0, 3) DMA:1's ids 4 and 20 to its
// arbiter; the check undoes the join rather than split them across channels
// (0, 3) does not have.
// Reduced from router_mutation.py npu1 seed 207380308 (multi-source).

// ERR: error: Unable to find a legal routing: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 0) DMA:0 -> (0, 1) DMA:4 (id 3)
// WARN: warning: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 0) DMA:0 -> (0, 1) DMA:4 (id 3)

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_3, DMA : 0, %t_0_2, DMA : 1)
    aie.packet_flow(4) { aie.packet_source<%t_0_5, Core : 0> aie.packet_source<%t_0_3, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(20) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(3) { aie.packet_source<%t_0_3, DMA : 1> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_5, DMA : 1> }
  }
}

// -----

// The same with id 18 from (0, 2) DMA:1 and (0, 0) DMA:1, joining at which
// would tie (0, 2)'s ids 0 and 24 to (0, 0)'s arbiter.
// Reduced from router_properties.py npu1 seed 207360700.

// ERR: error: Unable to find a legal routing: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 0) DMA:1 -> (0, 1) DMA:4 (id 18)
// WARN: warning: Packet flows into receivers they share can deadlock holding arbiters across switchboxes, and no routing found avoids it: packet flow (0, 0) DMA:1 -> (0, 1) DMA:4 (id 18)

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_5 = aie.tile(0, 5)
    aie.flow(%t_0_3, DMA : 0, %t_0_1, DMA : 3)
    aie.packet_flow(24) { aie.packet_source<%t_0_2, DMA : 1> aie.packet_dest<%t_0_0, DMA : 0> }
    aie.packet_flow(0) { aie.packet_source<%t_0_2, DMA : 1> aie.packet_dest<%t_0_1, DMA : 2> }
    aie.packet_flow(18) { aie.packet_source<%t_0_2, DMA : 1> aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 4> aie.packet_dest<%t_0_3, DMA : 0> }
    aie.packet_flow(29) { aie.packet_source<%t_0_3, Core : 0> aie.packet_dest<%t_0_1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 5> }
  }
}
