//===- arbiter_transit_control_flow.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// The hazard between a flow this tile emits and a co-tenant that has yet to
// leave the switchbox is symmetric, and both orders reach the allocator.
// arbiter_stall_isolation.mlir covers the transit flow arriving second; here it
// arrives first, which only a control flow can do, since those are allocated
// ahead of everything else and a slave port on DMA otherwise sorts before one
// on North.
//
// Flow 9 transits (0,1) southward and takes arbiter 5, the highest, being a
// control flow. Flows 0..4 then fill msel 0 on arbiters 0..4, so flow 5 --
// emitted by this tile, and so able to stall on the tile that consumes it --
// first reaches a free amsel on arbiter 5. Sharing with a transit flow is
// exactly what deadlocks, and no arbiter here is free of one, so flow 5 keeps
// the allocation it would have had and the conflict is reported instead.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %s4 = aie.tile(4, 0)
    %s5 = aie.tile(5, 0)
    %m  = aie.tile(0, 1)
    %c5 = aie.tile(0, 5)

    // Allocated first, and southbound, so it passes through (0,1).
    aie.packet_flow(9) {
      aie.packet_source<%c5, DMA : 0>
      aie.packet_dest<%s0, DMA : 0>
    } {priority_route = true}

    // Emitted here, one per DMA channel, filling msel 0 on arbiters 0..4.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s4, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s5, DMA : 0> }

    // Placed last, and the one that has to share.
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s0, DMA : 1> }
  }
}

// WARN: warning: at tile (0, 1), packet flow 5 shares arbiter 5 with packet flow 9, which it can deadlock against

// Both end up on arbiter 5, which is the point: the allocation is unchanged
// and the warning is the whole of the reaction.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SHARED:.*]] = aie.amsel<5> (0)
// CHECK:         %[[TRANSIT:.*]] = aie.amsel<5> (3)
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 5, %[[SHARED]])
// CHECK:         aie.packet_rules(North : 2) {
// CHECK-NEXT:      aie.rule(31, 9, %[[TRANSIT]])
