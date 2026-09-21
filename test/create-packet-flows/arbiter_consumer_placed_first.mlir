//===- arbiter_consumer_placed_first.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>/dev/null | FileCheck %s

// Direct coupling is symmetric, but the order the allocator meets it in is
// not: an emitted flow's slave port is DMA, which sorts ahead of the North an
// inbound flow arrives on. arbiter_deadlock_exhausted.mlir covers that order;
// here it is reversed, since control flow 9 is placed before everything.
//
// Flows 0..4 fill msel 0 on arbiters 0..4, leaving flow 5 with nothing free:
// arbiter 5 holds flow 9, which feeds the very DMA flow 5 drains. So flow 5
// wraps onto arbiter 0, a StallShape hazard (flow 0 can stall, flow 5 has yet
// to leave the switchbox) -- preferred over the demonstrated cycle, and named.

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

    // Placed first, and consumed by the memtile.
    aie.packet_flow(9) {
      aie.packet_source<%c5, DMA : 0>
      aie.packet_dest<%m, DMA : 0>
    } {priority_route = true}

    // Emitted here, filling msel 0 on arbiters 0..4.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%s1, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%s2, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%s3, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s4, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s5, DMA : 0> }

    // Placed last, and the one that has to share.
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s0, DMA : 0> }
  }
}

// WARN: warning: at tile (0, 1), packet flow 5 shares arbiter 0 with packet flow 0, which it can deadlock against

// Flow 9 keeps arbiter 5 to itself; flow 5 doubles up on arbiter 0.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FILLER:.*]] = aie.amsel<0> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<0> (1)
// CHECK:         %[[CTRL:.*]] = aie.amsel<5> (3)
// CHECK:         aie.masterset(DMA : 0, %[[CTRL]])
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 5, %[[SHARED]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 0, %[[FILLER]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 9, %[[CTRL]])
