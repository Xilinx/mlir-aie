//===- arbiter_stall_isolation.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) emits flows 0..3 and receives flows 4 and 5, filling all six
// arbiters. Flow 6 only passes through, so it has to share.
//
// An arbiter holds its grant until tlast, so a co-tenant that has yet to leave
// the switchbox waits behind whatever the grant holder is waiting on. None of
// flows 0..3 will do: each is emitted by this tile's DMA and stalls when its
// consumer falls behind, flow 3 included, since a shim DMA drains when the host
// says so and this pass cannot see the order the host waits in. Flows 4 and 5
// end at a DMA here, which drains them whatever the arbiter is doing.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %s2 = aie.tile(2, 0)
    %s3 = aie.tile(3, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // Core-bound, emitted here: each stalls on the core that drains it.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%c4, DMA : 0> }

    // Shim-bound, emitted here. Stalls too: the host drains it.
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%s1, DMA : 0> }

    // End at this memtile's DMA, so neither holds a grant waiting on anyone.
    aie.packet_flow(4) { aie.packet_source<%s2, DMA : 0>  aie.packet_dest<%m, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%s3, DMA : 0>  aie.packet_dest<%m, DMA : 1> }

    // Passes through (0,1) on its way south. Placed last, so it has to share.
    aie.packet_flow(6) { aie.packet_source<%c5, DMA : 0>  aie.packet_dest<%s0, DMA : 0> }
  }
}

// Flow 6 skips the shim-bound flow on arbiter 3 and joins flow 4 on arbiter 4.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SHIMBOUND:.*]] = aie.amsel<3> (0)
// CHECK:         %[[ENDSHERE:.*]] = aie.amsel<4> (0)
// CHECK:         %[[SHARED:.*]] = aie.amsel<4> (1)
// CHECK:         aie.masterset(DMA : 0, %[[ENDSHERE]])
// CHECK:         aie.masterset(South : 0, %[[SHARED]])
// CHECK:         aie.masterset(South : 2, %[[SHIMBOUND]])
// CHECK:         aie.packet_rules(North : 0) {
// CHECK-NEXT:      aie.rule(31, 6, %[[SHARED]])
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 4, %[[ENDSHERE]])
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(31, 3, %[[SHIMBOUND]])
