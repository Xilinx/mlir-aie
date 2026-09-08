//===- arbiter_deadlock_avoidance.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// An arbiter grants one slave port and holds that grant until tlast, so two
// slave ports sharing an arbiter serialize completely -- even when they leave
// the switchbox on different master ports. That is fatal when the two flows
// are in a producer/consumer relation, because the held grant is exactly what
// the other flow needs in order to release it.
//
// Here memtile (0,1) is a relay: packet flow 6 arrives at its DMA : 1 and
// packet flow 0 leaves from its DMA : 0. The remaining flows exist only to use
// up the switchbox's other arbiters, so that the allocator's scan wraps from
// msel 0 back onto an arbiter that is already occupied. It must not wrap onto
// the arbiter holding flow 0.

module {
  aie.device(npu2) {
    %t00 = aie.tile(0, 0)
    %t10 = aie.tile(1, 0)
    %t20 = aie.tile(2, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // The relay pair through memtile (0,1): flow 0 out, flow 6 in.
    aie.packet_flow(0) { aie.packet_source<%t01, DMA : 0>  aie.packet_dest<%t02, DMA : 0> }
    aie.packet_flow(6) { aie.packet_source<%t03, DMA : 1>  aie.packet_dest<%t01, DMA : 1> }

    // Filler, to exhaust the arbiters at (0,1).
    aie.packet_flow(1) { aie.packet_source<%t02, DMA : 0>  aie.packet_dest<%t00, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%t00, DMA : 0>  aie.packet_dest<%t03, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%t00, DMA : 1>  aie.packet_dest<%t04, DMA : 0> }
    aie.packet_flow(4) { aie.packet_source<%t10, DMA : 0>  aie.packet_dest<%t05, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t10, DMA : 1>  aie.packet_dest<%t01, DMA : 0> }
    aie.packet_flow(7) { aie.packet_source<%t20, DMA : 0>  aie.packet_dest<%t05, DMA : 1> }
    aie.packet_flow(8) { aie.packet_source<%t04, DMA : 1>  aie.packet_dest<%t00, DMA : 1> }
  }
}

// The switchbox is oversubscribed: seven independent groups, six arbiters. The
// seventh therefore shares, but it must pick an arbiter other than the one
// carrying flow 0. Before the fix it took arbiter 0 -- amsel<0> (1) -- and
// deadlocked.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FEED:.*]] = aie.amsel<0> (0)
// CHECK:         %[[RELAY:.*]] = aie.amsel<1> (1)
// CHECK:         aie.masterset(DMA : 1, %[[RELAY]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 6, %[[RELAY]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 0, %[[FEED]])
