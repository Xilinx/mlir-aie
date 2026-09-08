//===- arbiter_deadlock_avoidance.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) is a relay: flow 6 arrives at its DMA : 1 and flow 0 leaves
// from its DMA : 0. The other flows use up the switchbox's arbiters, so the
// amsel scan wraps from msel 0 onto an arbiter already in use. It must not
// wrap onto the arbiter carrying flow 0, which would serialize the relay
// against its own input and deadlock.

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

// Seven groups, six arbiters: the seventh shares, but not with flow 0.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FEED:.*]] = aie.amsel<0> (0)
// CHECK:         %[[RELAY:.*]] = aie.amsel<1> (1)
// CHECK:         aie.masterset(DMA : 1, %[[RELAY]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 6, %[[RELAY]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 0, %[[FEED]])
