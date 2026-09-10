//===- arbiter_deadlock_avoidance.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) is a relay: flow 6 arrives at DMA : 1, flow 0 leaves from
// DMA : 0. The other flows exhaust the arbiters, so the amsel scan wraps onto
// one already in use -- but must not pick flow 0's, which would serialize the
// relay against its own input and deadlock.

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

// Seven groups, six arbiters: the seventh shares, but not with flow 0. It also
// skips arbiters 1 and 2 (flows 2 and 3 out of shim (0,0)), which close a
// longer cycle back via (0,2) and flow 1. Arbiter 3 carries flow 5 from shim
// (1,0), which nothing here reaches, and is the first free of any coupling.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[FEED:.*]] = aie.amsel<0> (0)
// CHECK:         %[[RELAY:.*]] = aie.amsel<3> (1)
// CHECK:         aie.masterset(DMA : 1, %[[RELAY]])
// CHECK:         aie.packet_rules(North : 3) {
// CHECK-NEXT:      aie.rule(31, 6, %[[RELAY]])
// CHECK:         aie.packet_rules(DMA : 0) {
// CHECK-NEXT:      aie.rule(31, 0, %[[FEED]])
