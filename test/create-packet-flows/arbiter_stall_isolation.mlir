//===- arbiter_stall_isolation.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Memtile (0,1) emits flows 0..3 to cores and flows 4..5 to the shim, filling
// all six arbiters. Flow 6 only passes through, so it has to share.
//
// It must not share with a core-bound flow: that one stalls whenever its
// consumer core falls behind, and the arbiter holds the grant until tlast, so
// flow 6 would stall with it. Sharing with flow 5 is safe: that one drains to
// DDR, which no core gates.

module {
  aie.device(npu2) {
    %s0 = aie.tile(0, 0)
    %s1 = aie.tile(1, 0)
    %m  = aie.tile(0, 1)
    %c2 = aie.tile(0, 2)
    %c3 = aie.tile(0, 3)
    %c4 = aie.tile(0, 4)
    %c5 = aie.tile(0, 5)

    // Core-bound, emitted by this memtile: stalls on its consumer.
    aie.packet_flow(0) { aie.packet_source<%m, DMA : 0>  aie.packet_dest<%c2, DMA : 0> }
    aie.packet_flow(1) { aie.packet_source<%m, DMA : 1>  aie.packet_dest<%c3, DMA : 0> }
    aie.packet_flow(2) { aie.packet_source<%m, DMA : 2>  aie.packet_dest<%c4, DMA : 0> }
    aie.packet_flow(3) { aie.packet_source<%m, DMA : 3>  aie.packet_dest<%c5, DMA : 0> }

    // Shim-bound, emitted by this memtile: drains to DDR, never stalls here.
    aie.packet_flow(4) { aie.packet_source<%m, DMA : 4>  aie.packet_dest<%s0, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%m, DMA : 5>  aie.packet_dest<%s1, DMA : 0> }

    // Passes through (0,1) on its way north.
    aie.packet_flow(6) { aie.packet_source<%s0, DMA : 0>  aie.packet_dest<%c3, DMA : 1> }
  }
}

// Flow 6 wraps onto arbiter 5, past the four core-bound flows.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[SHIMBOUND:.*]] = aie.amsel<5> (0)
// CHECK:         %[[TRANSIT:.*]] = aie.amsel<5> (1)
// CHECK:         aie.packet_rules(South : 1) {
// CHECK-NEXT:      aie.rule(31, 6, %[[TRANSIT]])
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 5, %[[SHIMBOUND]])
