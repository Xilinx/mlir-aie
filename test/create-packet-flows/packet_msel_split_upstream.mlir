//===- packet_msel_split_upstream.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Five ids from the shim reach five different sets of memtile (0,1) S2MM
// channels 0-2, and existing connections leave the memtile no way out. Entering
// on one port, the ids would need all three master ports on one arbiter with
// five msels, one more than an arbiter has. The shim sends the ids bound for
// S2MM 2 up a second link instead, so that channel takes an arbiter of its own
// and the other two need three msels.

// CHECK-LABEL: aie.switchbox(%shim_noc_tile_0_0)
// CHECK:         %[[BOTH:.*]] = aie.amsel<0> (0)
// CHECK:         %[[DMA01:.*]] = aie.amsel<0> (1)
// CHECK:         %[[DMA2:.*]] = aie.amsel<0> (2)
// CHECK:         aie.masterset(North : 0, %[[BOTH]], %[[DMA2]])
// CHECK:         aie.masterset(North : 5, %[[BOTH]], %[[DMA01]])
// CHECK:         aie.packet_rules(South : 7) {
// CHECK-NEXT:      aie.rule(31, 5, %[[DMA2]])
// CHECK:         aie.packet_rules(South : 3) {
// CHECK-NEXT:      aie.rule(26, 0, %[[BOTH]])
// CHECK-NEXT:      aie.rule(31, 2, %[[BOTH]])
// CHECK-NEXT:      aie.rule(31, 3, %[[DMA01]])

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         %[[D0:.*]] = aie.amsel<0> (0)
// CHECK:         %[[D2:.*]] = aie.amsel<1> (0)
// CHECK:         %[[D1:.*]] = aie.amsel<0> (1)
// CHECK:         %[[D01:.*]] = aie.amsel<0> (2)
// CHECK:         aie.masterset(DMA : 0, %[[D0]], %[[D01]])
// CHECK:         aie.masterset(DMA : 1, %[[D1]], %[[D01]])
// CHECK:         aie.masterset(DMA : 2, %[[D2]])
// CHECK:         aie.packet_rules(South : 0) {
// CHECK-NEXT:      aie.rule(24, 0, %[[D2]])
// CHECK:         aie.packet_rules(South : 5) {
// CHECK-NEXT:      aie.rule(31, 3, %[[D01]])
// CHECK-NEXT:      aie.rule(31, 4, %[[D01]])
// CHECK-NEXT:      aie.rule(31, 2, %[[D1]])
// CHECK-NEXT:      aie.rule(31, 1, %[[D0]])

module {
  aie.device(npu1_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    // Every North and South master port of the memtile taken.
    %sb01 = aie.switchbox(%t01) {
      aie.connect<DMA : 0, North : 0>
      aie.connect<DMA : 1, North : 1>
      aie.connect<DMA : 2, North : 2>
      aie.connect<DMA : 3, North : 3>
      aie.connect<DMA : 4, North : 4>
      aie.connect<DMA : 5, North : 5>
      aie.connect<North : 0, South : 0>
      aie.connect<North : 1, South : 1>
      aie.connect<North : 2, South : 2>
      aie.connect<North : 3, South : 3>
    }
    aie.packet_flow(1) { aie.packet_source<%t00, DMA : 0> aie.packet_dest<%t01, DMA : 0> aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(2) { aie.packet_source<%t00, DMA : 0> aie.packet_dest<%t01, DMA : 1> aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(3) { aie.packet_source<%t00, DMA : 0> aie.packet_dest<%t01, DMA : 0> aie.packet_dest<%t01, DMA : 1> }
    aie.packet_flow(4) { aie.packet_source<%t00, DMA : 0> aie.packet_dest<%t01, DMA : 0> aie.packet_dest<%t01, DMA : 1> aie.packet_dest<%t01, DMA : 2> }
    aie.packet_flow(5) { aie.packet_source<%t00, DMA : 1> aie.packet_dest<%t01, DMA : 2> }
    %d0 = aie.buffer(%t01) : memref<16xi32>
    %d1 = aie.buffer(%t01) : memref<16xi32>
    %d2 = aie.buffer(%t01) : memref<16xi32>
    aie.memtile_dma(%t01) {
      %0 = aie.dma_start(S2MM, 0, ^a, ^n1)
    ^a:
      aie.dma_bd(%d0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^a
    ^n1:
      %1 = aie.dma_start(S2MM, 1, ^b, ^n2)
    ^b:
      aie.dma_bd(%d1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^b
    ^n2:
      %2 = aie.dma_start(S2MM, 2, ^c, ^end)
    ^c:
      aie.dma_bd(%d2 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^c
    ^end:
      aie.end
    }
  }
}
