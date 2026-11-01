//===- arbiter_self_conflict.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Packet flow 1 has two sources and fans out to two memtile channels whose
// receives may stall each other, so the router keeps the two sources off a
// shared arbiter. A tree that meets itself at a branch point shares an arbiter
// only with itself, which is no conflict. Counting it as one pushed the tree
// onto more channels than the link has, and routing failed.

// RUN: aie-opt --aie-create-pathfinder-flows --aie-find-flows %s | FileCheck %s

// CHECK:      aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%{{.*}}shim_noc_tile_0_0, DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%{{.*}}mem_tile_0_1, DMA : 5>
// CHECK:      aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%{{.*}}shim_noc_tile_0_0, DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%{{.*}}mem_tile_0_1, DMA : 4>
// CHECK:      aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%{{.*}}tile_0_4, Core : 0>
// CHECK-NEXT:   aie.packet_dest<%{{.*}}mem_tile_0_1, DMA : 5>
// CHECK:      aie.packet_flow(1) {
// CHECK-NEXT:   aie.packet_source<%{{.*}}tile_0_4, Core : 0>
// CHECK-NEXT:   aie.packet_dest<%{{.*}}mem_tile_0_1, DMA : 4>

module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %b0 = aie.buffer(%t01) : memref<16xi32>
    %b1 = aie.buffer(%t01) : memref<32xi32>
    %dma01 = aie.memtile_dma(%t01) {
      %d0 = aie.dma_start(S2MM, 4, ^bd0, ^next)
    ^bd0:
      aie.dma_bd(%b0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^bd0
    ^next:
      %d1 = aie.dma_start(S2MM, 5, ^bd1, ^end)
    ^bd1:
      aie.dma_bd(%b1 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^bd1
    ^end:
      aie.end
    }
    aie.flow(%t03, Core : 0, %t01, DMA : 0)
    aie.packet_flow(10) {
      aie.packet_source<%t03, DMA : 1>
      aie.packet_dest<%t01, DMA : 3>
    }
    aie.packet_flow(19) {
      aie.packet_source<%t04, DMA : 0>
      aie.packet_dest<%t00, DMA : 0>
    }
    aie.packet_flow(14) {
      aie.packet_source<%t04, Core : 0>
      aie.packet_dest<%t04, DMA : 1>
    }
    aie.packet_flow(1) {
      aie.packet_source<%t04, Core : 0>
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, DMA : 4>
      aie.packet_dest<%t01, DMA : 5>
    }
  }
}
