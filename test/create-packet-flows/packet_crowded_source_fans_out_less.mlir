//===- packet_crowded_source_fans_out_less.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s

// Spread over North channels of their own, the flows from (0, 1) DMA:1 need
// more packet rules there than its slave port holds, and nothing on (0, 1)
// takes packets apart. The router then leaves (0, 1) by fewer channels, so
// fewer sets of master ports, and splits the flows apart further up.

// CHECK:     aie.switchbox(%mem_tile_0_1) {
// CHECK-NOT: aie.masterset(North : {{[2-5]}}
// CHECK:     aie.switchbox(%tile_0_2) {

module {
  aie.device(npu1_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %buf_0_1_0 = aie.buffer(%t_0_1) : memref<16xi32>
    %buf_0_1_1 = aie.buffer(%t_0_1) : memref<16xi32>
    %mem_0_1 = aie.memtile_dma(%t_0_1) {
      aie.next_bd ^s0
    ^s0:
      %d0 = aie.dma_start(MM2S, 0, ^b0, ^s1)
    ^b0:
      aie.dma_bd(%buf_0_1_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^e
    ^s1:
      %d1 = aie.dma_start(MM2S, 1, ^b1, ^e)
    ^b1:
      aie.dma_bd(%buf_0_1_1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^e
    ^e:
      aie.end
    }
    %buf_0_2_0 = aie.buffer(%t_0_2) : memref<16xi32>
    %buf_0_2_1 = aie.buffer(%t_0_2) : memref<16xi32>
    %mem_0_2 = aie.mem(%t_0_2) {
      aie.next_bd ^s0
    ^s0:
      %d0 = aie.dma_start(S2MM, 0, ^b0, ^s1)
    ^b0:
      aie.dma_bd(%buf_0_2_0 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^b0
    ^s1:
      %d1 = aie.dma_start(S2MM, 1, ^b1, ^e)
    ^b1:
      aie.dma_bd(%buf_0_2_1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^b1
    ^e:
      aie.end
    }
    %buf_0_3_1 = aie.buffer(%t_0_3) : memref<16xi32>
    %mem_0_3 = aie.mem(%t_0_3) {
      aie.next_bd ^s0
    ^s0:
      %d0 = aie.dma_start(S2MM, 1, ^b0, ^e)
    ^b0:
      aie.dma_bd(%buf_0_3_1 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^b0
    ^e:
      aie.end
    }
    aie.packet_flow(1) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(3) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(12) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(20) { aie.packet_source<%t_0_1, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(0) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(5) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(23) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_3, DMA : 1> }
    aie.packet_flow(28) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_dest<%t_0_2, DMA : 1> }
  }
}
