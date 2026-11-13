//===- routing_overuse_stalls.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: asserts
// RUN: aie-opt --aie-create-pathfinder-flows -debug-only=aie-pathfinder %s 2>&1 >/dev/null | FileCheck %s

// An attempt whose routing never fits the fabric gives up once its fewest
// overused channels have not dropped for maxOverusedStall iterations, rather
// than run to the iteration limit; a later, relaxed attempt routes the design.
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)
// CHECK: giving up after {{[0-9]+}} iterations (the overuse stalled)
// CHECK-NOT: giving up after {{[0-9]+}} iterations (the iteration limit)

module {
  aie.device(npu1_3col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %t_1_0 = aie.tile(1, 0)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_3 = aie.tile(1, 3)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %t_2_0 = aie.tile(2, 0)
    %t_2_1 = aie.tile(2, 1)
    %t_2_3 = aie.tile(2, 3)
    %t_2_4 = aie.tile(2, 4)
    %t_2_5 = aie.tile(2, 5)
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(MM2S, 2, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %dma_0_3 = aie.mem(%t_0_3) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 1, ^p1b0, ^end)
    ^p1b0:
      aie.next_bd ^p1b0
    ^end:
      aie.end
    }
    %dma_1_1 = aie.memtile_dma(%t_1_1) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^p1)
    ^p0b0:
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 0, ^p1b0, ^p2)
    ^p1b0:
      aie.next_bd ^p1b0
    ^p2:
      %d2 = aie.dma_start(MM2S, 2, ^p2b0, ^end)
    ^p2b0:
      aie.next_bd ^p2b1
    ^p2b1:
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %dma_1_2 = aie.mem(%t_1_2) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %dma_1_5 = aie.mem(%t_1_5) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %dma_2_1 = aie.memtile_dma(%t_2_1) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^p1)
    ^p0b0:
      aie.next_bd ^p0b0
    ^p1:
      %d1 = aie.dma_start(MM2S, 3, ^p1b0, ^end, repeat_count = 1)
    ^p1b0:
      aie.next_bd ^p1b1
    ^p1b1:
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %dma_2_3 = aie.mem(%t_2_3) {
      aie.end
    }
    %dma_2_4 = aie.mem(%t_2_4) {
      aie.end
    }
    %dma_2_5 = aie.mem(%t_2_5) {
      %d0 = aie.dma_start(MM2S, 1, ^p0b0, ^end)
    ^p0b0:
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_0_3 = aie.core(%t_0_3) {
      aie.end
    }
    %core_1_2 = aie.core(%t_1_2) {
      aie.end
    }
    %core_1_5 = aie.core(%t_1_5) {
      aie.end
    }
    aie.flow(%t_0_0, DMA : 1, %t_1_3, DMA : 0)
    aie.flow(%t_0_1, DMA : 4, %t_0_3, Core : 0)
    aie.flow(%t_2_0, DMA : 1, %t_2_5, Core : 0)
    aie.flow(%t_2_1, DMA : 0, %t_1_3, Core : 0)
    aie.flow(%t_1_1, DMA : 0, %t_1_4, DMA : 1)
    aie.flow(%t_1_1, DMA : 5, %t_1_4, DMA : 0)
    aie.flow(%t_0_1, DMA : 5, %t_0_4, DMA : 1)
    aie.flow(%t_1_2, DMA : 1, %t_2_3, DMA : 0)
    aie.flow(%t_0_0, DMA : 0, %t_0_5, DMA : 0)
    aie.flow(%t_2_1, DMA : 2, %t_2_3, DMA : 1)
    aie.flow(%t_0_1, DMA : 0, %t_2_5, DMA : 1)
    aie.flow(%t_1_1, DMA : 1, %t_2_3, Core : 0)
    aie.flow(%t_1_1, DMA : 3, %t_2_4, Core : 0)
    aie.flow(%t_0_2, DMA : 1, %t_0_4, DMA : 0)
    aie.flow(%t_0_1, DMA : 1, %t_1_5, DMA : 0)
    aie.flow(%t_1_0, DMA : 0, %t_2_4, DMA : 0)
    aie.packet_flow(23) { aie.packet_source<%t_1_1, DMA : 2> aie.packet_dest<%t_0_5, Core : 0> aie.packet_dest<%t_1_0, DMA : 1> } {keep_pkt_header = true}
    aie.packet_flow(26) { aie.packet_source<%t_1_2, Core : 0> aie.packet_source<%t_2_4, Core : 0> aie.packet_dest<%t_0_3, DMA : 1> aie.packet_dest<%t_1_1, DMA : 1> aie.packet_dest<%t_2_0, DMA : 0> }
    aie.packet_flow(15) { aie.packet_source<%t_1_2, Core : 0> aie.packet_dest<%t_0_5, DMA : 1> aie.packet_dest<%t_1_5, DMA : 1> } {keep_pkt_header = true}
    aie.runtime_sequence @seq0(%a0: memref<64xi32>, %a1: memref<64xi32>) {
    }
  }
}
