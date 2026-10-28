//===- arbiter_hold_cycle_by_assumption_replan.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOWARN --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>/dev/null | FileCheck %s

// Every plan has a cycle through a receiver two trees share, but only by
// assumption, so the plan stands. Replanning for that must not put id 29 on an
// arbiter with id 12 at (0, 2): id 29 can fill (0, 0) S2MM 0, whose drain is
// assumed to wait, through (0, 0) MM2S 1, on (0, 5) S2MM 0, which id 12 feeds.

// NOWARN-NOT: {{warning|error}}

// CHECK-LABEL: aie.switchbox(%tile_0_2)
// CHECK-DAG:     %[[A:[0-9]+]] = aie.amsel<1> (0)
// CHECK-DAG:     %[[B:[0-9]+]] = aie.amsel<1> (1)
// CHECK-DAG:     %[[C:[0-9]+]] = aie.amsel<4> (0)
// CHECK-DAG:     aie.rule(31, 12, %[[A]])
// CHECK-DAG:     aie.rule(31, 12, %[[B]])
// CHECK-DAG:     aie.rule(31, 29, %[[C]])
// CHECK-LABEL: aie.switchbox(%tile_0_3)

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %b_0_5_0 = aie.buffer(%t_0_5) {sym_name = "b_0_5_0"} : memref<8xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_5_0 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.packet_flow(22) { aie.packet_source<%t_0_5, Core : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(12) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(29) { aie.packet_source<%t_0_1, DMA : 4> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_3, DMA : 0> }
    aie.packet_flow(0) { aie.packet_source<%t_0_1, DMA : 1> aie.packet_source<%t_0_0, DMA : 0> aie.packet_dest<%t_0_4, Core : 0> aie.packet_dest<%t_0_5, DMA : 0> }
    aie.packet_flow(11) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_1, DMA : 5> }
    aie.packet_flow(30) { aie.packet_source<%t_0_5, DMA : 0> aie.packet_dest<%t_0_1, DMA : 5> }
  }
}
