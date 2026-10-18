//===- packet_sources_learned_apart.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s

// Draining flow 9's receiver at (0,3) DMA:1 can wait on the circuit from (0,4)
// and on (0,4)'s core, which waits on (0,1) DMA:3. Sharing an arbiter at (0,1)
// with (0,1) DMA:3's flows, flow 9 can hold it and deadlock them. The first
// routing shares one, so the router learns to route the two sources apart.
// Reduced from router_properties.py npu2 seed 260.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK-DAG:     %[[A0:.*]] = aie.amsel<0> (0)
// CHECK-DAG:     %[[A1:.*]] = aie.amsel<1> (0)
// CHECK:         aie.packet_rules(DMA : 5) {
// CHECK-NEXT:      aie.rule(31, 9, %[[A1]])
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-NEXT:      aie.rule(10, 2, %[[A0]])

module {
  aie.device(npu2_1col) {
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<32xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(MM2S, 3, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.flow(%t_0_4, DMA : 0, %t_0_3, DMA : 0)
    aie.packet_flow(22) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_4, Core : 0> }
    aie.packet_flow(3) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_2, DMA : 0> }
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 5> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_3, DMA : 1> }
  }
}
