//===- arbiter_unit_outnumbers_msels.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s | FileCheck %s
// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=WARN --allow-empty

// The flows out of (0,1) first route onto masters tied to one arbiter with
// more master sets than its four msels. No arbiter plan fits that, whatever
// the hazards between the flows, so the router moves flows to other masters
// instead of blaming a hazard no plan was ever tried against.
// Reduced from router_properties.py npu2 seed 5412.

// CHECK-LABEL: aie.switchbox(%mem_tile_0_1)
// CHECK:         aie.packet_rules(DMA : 3) {
// CHECK-DAG:       aie.rule(31, 9,
// CHECK-DAG:       aie.rule(31, 7,
// CHECK-DAG:       aie.rule(31, 20,

// WARN-NOT: {{warning|error}}

module {
  aie.device(npu2_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_1_2 = aie.lock(%t_0_1, 2) {init = 1 : i32, sym_name = "l_0_1_2"}
    %l_0_1_3 = aie.lock(%t_0_1, 3) {init = 0 : i32, sym_name = "l_0_1_3"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<48xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 3, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_1_0 : memref<48xi32> offset = 0 len = 48)
      aie.use_lock(%l_0_1_3, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    aie.flow(%t_0_1, DMA : 1, %t_0_5, Core : 0)
    aie.packet_flow(6) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_dest<%t_0_0, DMA : 1> }
    aie.packet_flow(26) { aie.packet_source<%t_0_1, DMA : 2> aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_0, DMA : 1> }
    aie.packet_flow(1) { aie.packet_source<%t_0_4, DMA : 0> aie.packet_dest<%t_0_1, DMA : 3> }
    aie.packet_flow(20) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_1, DMA : 3> }
    aie.packet_flow(7) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_2, DMA : 1> }
    aie.packet_flow(9) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_0, DMA : 0> aie.packet_dest<%t_0_3, DMA : 0> }
    aie.packet_flow(10) { aie.packet_source<%t_0_0, DMA : 1> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.shim_dma_allocation @in0_1(%t_0_0, MM2S, 1, <pkt_id = 10, pkt_type = 0>)
    aie.shim_dma_allocation @out0_0(%t_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0_1(%t_0_0, S2MM, 1)
    aie.runtime_sequence @seq0(%a0: memref<16xi32>, %a1: memref<16xi32>, %a2: memref<64xi32>, %a3: memref<16xi32>, %a4: memref<16xi32>) {
      %task0 = aiex.dma_configure_task(%t_0_0, MM2S, 0) {
        aie.dma_bd(%a0 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.npu.dma_memcpy_nd(%a1[0, 0, 0, 0][1, 1, 1, 16][16, 0, 0, 1]) { metadata = @out0_0, id = 1 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%a2[0, 0, 0, 0][1, 1, 1, 64][64, 0, 0, 1]) { metadata = @out0_1, id = 2 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_memcpy_nd(%a3[0, 0, 0, 0][1, 1, 1, 16][16, 0, 0, 1], packet = <pkt_id = 10, pkt_type = 0>) { metadata = @in0_1, id = 3 : i64, issue_token = true } : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out0_1}
      %lb0 = arith.constant 0 : index
      %ub0 = arith.constant 2 : index
      %st0 = arith.constant 1 : index
      scf.for %i0 = %lb0 to %ub0 step %st0 {
        aiex.npu.dma_memcpy_nd(%a4[0, 0, 0, 0][1, 1, 1, 16][16, 0, 0, 1], packet = <pkt_id = 10, pkt_type = 0>) { metadata = @in0_1, id = 4 : i64, issue_token = true } : memref<16xi32>
      }
      aiex.npu.dma_wait {symbol = @out0_0}
      aiex.dma_start_task(%task0)
      aiex.dma_start_task(%task0)
      aiex.npu.dma_wait {symbol = @in0_1}
      aiex.dma_await_task(%task0)
    }
  }
}

