//===- packet_meeting_routes_ids_elsewhere_apart.mlir --------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Id 31 from (0, 3) Core:0 and ids 6 and 18 from (0, 5) DMA:1 reach (0, 2)
// Core:0 and (0, 4) DMA:1 both, so the trees meet. (0, 5) DMA:1 also sends
// id 19 back into itself, which conflicts with id 31, so id 19 leaves as a
// part of its own and the rest of the tree meets id 31's.
// Reduced from router_properties.py npu1 seed 0.

module {
  aie.device(npu1_1col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_2 = aie.tile(0, 2)
    %t_0_3 = aie.tile(0, 3)
    %t_0_4 = aie.tile(0, 4)
    %t_0_5 = aie.tile(0, 5)
    %l_0_3_0 = aie.lock(%t_0_3, 0) {init = 0 : i32, sym_name = "l_0_3_0"}
    %l_0_3_1 = aie.lock(%t_0_3, 1) {init = 1 : i32, sym_name = "l_0_3_1"}
    %b_0_2_0 = aie.buffer(%t_0_2) {sym_name = "b_0_2_0"} : memref<8xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_2_0 : memref<8xi32> offset = 0 len = 8)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_3_1 = aie.buffer(%t_0_3) {sym_name = "b_0_3_1"} : memref<32xi32>
    %b_0_3_2 = aie.buffer(%t_0_3) {sym_name = "b_0_3_2"} : memref<32xi32>
    %dma_0_3 = aie.mem(%t_0_3) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_3_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_3_1 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_0_3_2 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 19>}
      aie.use_lock(%l_0_3_1, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %b_0_4_3 = aie.buffer(%t_0_4) {sym_name = "b_0_4_3"} : memref<32xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_4_3 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_5_4 = aie.buffer(%t_0_5) {sym_name = "b_0_5_4"} : memref<16xi32>
    %dma_0_5 = aie.mem(%t_0_5) {
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_5_4 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_0_3 = aie.core(%t_0_3) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_3_1, AcquireGreaterEqual, %c1)
      aie.use_lock(%l_0_3_0, Release, %c1)
      aie.end
    }
    aie.flow(%t_0_2, Core : 0, %t_0_3, DMA : 0)
    aie.packet_flow(31) { aie.packet_source<%t_0_3, Core : 0> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.packet_flow(6) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_2, Core : 0> aie.packet_dest<%t_0_2, DMA : 0> aie.packet_dest<%t_0_4, DMA : 1> }
    aie.packet_flow(19) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_source<%t_0_3, DMA : 0> aie.packet_dest<%t_0_5, DMA : 1> }
    aie.packet_flow(18) { aie.packet_source<%t_0_5, DMA : 1> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_2, Core : 0> } {keep_pkt_header = false}
    aie.packet_flow(15) { aie.packet_source<%t_0_0, TileControl : 0> aie.packet_dest<%t_0_0, South : 0> } {keep_pkt_header = true, priority_route = true}
    aie.shim_dma_allocation @out0_1(%t_0_0, S2MM, 1)
    aie.runtime_sequence @seq0(%a0: memref<32xi32>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 32][32, 0, 0, 1]) { metadata = @out0_1, id = 0 : i64, issue_token = true } : memref<32xi32>
    }
  }
}
