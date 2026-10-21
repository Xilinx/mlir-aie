//===- packet_cluster_member_leads.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-create-pathfinder-flows %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty
// RUN: aie-opt --aie-create-pathfinder-flows="circuit-switch-hops=false" %s 2>&1 >/dev/null | FileCheck %s --check-prefix=NOERR --allow-empty

// NOERR-NOT: {{warning|error}}

// Id 6 from (0, 1) DMA:3 and id 15 from (1, 2) Core:0 and (0, 4) DMA:0 reach
// (0, 0) DMA:1 and (1, 2) DMA:0 all, so the three trees have to meet. With
// (0, 1) DMA:3's tree routed first they find no switchbox to meet at that
// keeps them out of a hold cycle; with (0, 4) DMA:0's first they do.
// Reduced from router_properties.py npu1 seed 207460169.

module {
  aie.device(npu1_2col) {
    %t_0_0 = aie.tile(0, 0)
    %t_0_1 = aie.tile(0, 1)
    %t_0_2 = aie.tile(0, 2)
    %t_0_4 = aie.tile(0, 4)
    %t_1_1 = aie.tile(1, 1)
    %t_1_2 = aie.tile(1, 2)
    %t_1_4 = aie.tile(1, 4)
    %t_1_5 = aie.tile(1, 5)
    %l_0_2_0 = aie.lock(%t_0_2, 0) {init = 2 : i32, sym_name = "l_0_2_0"}
    %l_0_2_1 = aie.lock(%t_0_2, 1) {init = 0 : i32, sym_name = "l_0_2_1"}
    %l_1_2_0 = aie.lock(%t_1_2, 0) {init = 2 : i32, sym_name = "l_1_2_0"}
    %l_1_2_1 = aie.lock(%t_1_2, 1) {init = 0 : i32, sym_name = "l_1_2_1"}
    %l_1_5_0 = aie.lock(%t_1_5, 0) {init = 1 : i32, sym_name = "l_1_5_0"}
    %l_1_5_1 = aie.lock(%t_1_5, 1) {init = 0 : i32, sym_name = "l_1_5_1"}
    %b_0_1_0 = aie.buffer(%t_0_1) {sym_name = "b_0_1_0"} : memref<16xi32>
    %b_0_1_1 = aie.buffer(%t_0_1) {sym_name = "b_0_1_1"} : memref<32xi32>
    %dma_0_1 = aie.memtile_dma(%t_0_1) {
      %d0 = aie.dma_start(MM2S, 3, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_1_0 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_0_1_1 : memref<32xi32> offset = 0 len = 32) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 6>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_2_2 = aie.buffer(%t_0_2) {sym_name = "b_0_2_2"} : memref<8xi32>
    %dma_0_2 = aie.mem(%t_0_2) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 1, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_0_2_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_0_2_2 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_0_2_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_0_4_3 = aie.buffer(%t_0_4) {sym_name = "b_0_4_3"} : memref<16xi32>
    %b_0_4_4 = aie.buffer(%t_0_4) {sym_name = "b_0_4_4"} : memref<8xi32>
    %dma_0_4 = aie.mem(%t_0_4) {
      %d0 = aie.dma_start(MM2S, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_0_4_3 : memref<16xi32> offset = 0 len = 16) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
      aie.next_bd ^p0b1
    ^p0b1:
      aie.dma_bd(%b_0_4_4 : memref<8xi32> offset = 0 len = 8) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_1_1_5 = aie.buffer(%t_1_1) {sym_name = "b_1_1_5"} : memref<32xi32>
    %dma_1_1 = aie.memtile_dma(%t_1_1) {
      %d0 = aie.dma_start(S2MM, 4, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_1_1_5 : memref<32xi32> offset = 0 len = 32)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_1_2_6 = aie.buffer(%t_1_2) {sym_name = "b_1_2_6"} : memref<8xi32>
    %dma_1_2 = aie.mem(%t_1_2) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_1_2_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_2_6 : memref<8xi32> offset = 0 len = 8)
      aie.use_lock(%l_1_2_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_1_4_7 = aie.buffer(%t_1_4) {sym_name = "b_1_4_7"} : memref<16xi32>
    %dma_1_4 = aie.mem(%t_1_4) {
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.dma_bd(%b_1_4_7 : memref<16xi32> offset = 0 len = 16)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %b_1_5_8 = aie.buffer(%t_1_5) {sym_name = "b_1_5_8"} : memref<16xi32>
    %dma_1_5 = aie.mem(%t_1_5) {
      %c1 = arith.constant 1 : i32
      %d0 = aie.dma_start(S2MM, 0, ^p0b0, ^end)
    ^p0b0:
      aie.use_lock(%l_1_5_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%b_1_5_8 : memref<16xi32> offset = 0 len = 16)
      aie.use_lock(%l_1_5_1, Release, %c1)
      aie.next_bd ^p0b0
    ^end:
      aie.end
    }
    %core_0_2 = aie.core(%t_0_2) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_0_2_0, Release, %c1)
      aie.use_lock(%l_0_2_1, AcquireGreaterEqual, %c1)
      aie.end
    }
    %core_1_2 = aie.core(%t_1_2) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_1_2_0, Release, %c1)
      aie.use_lock(%l_1_2_1, AcquireGreaterEqual, %c1)
      aie.end
    }
    %core_1_5 = aie.core(%t_1_5) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%l_1_5_0, Release, %c1)
      aie.use_lock(%l_1_5_1, AcquireGreaterEqual, %c1)
      aie.end
    }
    aie.packet_flow(6) { aie.packet_source<%t_0_1, DMA : 3> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_0_2, DMA : 1> aie.packet_dest<%t_1_2, DMA : 0> }
    aie.packet_flow(15) { aie.packet_source<%t_1_2, Core : 0> aie.packet_source<%t_0_4, DMA : 0> aie.packet_dest<%t_0_0, DMA : 1> aie.packet_dest<%t_1_2, DMA : 0> }
    aie.shim_dma_allocation @out0_1(%t_0_0, S2MM, 1)
    aie.runtime_sequence @seq0(%a0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%a0[0, 0, 0, 0][1, 1, 1, 64][64, 0, 0, 1]) { metadata = @out0_1, id = 0 : i64, issue_token = true } : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out0_1}
    }
  }
}
