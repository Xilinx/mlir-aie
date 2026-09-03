//===- dma_task_out_of_order_id_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A BD nested in a runtime-sequence DMA task is skipped by DMABDOp::verify, so
// DMAConfigureTaskOp::verify validates its out_of_order_id.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{out_of_order_id must be in [0, 63]}}
        aie.dma_bd(%arg0 : memref<4xi16> offset = 0 len = 4) {bd_id = 3 : i32, out_of_order_id = 100 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{out_of_order_id requires a packet-enabled BD}}
        aie.dma_bd(%arg0 : memref<4xi16> offset = 0 len = 4) {bd_id = 3 : i32, out_of_order_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}

// -----

// out_of_order is only valid on an S2MM channel.
module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // expected-error@+1 {{out_of_order is only valid on an S2MM channel}}
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// out_of_order and issue_token are mutually exclusive.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    aie.runtime_sequence(%arg0: memref<4xi32>) {
      // expected-error@+2 {{out_of_order channel cannot issue a completion token}}
      // expected-note@+1 {{set issue_token = false on this out-of-order task}}
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.end
      } {issue_token = true, out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}

// -----

// out-of-order S2MM prohibits an inter-BD lock dependency.
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %lockA = aie.lock(%tile_0_1, 0) { init = 0 : i32 }
    aie.runtime_sequence(%arg0: memref<4xi32>, %arg1: memref<8xi32>) {
      %t = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        %c1 = arith.constant 1 : i32
        aie.use_lock(%lockA, AcquireGreaterEqual, %c1)
        aie.dma_bd_packet(0, 0)
        // expected-error@+1 {{out-of-order S2MM prohibits inter-BD lock dependencies; can deadlock}}
        aie.dma_bd(%arg0 : memref<4xi32> offset = 0 len = 4) {bd_id = 0 : i32}
        aie.next_bd ^bd1
      ^bd1:
        %c2 = arith.constant 1 : i32
        aie.dma_bd_packet(0, 0)
        aie.dma_bd(%arg1 : memref<8xi32> offset = 4 len = 4) {bd_id = 1 : i32}
        aie.use_lock(%lockA, Release, %c2)
        aie.end
      } {out_of_order}
      aiex.dma_start_task(%t)
    }
  }
}
