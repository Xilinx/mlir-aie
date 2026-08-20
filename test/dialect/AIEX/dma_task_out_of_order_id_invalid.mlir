//===- dma_task_out_of_order_id_invalid.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A BD nested in a runtime-sequence DMA task is skipped by DMABDOp::verify, so
// DMAConfigureTaskOp::verify validates its out_of_order_id: an out-of-range id
// would silently mask to a wrong merge slot, and the id only rides a packet
// header, so the BD must be packet-enabled.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{out_of_order_id must be in [0, 63]}}
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8) {bd_id = 3 : i32, out_of_order_id = 100 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = 1>}
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
    aie.runtime_sequence(%arg0: memref<8xi16>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        // expected-error@+1 {{out_of_order_id requires a packet-enabled BD}}
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8) {bd_id = 3 : i32, out_of_order_id = 5 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
    }
  }
}
