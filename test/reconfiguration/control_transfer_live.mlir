// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: aie-opt %s --split-input-file -aie-substitute-shim-dma-allocations -aie-assign-runtime-sequence-bd-ids -verify-diagnostics

// A configuration's control packets start on shim tile (0, 0) while a data
// transfer there is still running: no reset separates them.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @live(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      // expected-note @+1 {{the transfer starts here}}
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer}
      // expected-warning @+1 {{starts control packets on tile (0,0) while a DMA transfer there is neither awaited nor freed}}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// A second control packet task beside the same transfer warns no more.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @twice(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      // expected-note @+1 {{the transfer starts here}}
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer}
      // expected-warning @+1 {{starts control packets on tile (0,0)}}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      %u = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer}
      aiex.dma_start_task(%u)
      aiex.dma_await_task(%u)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}

// -----

// Awaited first, the transfer is done.
module {
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @synced(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
    }
  }
}

// -----

// A load_pdi between them resets the array.
module {
  aie.device(npu2_1col) @other {
    %t00 = aie.tile(0, 0)
  }
  aie.device(npu2_1col) {
    %t00 = aie.tile(0, 0)
    aie.shim_dma_allocation @ctrl(%t00, MM2S, 0)
    aie.shim_dma_allocation @data(%t00, MM2S, 1)
    aie.runtime_sequence @reset(%buf : memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c1 = arith.constant 1 : i32
      aiex.npu.dma_memcpy_nd(%buf[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {metadata = @data, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.load_pdi {device_ref = @other}
      %t = aiex.dma_configure_task_for @ctrl {
        aie.dma_bd(%buf : memref<16xi32> offset = 0 len = 16)
        aie.end
      } {issue_token = true, aiex.ctrl_pkt_transfer}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.npu.sync(%c0, %c0, %c1, %c1, %c1, %c1) : i32, i32, i32, i32, i32, i32
    }
  }
}
