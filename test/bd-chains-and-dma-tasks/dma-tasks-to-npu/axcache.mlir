//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A configured $axcache on a task-path shim BD reaches the writebd; a memtile BD
// leaves it unset, since AXCache only exists on the AXI-MM side.

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) { address = 0xBEEF : i32 } : memref<32xi8>

    aie.runtime_sequence(%arg0: memref<8xi16>) {
      // CHECK: aiex.npu.writebd {axcache = 15 : i32, bd_id = 7 : i32
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8) {bd_id = 7 : i32, axcache = 15 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)

      // Unset on a shim BD still resolves to the target default.
      // CHECK: aiex.npu.writebd {axcache = 2 : i32, bd_id = 8 : i32
      %t2 = aiex.dma_configure_task(%tile_0_0, MM2S, 1) {
        aie.dma_bd(%arg0 : memref<8xi16> offset = 0 len = 8) {bd_id = 8 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2)

      // CHECK: aiex.npu.writebd {bd_id = 9 : i32
      %t3 = aiex.dma_configure_task(%tile_0_1, S2MM, 0) {
        aie.dma_bd(%buf : memref<32xi8> offset = 0 len = 32) {bd_id = 9 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3)
    }
  }
}
