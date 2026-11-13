//===- dma_channel_valid.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The last channel of each op checked by index is accepted, and a PLIO
// allocation on a shim PL tile names a stream port, not a DMA channel.

// RUN: aie-opt %s | FileCheck %s

// CHECK: aie.dma(S2MM, 5)
// CHECK: aie.shim_dma_allocation @in(%{{.*}}, MM2S, 1)
// CHECK: aie.shim_dma_allocation @plio(%{{.*}}, MM2S, 0) {plio = true}
// CHECK: aiex.npu.push_queue(2, 0, S2MM : 1)
// CHECK: aiex.npu.push_queue(2, 1, S2MM : 5)
// CHECK: aiex.dma_start_bd_chain @c(%{{.*}}) : (memref<16xi32>) on (%{{.*}}, MM2S, 1)
module {
  aie.device(xcve2802) {
    %pl = aie.tile(0, 0)
    %shim = aie.tile(2, 0)
    %mem = aie.tile(2, 1)
    %b = aie.buffer(%mem) : memref<16xi32>
    %l0 = aie.lock(%mem, 0) {init = 1 : i32}
    %l1 = aie.lock(%mem, 1) {init = 0 : i32}
    aie.memtile_dma(%mem) {
      %0 = aie.dma(S2MM, 5) [{
        %c1 = arith.constant 1 : i32
        aie.use_lock(%l0, AcquireGreaterEqual, %c1)
        aie.dma_bd(%b : memref<16xi32>)
        aie.use_lock(%l1, Release, %c1)
      }]
      aie.end
    }
    aie.shim_dma_allocation @in(%shim, MM2S, 1)
    aie.shim_dma_allocation @plio(%pl, MM2S, 0) {plio = true}
    aie.bd_chain @c(%arg0: memref<16xi32>) {
      aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<16xi32>) {
      %c0 = arith.constant 0 : i32
      %c24 = arith.constant 24 : i32
      aiex.npu.push_queue(2, 0, S2MM : 1) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
      aiex.npu.push_queue(2, 1, S2MM : 5) bd_id %c24 repeat %c0 {issue_token = false} : i32, i32
      %t1 = aiex.dma_start_bd_chain @c(%arg0) : (memref<16xi32>) on (%shim, MM2S, 1)
    }
  }
}
