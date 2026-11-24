//===- bad_dma_channel_runtime.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A push to a task queue and a started BD chain name one of the tile's DMA
// channels, and a shim PL tile has no DMA. Without the check, a push to shim
// S2MM 2 is lowered to a write of the MM2S 0 task queue.

// RUN: aie-opt --split-input-file --verify-diagnostics %s

aie.device(npu1) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op S2MM channel 2 is out of range: the tile has 2 S2MM channels}}
    aiex.npu.push_queue(0, 0, S2MM : 2) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu1) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op S2MM channel 6 is out of range: the tile has 6 S2MM channels}}
    aiex.npu.push_queue(0, 1, S2MM : 6) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

// xcve2802 column 0 is a shim PL tile.
aie.device(xcve2802) {
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    %c0 = arith.constant 0 : i32
    // expected-error@+1 {{'aiex.npu.push_queue' op tile (0, 0) is a shim PL tile and has no DMA}}
    aiex.npu.push_queue(0, 0, S2MM : 0) bd_id %c0 repeat %c0 {issue_token = false} : i32, i32
  }
}

// -----

aie.device(npu1) {
  %t = aie.tile(0, 0)
  aie.bd_chain @c(%arg0: memref<16xi32>) {
    aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16)
    aie.end
  }
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    // expected-error@+1 {{'aiex.dma_start_bd_chain' op MM2S channel 3 is out of range: the tile has 2 MM2S channels}}
    %t1 = aiex.dma_start_bd_chain @c(%arg0) : (memref<16xi32>) on (%t, MM2S, 3)
  }
}

// -----

aie.device(xcve2802) {
  %t = aie.tile(0, 0)
  aie.runtime_sequence(%arg0: memref<16xi32>) {
    // expected-error@+1 {{'aiex.dma_configure_task' op tile (0, 0) is a shim PL tile and has no DMA}}
    %t1 = aiex.dma_configure_task(%t, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<16xi32> offset = 0 len = 16) {bd_id = 0 : i32}
      aie.end
    }
  }
}
