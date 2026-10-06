//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static oracle for rolled_loop_chain.mlir (not a standalone test; lit ignores
// Inputs/): the rolled 2-BD chain ping-pong hand-unrolled to n = 2. The static
// allocator pins the same ids a fresh pool pops, so both program the same BD
// registers, next_bd fields included.
//
//===----------------------------------------------------------------------===//

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @static2(%arg0: memref<1024xi32>) {
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.next_bd ^bd1
    ^bd1:
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 512 len = 512 sizes = [1, 2, 8, 32] strides = [4096, 256, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_free_task(%init)
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
  }
}
