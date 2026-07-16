//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Hand-unrolled 2-iteration ping-pong: the static oracle for the rolled
// dynamic loop at n = 2 (rolled_loop.mlir). Two configures ping-pong, the
// allocator assigns ids 0 then 1 (lowest-free-first), matching the pool's pop
// order. Companion input under Inputs/ so lit ignores it.
//
//===----------------------------------------------------------------------===//

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @static2(%arg0: memref<1024xi32>) {
    %init = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%init)
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_free_task(%init)
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
  }
}
