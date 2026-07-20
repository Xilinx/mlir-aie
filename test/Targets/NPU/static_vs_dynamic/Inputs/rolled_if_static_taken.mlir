//===----------------------------------------------------------------------===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Static oracle for the TAKEN branch of the rolled dynamic scf.if
// (rolled_if.mlir). One configure through the static allocator (bd id 0),
// matching the pool's pop order when the branch executes. Companion input under
// Inputs/ so lit ignores it.
//
//===----------------------------------------------------------------------===//

aie.device(npu1) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @static_taken(%arg0: memref<1024xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<1024xi32> offset = 0 len = 1024 sizes = [1, 4, 8, 32] strides = [4096, 512, 32, 1])
      aie.end
    } {issue_token = true}
    aiex.dma_start_task(%t)
    aiex.dma_await_task(%t)
    aiex.dma_free_task(%t)
  }
}
