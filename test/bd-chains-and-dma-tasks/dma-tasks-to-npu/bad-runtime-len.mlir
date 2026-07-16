//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --verify-diagnostics --aie-dma-tasks-to-npu %s

// A runtime-valued dma_bd len would be silently baked to the buffer's element
// count by getLenInBytes() on the static NPU lowering path. Reject it with a
// clear diagnostic (mirroring the runtime-valued size/stride guard), rather
// than encode a wrong length. Genuinely runtime-valued len arrives with the
// dynamic BD-word encoder (--aie-npu-to-cpp).

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<4096xi32>, %len: i32) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          // expected-error@+1 {{runtime-valued BD len is not supported on the static NPU lowering path}}
          aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = %len sizes = [1, 8, 16, 32] strides = [4096, 512, 32, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
