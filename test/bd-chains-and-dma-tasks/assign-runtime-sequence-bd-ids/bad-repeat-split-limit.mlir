//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-split-long-repeats='max-pushes=2' --verify-diagnostics %s

// 601 runs need three 256-run pushes, one more than the bound allows.
aie.device(npu2) {
  %tile_0_0 = aie.tile(0, 0)
  aie.runtime_sequence @too_many(%arg0: memref<256xi32>) {
    %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
      aie.dma_bd(%arg0 : memref<256xi32> offset = 0 len = 256)
      aie.end
    } {repeat_count = 600 : i32}
    // expected-error@+1 {{repeat count 600 needs 3 queue pushes, more than max-pushes (2)}}
    aiex.dma_start_task(%t)
  }
}
