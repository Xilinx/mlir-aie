//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-dma-tasks-to-npu %s | FileCheck %s

// A runtime INNERMOST stride on the dma_task path (int8) is accepted at compile
// time and guarded at dispatch: the DMA steps whole 4-byte granules, so a
// 1-byte element walk is realizable only when contiguous (stride == 1). Parity
// with the dma_memcpy_nd path.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: %[[UNIT:.*]] = arith.cmpi eq, %arg2, %{{.*}} : i64
// CHECK: cf.assert %[[UNIT]], "a runtime DMA d0 stride must be 1 for 1-byte elements (the DMA moves whole 4-byte granules)"
// CHECK: aiex.npu.blockwrite_values

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)

    aie.runtime_sequence(%arg0: memref<4096xi8>, %len: i32, %s: i64) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<4096xi8> offset = 0 len = %len sizes = [1, 8, 16, 4] strides = [4096, 512, 4, %s]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
