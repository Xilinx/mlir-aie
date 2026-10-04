//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// Runtime sizes and strides are i64, and every guard compares them unsigned in
// i64 before they are narrowed to the i32 BD fields, so a value like 2^32 + 1
// cannot truncate to 1 and pass the field guard behind it.

// CHECK-LABEL: @memtile_wide
// CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be in [1:1023]"
// CHECK: cf.assert %{{.*}}, "a runtime DMA d1 size must be in [1:1023]"
// CHECK: cf.assert %{{.*}}, "a runtime DMA d2 stride must be in [1:131072] when its size > 1"
// CHECK: cf.assert %{{.*}}, "a runtime DMA transfer exceeds the 131071-granule BD buffer_length"
// CHECK: cf.assert %{{.*}}, "a runtime DMA length must equal the d0*d1*d2 extent of its dimensions"
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: aiex.npu.blockwrite_values
module @memtile_wide {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<4096xi32>
    aie.runtime_sequence(%n: i64, %stride: i64, %d0: i64) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 1024 sizes = [1, 8, %n, %d0] strides = [0, %stride, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
