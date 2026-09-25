//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu --split-input-file %s | FileCheck %s

// A runtime stride is masked into the BD's step field by the packer, so it
// needs the same bound a constant stride gets from verifyStridesWraps -- else
// an oversized value is silently truncated on hardware. The bound is per tile
// type, and the same descriptor on three tile types is what pins that wiring
// to the target model rather than to a hardcoded shim width.
//
// AIE2 step field widths: shim NOC 20 bits (1048575), mem tile 17 (131071),
// core tile 13 (8191).

// CHECK-LABEL: @shim_stride
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 1048575 : i32}
module @shim_stride {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    aie.runtime_sequence(%arg0: memref<4096xi32>, %stride: i64) {
      %t = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
          aie.dma_bd(%arg0 : memref<4096xi32> offset = 0 len = 1024 sizes = [1, 8, 16, 8] strides = [0, %stride, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// CHECK-LABEL: @memtile_stride
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 131071 : i32}
module @memtile_stride {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %buf = aie.buffer(%tile_0_1) : memref<4096xi32>
    aie.runtime_sequence(%stride: i64) {
      %t = aiex.dma_configure_task(%tile_0_1, MM2S, 0) {
          aie.dma_bd(%buf : memref<4096xi32> offset = 0 len = 1024 sizes = [1, 8, 16, 8] strides = [0, %stride, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}

// -----

// CHECK-LABEL: @coretile_stride
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 8191 : i32}
module @coretile_stride {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %buf = aie.buffer(%tile_0_2) : memref<1024xi32>
    aie.runtime_sequence(%stride: i64) {
      %t = aiex.dma_configure_task(%tile_0_2, MM2S, 0) {
          aie.dma_bd(%buf : memref<1024xi32> offset = 0 len = 512 sizes = [1, 8, 8, 8] strides = [0, %stride, 8, 1]) {bd_id = 0 : i32}
          aie.end
      }
    }
  }
}
