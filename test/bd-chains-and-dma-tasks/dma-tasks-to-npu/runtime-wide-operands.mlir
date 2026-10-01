//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// RUN: aie-opt --aie-prepare-buffers --aie-assign-buffer-addresses --aie-dma-tasks-to-npu %s | FileCheck %s

// Runtime sizes and strides are i64, but the BD encoding is i32 arithmetic.
// Each operand is bounded at its original width before it is narrowed, so a
// value like 2^32 + 1 cannot truncate to 1 and pass the field guard behind it.
// A size used as is gets the i32 bound. d0's size and every stride are first
// multiplied by the element width in bits, so they are bounded to what that
// product can hold in i32: 2^31 - 1 over 32 bits is 67108863.

// CHECK-LABEL: @memtile_wide
// CHECK: aiex.npu.assert_bd_field(%arg2) {max = 67108863 : i32} : i64
// CHECK: aiex.npu.assert_bd_field(%arg0) {max = 2147483647 : i32} : i64
// CHECK: aiex.npu.assert_bd_field(%arg1) {max = 67108863 : i32} : i64
// CHECK: arith.trunci %arg2 : i64 to i32
// CHECK: arith.trunci %arg0 : i64 to i32
// CHECK: arith.trunci %arg1 : i64 to i32
// The field guards run on the narrowed values.
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 1023 : i32} : i32
// CHECK: aiex.npu.assert_bd_field(%{{.*}}) {max = 131071 : i32} : i32
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
