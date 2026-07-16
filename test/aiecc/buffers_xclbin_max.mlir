//===- buffers_xclbin_max.mlir ----------------------------------*- MLIR -*-===//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A runtime_sequence with 17 host buffers exceeds the maximum supported and
// verified count (16). aiecc must reject it at compile time rather than emit an
// xclbin whose extra buffers would not be DMA'd correctly.

// RUN: not %python aiecc.py -n --no-link --aie-generate-xclbin %s 2>&1 | FileCheck %s

// CHECK: error: device 'main' has 17 host buffer arguments
// CHECK-SAME: exceeds the maximum supported and verified count of 16

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %arg2: memref<8xi32>, %arg3: memref<8xi32>, %arg4: memref<8xi32>, %arg5: memref<8xi32>, %arg6: memref<8xi32>, %arg7: memref<8xi32>, %arg8: memref<8xi32>, %arg9: memref<8xi32>, %arg10: memref<8xi32>, %arg11: memref<8xi32>, %arg12: memref<8xi32>, %arg13: memref<8xi32>, %arg14: memref<8xi32>, %arg15: memref<8xi32>, %arg16: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<8xi32>
      // operands: column, row, direction, channel, column_num, row_num
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 0 : i32
      %cst_npu_3 = arith.constant 0 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
    }
  }
}
