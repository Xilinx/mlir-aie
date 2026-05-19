//===- buffers_xclbin_few.mlir ----------------------------------*- MLIR -*-===//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %python aiecc.py -n --no-compile --no-link --aie-generate-xclbin %s
// RUN: FileCheck %s --input-file=buffers_xclbin_few.mlir.prj/kernels_main.json

// The host BO count is the runtime_sequence argument count, floored at 5 to
// satisfy the NPU firmware command-chain (xrt::runlist) ABI. A 2-argument
// sequence therefore emits bo0..bo4 -- the two real arguments plus padding up
// to the floor -- and no more.

// CHECK: "name": "bo0"
// CHECK: "offset": "0x14"
// CHECK: "name": "bo1"
// CHECK: "offset": "0x1C"
// CHECK: "name": "bo2"
// CHECK: "name": "bo3"
// CHECK: "name": "bo4"
// CHECK-NOT: "name": "bo5"

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @out0 (%tile_0_0, MM2S, 0)
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, metadata = @out0} : memref<1024xi32>
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
