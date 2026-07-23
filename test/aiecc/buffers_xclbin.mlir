//===- buffers_xclbin.mlir --------------------------------------*- MLIR -*-===//
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The host BO count is derived from the runtime_sequence argument count, so a
// 6-argument sequence emits exactly bo0..bo5. Only the generated kernels JSON
// metadata is inspected, so this runs the front-end without compiling cores.

// RUN: %aiecc -n --get-xclbin %s
// RUN: FileCheck %s --input-file=buffers_xclbin.mlir.prj/kernels_main.json

// CHECK: "name": "bo0"
// CHECK: "offset": "0x14"
// CHECK: "name": "bo1"
// CHECK: "name": "bo2"
// CHECK: "name": "bo3"
// CHECK: "name": "bo4"
// CHECK: "name": "bo5"
// CHECK: "offset": "0x3C"
// CHECK-NOT: "name": "bo6"

module {
  aie.device(npu1) {
    %02 = aie.tile(0, 2)
    %12 = aie.tile(1, 2)
    %22 = aie.tile(2, 2)
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @in1 (%tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @in2 (%tile_0_0, S2MM, 2)
    aie.shim_dma_allocation @out0 (%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @out1 (%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @out2 (%tile_0_0, MM2S, 2)
    aie.runtime_sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: memref<1024xi32>, %arg4: memref<1024xi32>, %arg5: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 1 : i64, metadata = @out0} : memref<1024xi32>
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 0 : i32
      %cst_npu_3 = arith.constant 0 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd (%arg2[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @in1} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%arg3[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 3 : i64, metadata = @out1} : memref<1024xi32>
      %cst_npu_6 = arith.constant 0 : i32
      %cst_npu_7 = arith.constant 0 : i32
      %cst_npu_8 = arith.constant 0 : i32
      %cst_npu_9 = arith.constant 1 : i32
      %cst_npu_10 = arith.constant 1 : i32
      %cst_npu_11 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_6, %cst_npu_7, %cst_npu_8, %cst_npu_9, %cst_npu_10, %cst_npu_11) : i32, i32, i32, i32, i32, i32
      aiex.npu.dma_memcpy_nd (%arg4[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 2 : i64, metadata = @in2} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd (%arg5[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0, 1]) {id = 3 : i64, metadata = @out2} : memref<1024xi32>
      %cst_npu_12 = arith.constant 2 : i32
      %cst_npu_13 = arith.constant 0 : i32
      %cst_npu_14 = arith.constant 0 : i32
      %cst_npu_15 = arith.constant 0 : i32
      %cst_npu_16 = arith.constant 1 : i32
      %cst_npu_17 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_12, %cst_npu_13, %cst_npu_14, %cst_npu_15, %cst_npu_16, %cst_npu_17) : i32, i32, i32, i32, i32, i32
    }
  }
}
