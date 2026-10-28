//===- buffers_xclbin_scalars.mlir ------------------------------*- MLIR -*-===//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: %aiecc -n --get-xclbin %s
// RUN: FileCheck %s --input-file=buffers_xclbin_scalars.mlir.prj/kernels_main.json

// Only a runtime_sequence's memref arguments are host buffers. Its scalar
// arguments are dispatch-time values, folded into the instruction stream the
// dispatch bridge generates per call and never passed to the kernel, so 2
// buffers and 17 scalars are 2 host buffers (padded to the floor of 5), not
// 19, past the ceiling of 16.

// CHECK: "name": "bo0"
// CHECK: "name": "bo4"
// CHECK-NOT: "name": "bo5"

module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @in0 (%tile_0_0, S2MM, 0)
    aie.runtime_sequence(%arg0: memref<8xi32>, %arg1: memref<8xi32>, %s0: i32, %s1: i32, %s2: i32, %s3: i32, %s4: i32, %s5: i32, %s6: i32, %s7: i32, %s8: i32, %s9: i32, %s10: i32, %s11: i32, %s12: i32, %s13: i32, %s14: i32, %s15: i32, %s16: i32) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 0 : i64, metadata = @in0} : memref<8xi32>
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_0, %cst_npu_0, %cst_npu_0, %cst_npu_1, %cst_npu_1) : i32, i32, i32, i32, i32, i32
    }
  }
}
