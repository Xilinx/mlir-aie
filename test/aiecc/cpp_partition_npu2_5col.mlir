//===- cpp_partition_npu2_5col.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Tests the legal start columns for a 5-column partition of the 8-column NPU2
// array. A 5-wide partition fits at every offset up to 8 - 5.

// REQUIRES: peano

// RUN: aiecc --get-xclbin %s
// RUN: FileCheck %s --input-file=cpp_partition_npu2_5col.mlir.prj/partition_main.json

// CHECK: "column_width": 5
// CHECK: "start_columns": [
// CHECK-NEXT: 0
// CHECK-NEXT: 1
// CHECK-NEXT: 2
// CHECK-NEXT: 3
// CHECK-NEXT: ]

module {
  aie.device(npu2_5col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @of(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    %core = aie.core(%tile_0_2) {
      aie.end
    }
    aie.runtime_sequence(%arg0 : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%arg0[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of, id = 0 : i64} : memref<16xi32>
    }
  }
}
