//===- cpp_partition_npu1.mlir -----------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Tests that partition JSON has column_width=4 for npu1 (full 4-col).

// REQUIRES: peano

// RUN: aiecc --no-xchesscc --no-xbridge --aie-generate-xclbin -n %s
// RUN: FileCheck %s --input-file=cpp_partition_npu1.mlir.prj/main_aie_partition.json

// CHECK: "column_width": 4
// CHECK: "start_columns": [
// CHECK: 0

module {
  aie.device(npu1) {
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
