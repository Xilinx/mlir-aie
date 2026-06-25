//===- dma_to_npu_width_conversion.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
//
// Date: July 3rd 2023
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:  aie.device(xcve2302) {
// CHECK:     aie.runtime_sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
// CHECK:       %0 = memref.get_global @blockwrite_data_0 : memref<8xi32>
// CHECK:       aiex.npu.blockwrite(%0) {address = 67227648 : ui32} : memref<8xi32>
// CHECK:       aiex.npu.address_patch(%{{.*}} : i32) {addr = 67227652 : ui32, arg_idx = 0 : i32}
// CHECK-DAG:       %[[V:.*]] = arith.constant -2147287040 : i32
// CHECK-DAG:       %[[A:.*]] = arith.constant 67228164 : i32
// CHECK:       aiex.npu.write32(%[[A]], %[[V]]) : i32, i32
// CHECK:       aiex.npu.sync(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : i32, i32, i32, i32, i32, i32
// CHECK:     }
// CHECK:     aie.shim_dma_allocation @toMem(%shim_noc_tile_2_0, S2MM, 0)
// CHECK:   }
// CHECK: }

module @shimDmaMemcpy{
  aie.device(xcve2302) {
    aie.runtime_sequence(%arg0: memref<65536xbf16>, %arg1: memref<65536xbf16>, %arg2: memref<65536xbf16>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][4, 4, 64, 64][0, 64, 256, 1]) {id = 0 : i64, metadata = @toMem} : memref<65536xbf16>
      %cst_npu_0 = arith.constant 0 : i32
      %cst_npu_1 = arith.constant 0 : i32
      %cst_npu_2 = arith.constant 0 : i32
      %cst_npu_3 = arith.constant 0 : i32
      %cst_npu_4 = arith.constant 1 : i32
      %cst_npu_5 = arith.constant 1 : i32
      aiex.npu.sync(%cst_npu_0, %cst_npu_1, %cst_npu_2, %cst_npu_3, %cst_npu_4, %cst_npu_5) : i32, i32, i32, i32, i32, i32
    }
    %tile_2_0 = aie.tile(2, 0)
    aie.shim_dma_allocation @toMem (%tile_2_0, S2MM, 0)
  }
}

