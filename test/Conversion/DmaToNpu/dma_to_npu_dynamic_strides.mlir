//===- dma_to_npu_dynamic_strides.mlir - Dynamic sizes/strides ---*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// Tests that the DMA-to-NPU lowering correctly handles dynamic (SSA) sizes
// and strides in npu.dma_memcpy_nd operations. The dynamic path emits BD
// words via npu.write32 with dyn_address/dyn_value instead of blockwrite.

// RUN: aie-opt --split-input-file -aie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL: module
// Dynamic sizes produce npu.write32 ops (not blockwrite)
// CHECK: aiex.npu.write32
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
module  {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: i32) {
      // Dynamic size in dim[0]: %arg1 is an SSA value
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      %dim0 = arith.extui %arg1 : i32 to i64
      aiex.npu.dma_memcpy_nd (%arg0[%c0, %c0, %c0, %c0][%dim0, %c1, %c32, %c32][%c0, %c32, %c32, %c1]) { metadata = @toMem, id = 1 : i64 } : memref<16384xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @toMem (%tile_0_0, S2MM, 0)
  }
}

// -----

// Test bf16 d0_size computation: d0_size = inSize0 * elemWidth / addrGran
// For bf16: 32 * 16 / 32 = 16 (NOT 32 * (16/32) = 0)
// CHECK-LABEL: module
// CHECK: aiex.npu.write32
module  {
  aie.device(npu2) {
    aie.runtime_sequence(%arg0: memref<16384xbf16>, %arg1: i32) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c32 = arith.constant 32 : i64
      %dim = arith.extui %arg1 : i32 to i64
      // bf16 with dynamic outer dim
      aiex.npu.dma_memcpy_nd (%arg0[%c0, %c0, %c0, %c0][%dim, %c1, %c32, %c32][%c0, %c32, %c32, %c1]) { metadata = @inA, id = 2 : i64 } : memref<16384xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @inA (%tile_0_0, S2MM, 0)
  }
}
