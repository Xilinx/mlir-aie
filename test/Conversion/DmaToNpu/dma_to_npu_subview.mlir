//===- dma_to_npu_subview.mlir ---------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// Test that memref.subview is correctly traced through to adjust address patches

// CHECK-LABEL: module
// CHECK: aiex.npu.address_patch
// CHECK-SAME: arg_idx = 0 : i32
// CHECK-SAME: arg_plus = 512 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<1024xi32>) {
      // Create a subview at offset 128 elements (512 bytes for i32)
      %subview = memref.subview %arg0[128] [256] [1] : memref<1024xi32> to memref<256xi32, strided<[1], offset: 128>>
      %reinterpret = memref.reinterpret_cast %subview to offset: [0], sizes: [256], strides: [1] : memref<256xi32, strided<[1], offset: 128>> to memref<256xi32>
      // DMA should use the reinterpreted memref, and the pass should trace back to %arg0 with offset
      aiex.npu.dma_memcpy_nd (%reinterpret[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1]) { metadata = @buffer, id = 0 : i64 } : memref<256xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer (%tile_0_0, S2MM, 0)
  }
}
