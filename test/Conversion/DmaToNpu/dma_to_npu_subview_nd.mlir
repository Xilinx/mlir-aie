//===- dma_to_npu_subview_nd.mlir -------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --aie-dma-to-npu %s | FileCheck %s

// Exercises traceSubviewToBlockArgument on multi-dimensional, including
// rank-reducing, memref.subview chains feeding aiex.npu.dma_memcpy_nd.

// 2D -> 1D rank-reducing subview: row 1 of a memref<2x256xi32>.
// Byte offset = 1 * 256 * 4 = 1024.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-SAME: memref<2x256xi32>
// CHECK: %[[AP0:.+]] = arith.constant 1024 : i32
// CHECK: aiex.npu.address_patch(%[[AP0]] : i32)
// CHECK-SAME: arg_idx = 0 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<2x256xi32>) {
      %row = memref.subview %arg0[1, 0] [1, 256] [1, 1]
        : memref<2x256xi32> to memref<256xi32, strided<[1], offset: 256>>
      %row_cast = memref.reinterpret_cast %row to offset: [0], sizes: [256], strides: [1]
        : memref<256xi32, strided<[1], offset: 256>> to memref<256xi32>
      aiex.npu.dma_memcpy_nd (%row_cast[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1])
        { metadata = @buffer_2d_1d, id = 0 : i64 } : memref<256xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer_2d_1d (%tile_0_0, S2MM, 0)
  }
}

// -----

// 3D -> 1D rank-reducing subview: pick (slice 2, row 3, all cols) of a
// memref<4x8x32xi32>. Byte offset = (2 * 8 * 32 + 3 * 32) * 4 = 2432.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-SAME: memref<4x8x32xi32>
// CHECK: %[[AP1:.+]] = arith.constant 2432 : i32
// CHECK: aiex.npu.address_patch(%[[AP1]] : i32)
// CHECK-SAME: arg_idx = 0 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<4x8x32xi32>) {
      %row = memref.subview %arg0[2, 3, 0] [1, 1, 32] [1, 1, 1]
        : memref<4x8x32xi32> to memref<32xi32, strided<[1], offset: 608>>
      %row_cast = memref.reinterpret_cast %row to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 608>> to memref<32xi32>
      aiex.npu.dma_memcpy_nd (%row_cast[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1])
        { metadata = @buffer_3d_1d, id = 0 : i64 } : memref<32xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer_3d_1d (%tile_0_0, S2MM, 0)
  }
}

// -----

// 4D -> 1D rank-reducing subview through three dropped dims.
// memref<2x4x8x32xi32>, slice [1, 2, 3, 0] of size [1, 1, 1, 32].
// Byte offset = (1*4*8*32 + 2*8*32 + 3*32) * 4 = 1632 * 4 = 6528.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-SAME: memref<2x4x8x32xi32>
// CHECK: %[[AP2:.+]] = arith.constant 6528 : i32
// CHECK: aiex.npu.address_patch(%[[AP2]] : i32)
// CHECK-SAME: arg_idx = 0 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<2x4x8x32xi32>) {
      %row = memref.subview %arg0[1, 2, 3, 0] [1, 1, 1, 32] [1, 1, 1, 1]
        : memref<2x4x8x32xi32> to memref<32xi32, strided<[1], offset: 1632>>
      %row_cast = memref.reinterpret_cast %row to offset: [0], sizes: [32], strides: [1]
        : memref<32xi32, strided<[1], offset: 1632>> to memref<32xi32>
      aiex.npu.dma_memcpy_nd (%row_cast[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1])
        { metadata = @buffer_4d_1d, id = 0 : i64 } : memref<32xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer_4d_1d (%tile_0_0, S2MM, 0)
  }
}

// -----

// 2D -> 1D rank-reducing subview with bf16 element type.
// memref<2x512xbf16>, row 1. Byte offset = 1 * 512 * 2 = 1024.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-SAME: memref<2x512xbf16>
// CHECK: %[[AP3:.+]] = arith.constant 1024 : i32
// CHECK: aiex.npu.address_patch(%[[AP3]] : i32)
// CHECK-SAME: arg_idx = 0 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<2x512xbf16>) {
      %row = memref.subview %arg0[1, 0] [1, 512] [1, 1]
        : memref<2x512xbf16> to memref<512xbf16, strided<[1], offset: 512>>
      %row_cast = memref.reinterpret_cast %row to offset: [0], sizes: [512], strides: [1]
        : memref<512xbf16, strided<[1], offset: 512>> to memref<512xbf16>
      aiex.npu.dma_memcpy_nd (%row_cast[0, 0, 0, 0][1, 1, 1, 512][0, 0, 0, 1])
        { metadata = @buffer_bf16, id = 0 : i64 } : memref<512xbf16>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer_bf16 (%tile_0_0, S2MM, 0)
  }
}

// -----

// Chained 3D -> 2D rank-reducing subview, then 2D -> 1D rank-reducing subview.
// memref<2x4x256xi32> arg, take outer index 1 (offset 4*256*4=4096),
// then take row 2 (offset 2*256*4=2048). Total = 6144 bytes.

// CHECK-LABEL: aie.runtime_sequence
// CHECK-SAME: memref<2x4x256xi32>
// CHECK: %[[AP4:.+]] = arith.constant 6144 : i32
// CHECK: aiex.npu.address_patch(%[[AP4]] : i32)
// CHECK-SAME: arg_idx = 0 : i32
module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<2x4x256xi32>) {
      %plane = memref.subview %arg0[1, 0, 0] [1, 4, 256] [1, 1, 1]
        : memref<2x4x256xi32> to memref<4x256xi32, strided<[256, 1], offset: 1024>>
      %row = memref.subview %plane[2, 0] [1, 256] [1, 1]
        : memref<4x256xi32, strided<[256, 1], offset: 1024>>
        to memref<256xi32, strided<[1], offset: 1536>>
      %row_cast = memref.reinterpret_cast %row to offset: [0], sizes: [256], strides: [1]
        : memref<256xi32, strided<[1], offset: 1536>> to memref<256xi32>
      aiex.npu.dma_memcpy_nd (%row_cast[0, 0, 0, 0][1, 1, 1, 256][0, 0, 0, 1])
        { metadata = @buffer_chain, id = 0 : i64 } : memref<256xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @buffer_chain (%tile_0_0, S2MM, 0)
  }
}
