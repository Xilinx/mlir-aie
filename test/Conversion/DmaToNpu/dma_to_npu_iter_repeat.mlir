//===- dma_to_npu_iter_repeat.mlir ------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//

// Covers the new explicit iteration/repeat attributes on aiex.npu.dma_memcpy_nd
// through the aie-dma-to-npu lowering:
//   - iter_size/iter_stride  -> the BD iteration register (word 6 of the BD)
//   - repeat_count           -> the push-queue write32 (extra replays)
//
// For the iter BD: iter_size=4 -> wrap=3, iter_stride=64 i32 words -> step=63,
// packed as 0x0030003F = 3145791.
// For the repeat BD: repeat_count=3 is enqueued via the push-queue command word
// 0x00030001 = 196609 (repeat in the high half, enqueue bit in the low half).

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK: memref.global "private" constant @blockwrite_data_0 : memref<8xi32> = dense<[64, 0, 0, 0, -2147483648, 33554432, 3145791, 33554432]>
// CHECK: memref.global "private" constant @blockwrite_data_1 : memref<8xi32> = dense<[64, 0, 0, 0, -2147483648, 33554432, 0, 33554432]>
// CHECK: aiex.npu.write32 {address = 119316 : ui32, value = 0 : ui32}
// CHECK: aiex.npu.write32 {address = 119316 : ui32, value = 196609 : ui32}

module {
  aie.device(npu1) {
    aie.runtime_sequence(%arg0: memref<2048xi32>) {
      // Strided BD iteration via explicit iter_* attributes.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0][1, 1, 64][0, 0, 1]) {id = 0 : i64, metadata = @md, iter_size = 4 : i64, iter_stride = 64 : i64} : memref<2048xi32>
      // Pure repeat (no address increment) via repeat_count: 3 extra replays.
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0][1, 1, 64][0, 0, 1]) {id = 1 : i64, metadata = @md, repeat_count = 3 : i32} : memref<2048xi32>
    }
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @md (%tile_0_0, MM2S, 0)
  }
}
