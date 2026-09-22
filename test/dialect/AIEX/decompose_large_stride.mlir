//===- decompose_large_stride.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A stride past the BD's step field is sliced one index at a time, the
// stride moving into each slice's offset. The shim's step field is 20 bits
// of granules (4 bytes), so a bf16 stride of 2^22 elements (2^21 granules)
// cannot be encoded; four blocks at that stride become four transfers.
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @strided_blocks
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[0, 0, 0, 0][1, 1, 64, 512][1, 0, 8192, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[4194304, 0, 0, 0][1, 1, 64, 512][1, 0, 8192, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[8388608, 0, 0, 0][1, 1, 64, 512][1, 0, 8192, 1])
// CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[12582912, 0, 0, 0][1, 1, 64, 512][1, 0, 8192, 1])
// CHECK-NOT: aiex.npu.dma_memcpy_nd
module {
  aie.device(npu2_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @b (%t, MM2S, 0)
    aie.runtime_sequence @strided_blocks(%B: memref<16777216xbf16>) {
      aiex.npu.dma_memcpy_nd(%B[0, 0, 0, 0][4, 1, 64, 512][4194304, 0, 8192, 1]) {id = 0 : i64, metadata = @b} : memref<16777216xbf16>
    }
  }
}
