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
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-decompose-large-dma-bd))' --split-input-file %s | FileCheck %s

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

    // -----

    // Peel the legal outer dimension before slicing the oversized inner stride.
    // Nonzero offsets must retain their contribution to each transfer's address.
    // CHECK-LABEL: aie.runtime_sequence @inner_strided_blocks
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[1, 0, 4194304, 3][1, 1, 1, 2][64, 0, 1, 1])
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[1, 0, 6291456, 3][1, 1, 1, 2][64, 0, 1, 1])
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[2, 0, 4194304, 3][1, 1, 1, 2][64, 0, 1, 1])
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[2, 0, 6291456, 3][1, 1, 1, 2][64, 0, 1, 1])
    // CHECK-NOT: aiex.npu.dma_memcpy_nd
    module {
      aie.device(npu2_1col) {
        %t = aie.tile(0, 0)
        aie.shim_dma_allocation @b (%t, MM2S, 0)
        aie.runtime_sequence @inner_strided_blocks(%B: memref<8388608xi32>) {
          aiex.npu.dma_memcpy_nd(%B[1, 0, 2, 3][2, 1, 2, 2][64, 0, 2097152, 1]) {id = 0 : i64, metadata = @b} : memref<8388608xi32>
        }
      }
    }

    // -----

    // A wrap-limit split with a singleton tail must retain its legal stride.
    // CHECK-LABEL: aie.runtime_sequence @singleton_wrap_tail
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[0, 0, 0, 0][64, 1, 1, 2][3, 0, 0, 1])
    // CHECK: aiex.npu.dma_memcpy_nd(%{{.*}}[64, 0, 0, 0][1, 1, 1, 2][3, 0, 0, 1])
    // CHECK-NOT: aiex.npu.dma_memcpy_nd
    module {
      aie.device(npu2_1col) {
        %t = aie.tile(0, 0)
        aie.shim_dma_allocation @b (%t, MM2S, 0)
        aie.runtime_sequence @singleton_wrap_tail(%B: memref<256xi32>) {
          aiex.npu.dma_memcpy_nd(%B[0, 0, 0, 0][65, 1, 1, 2][3, 0, 0, 1]) {id = 0 : i64, metadata = @b} : memref<256xi32>
        }
      }
    }
  }
}
