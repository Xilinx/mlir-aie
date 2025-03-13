//===- dma_to_npu_burst_length.mlir -----------------------------------------*- MLIR -*-===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-dma-to-npu %s | FileCheck %s

// Note that the writebd operation is being tested in this file too, through dma_memcpy_nd's conversion in the pass, so no need for a explicit check.

module {
  aie.device(npu2) {
    memref.global "public" @of_fromMem : memref<32xi32>
    memref.global "public" @of_toMem : memref<32xi32>
    aiex.runtime_sequence(%in : memref<4x2x8xi32>, %buf : memref<32xi32>, %out : memref<64xi32>) {
        // Note that The burst length encoding is mixed with the stride, so the encoding does not exactly correspond to the burst length.
        // CHECK: memref.global "private" constant {{.*}} = dense<[{{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, 2097152, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}]>
        aiex.npu.dma_memcpy_nd (%in[0,2,0,0][1,2,2,8][0,16,1,1]) { metadata = @of_fromMem, id = 0 : i64, burst_length = 64 : i64} : memref<4x2x8xi32>
        // Here the burst length encoding is not mixed with the stride, since the stride is 0. This is 0xC0000000.
        // CHECK: memref.global "private" constant {{.*}} = dense<[{{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}, -1073741824, {{[0-9]*}}, {{[0-9]*}}, {{[0-9]*}}]>
        aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,32][0,0,0,1]) { metadata = @of_toMem, id = 1 : i64, burst_length = 512 : i64} : memref<64xi32>
    }
    aie.shim_dma_allocation @of_fromMem (MM2S, 0, 0)
    aie.shim_dma_allocation @of_toMem (S2MM, 0, 0)
  }
}
