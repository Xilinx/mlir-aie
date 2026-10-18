// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// REQUIRES: peano
// RUN: aie-opt --aie-dma-tasks-to-npu %s -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: aie-translate --aie-npu-to-cpp %t.mlir > %t.h
// RUN: %host_clang -std=c++17 -I%S/../../../../include -DGEN_HDR='"%t.h"' \
// RUN:   %S/Inputs/runtime_bd_guards.cpp %host_link_flags -o %t.exe
// RUN: %t.exe

module {
  aie.device(npu2) {
    %mem = aie.tile(0, 1)
    %core = aie.tile(0, 2)
    %shim = aie.tile(0, 0)
    %bytes = aie.buffer(%mem) {address = 4096 : i32} : memref<4096xi8>
    %halves = aie.buffer(%core) {address = 4096 : i32} : memref<1024xi16>
    %words = aie.buffer(%mem) {address = 8192 : i32} : memref<1024xi32>

    // CHECK-LABEL: @mem_bytes
    // CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be in [1:524284]"
    // CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be a multiple of 4 elements (whole 4-byte granules)"
    // CHECK: aiex.npu.blockwrite_values
    // Includes the NPU2 internal memtile aperture (0x80000) and buffer base.
    // CHECK: %[[BYTE_ADDR:.*]] = arith.addi %{{.*}}, %c528384_i64 : i64
    // CHECK: cf.assert %{{.*}}, "a runtime DMA offset is not 4-byte aligned"
    // CHECK: %[[FITS:.*]] = arith.cmpi ule, %[[BYTE_ADDR]], %c2097151_i64 : i64
    // CHECK: cf.assert %[[FITS]], "a runtime DMA offset puts the buffer address past the tile's DMA address field"
    // CHECK: %[[WORD_ADDR:.*]] = arith.divui %[[BYTE_ADDR]],
    // CHECK: %[[WORD_ADDR32:.*]] = arith.trunci %[[WORD_ADDR]] : i64 to i32
    // CHECK: aiex.npu.maskwrite32(%{{.*}}, %[[WORD_ADDR32]],
    aie.runtime_sequence @mem_bytes(%offset: i32, %len: i32) {
      %task = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%bytes : memref<4096xi8> offset = %offset len = %len sizes = [4] strides = [1]) {bd_id = 0 : i32}
        aie.end
      }
    }

    // CHECK-LABEL: @core_halves
    // CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be in [1:32766]"
    // CHECK: cf.assert %{{.*}}, "a runtime DMA d0 size must be a multiple of 2 elements (whole 4-byte granules)"
    // CHECK: aiex.npu.blockwrite_values
    // CHECK: %[[CORE_BYTES:.*]] = arith.addi %{{.*}}, %c4096_i64 : i64
    // CHECK: cf.assert %{{.*}}, "a runtime DMA offset is not 4-byte aligned"
    // CHECK: %[[CORE_FITS:.*]] = arith.cmpi ule, %[[CORE_BYTES]], %c65535_i64 : i64
    // CHECK: cf.assert %[[CORE_FITS]], "a runtime DMA offset puts the buffer address past the tile's DMA address field"
    // CHECK: %[[CORE_WORDS:.*]] = arith.divui %[[CORE_BYTES]],
    // CHECK: %[[CORE_WORDS32:.*]] = arith.trunci %[[CORE_WORDS]] : i64 to i32
    // CHECK: arith.shli %[[CORE_WORDS32]],
    // CHECK: aiex.npu.maskwrite32
    aie.runtime_sequence @core_halves(%offset: i32, %len: i32) {
      %task = aiex.dma_configure_task(%core, MM2S, 0) {
        aie.dma_bd(%halves : memref<1024xi16> offset = %offset len = %len sizes = [2] strides = [1]) {bd_id = 0 : i32}
        aie.end
      }
    }

    aie.runtime_sequence @mem_words(%offset: i32) {
      %task = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%words : memref<1024xi32> offset = %offset len = 4) {bd_id = 0 : i32}
        aie.end
      }
    }

    aie.runtime_sequence @stride(%size: i64, %stride: i64) {
      %task = aiex.dma_configure_task(%mem, MM2S, 0) {
        aie.dma_bd(%words : memref<1024xi32> offset = 0 len = 64 sizes = [%size, 8] strides = [%stride, 1]) {bd_id = 0 : i32}
        aie.end
      }
    }

    aie.runtime_sequence @shim_bytes(%buf: memref<4096xi8>, %len: i32) {
      %task = aiex.dma_configure_task(%shim, MM2S, 0) {
        aie.dma_bd(%buf : memref<4096xi8> offset = 0 len = %len sizes = [4] strides = [1]) {bd_id = 0 : i32}
        aie.end
      }
    }
  }
}
