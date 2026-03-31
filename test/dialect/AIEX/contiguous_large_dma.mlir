//===- contiguous_large_dma.mlir --------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Positive tests: verify that contiguous ND accesses with d0 or d1 > 1023
// are accepted and correctly canonicalized / lowered to linear mode.
//
// These complement the negative tests in bad_npu_nd.mlir.  Contiguous
// patterns are those where innermost stride == 1 and each outer stride equals
// the product of inner sizes; they are linearized by the compiler so the 10-
// bit ND wrap-size limit does not apply.
//
//===----------------------------------------------------------------------===//

// -----
// Test 1: NpuDmaMemcpyNdOp with d0 > 1023 (contiguous) is accepted and folded.
//
// RUN: aie-opt --canonicalize --split-input-file %s | FileCheck %s --check-prefix=CANON

// CANON-LABEL: aie.device(npu1)
// CANON:         aie.runtime_sequence @large_d0_contiguous
// CANON:           aiex.npu.dma_memcpy_nd
// CANON-SAME:        [0, 0, 0, 0][1, 1, 1, 2073600][0, 0, 0, 1]
module {
  aie.device(npu1) {
    // 1080 rows x 1920 i32 words/row = 2073600 elements.
    // d0 = 1920, d1 = 1080: both exceed 1023 but the access is contiguous.
    aie.runtime_sequence @large_d0_contiguous(%arg0 : memref<2073600xi32>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1080, 1920][0, 0, 1920, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<2073600xi32>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----
// Test 2: NpuDmaMemcpyNdOp contiguous with i8 (lineWidthInBytes > 1023).
//         Motivating case: 1920x1080 RGBA image as int8.
//         d0 = 7680 i8 elements = 1920 in 32-bit word units, d1 = 1080.
//
// CANON-LABEL: aie.device(npu1)
// CANON:         aie.runtime_sequence @large_i8_image
// CANON:           aiex.npu.dma_memcpy_nd
// CANON-SAME:        [0, 0, 0, 0][1, 1, 1, 8294400][0, 0, 0, 1]
module {
  aie.device(npu1) {
    aie.runtime_sequence @large_i8_image(%arg0 : memref<8294400xi8>) {
      aiex.npu.dma_memcpy_nd (%arg0[0, 0, 0, 0][1, 1, 1080, 7680][0, 0, 7680, 1])
        { metadata = @of_fromMem, id = 0 : i64 } : memref<8294400xi8>
    }
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%tile, MM2S, 0)
  }
}

// -----
// Test 3: After aie-dma-tasks-to-npu, a contiguous shim BD with d0/d1 > 1023
//         is lowered to linear mode (d0_wrap=0 and d1_wrap=0 in the output BD
//         configuration instruction, meaning the hardware uses buffer_length).
//
// RUN: aie-opt --pass-pipeline='any(aie.device(aie-dma-tasks-to-npu))' \
// RUN:   --split-input-file %s | FileCheck %s --check-prefix=LOWER

// LOWER-LABEL: aie.device(npu1)
// LOWER:         aie.runtime_sequence @tasks_to_npu_large
// -- aiex.npu.writebd is emitted (the BD was lowered to hardware registers):
// LOWER:           aiex.npu.writebd
// -- Linear mode: d0_size and d1_size must both be 0 (unused in linear mode);
// -- buffer_length carries the full 2073600-element transfer count.
// LOWER-SAME:        buffer_length = 2073600
// LOWER-SAME:        d0_size = 0
// LOWER-SAME:        d1_size = 0
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @of_fromMem (%shim_noc_tile_0_0, MM2S, 0)
    aie.runtime_sequence @tasks_to_npu_large(%arg0: memref<2073600xi32>) {
      %0 = aiex.dma_configure_task(%shim_noc_tile_0_0, MM2S, 0) {
        // 1080 x 1920 i32: d0=1920 > 1023, d1=1080 > 1023, contiguous.
        // aie-dma-tasks-to-npu must lower to linear mode (d0_size=d1_size=0).
        aie.dma_bd(%arg0 : memref<2073600xi32>, 0, 2073600,
          [<size = 1080, stride = 1920>, <size = 1920, stride = 1>])
          {bd_id = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_await_task(%0)
    }
  }
}
