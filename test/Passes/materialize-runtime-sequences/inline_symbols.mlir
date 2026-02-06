//===- inline_symbols.mlir -------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-materialize-runtime-sequences %s | FileCheck %s

// Test that symbol definitions (like shim_dma_allocation) are properly preserved
// when inlining runtime sequences from other devices

module {
  aie.device(npu2) @main {
    %tile00 = aie.tile(0, 0)
    
    // CHECK-LABEL: aie.runtime_sequence @main_seq
    aie.runtime_sequence @main_seq(%arg0: memref<64xi32>) {
      // CHECK: aiex.npu.load_pdi {device_ref = @config_with_symbols}
      // CHECK: aiex.npu.dma_memcpy_nd(%[[ARG0:.*]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1])
      // CHECK-SAME: metadata = @buffer_in
      aiex.configure @config_with_symbols {
        aiex.run @seq_with_dma(%arg0) : (memref<64xi32>)
      }
    }
  }
  
  // CHECK: aie.device(npu2) @config_with_symbols
  aie.device(npu2) @config_with_symbols {
    %tile20 = aie.tile(2, 0)
    // CHECK: aie.shim_dma_allocation @buffer_in
    aie.shim_dma_allocation @buffer_in(%tile20, S2MM, 0)
    
    aie.runtime_sequence @seq_with_dma(%arg0: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) { metadata = @buffer_in, id = 0 : i64 } : memref<64xi32>
    }
  }
}
