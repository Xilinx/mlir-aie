//===- bfp.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-transform-bfp-types -split-input-file %s | FileCheck %s

module {
  aie.device(npu2) {
    // This is checking proper conversion of attributes and memrefs
    // CHECK: func.func private @bfp16_passthrough_vectorized(memref<16xi72>, memref<16xi72>)
    func.func private @bfp16_passthrough_vectorized(memref<16x!aiex.bfp<"v8bfp16ebs8">>, memref<16x!aiex.bfp<"v8bfp16ebs8">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    // Proper conversion of objectfifo
    // CHECK: : !aie.objectfifo<memref<16xi72>>
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
    %core_1_2 = aie.core(%tile_1_2) {
        // Proper conversion of objectfifosubview
        // CHECK: : !aie.objectfifosubview<memref<16xi72>>
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
      aie.end
    }
  }
}

// -----

module {
  aie.device(npu2) {
    // This is checking proper conversion of attributes and memrefs
    // CHECK: func.func private @bfp16_passthrough_vectorized(memref<16xi136>, memref<16xi136>)
    func.func private @bfp16_passthrough_vectorized(memref<16x!aiex.bfp<"v16bfp16ebs16">>, memref<16x!aiex.bfp<"v16bfp16ebs16">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    // Proper conversion of objectfifo
    // CHECK: : !aie.objectfifo<memref<16xi136>>
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v16bfp16ebs16">>>
    %core_1_2 = aie.core(%tile_1_2) {
        // Proper conversion of objectfifosubview
        // CHECK: : !aie.objectfifosubview<memref<16xi136>>
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"v16bfp16ebs16">>>
      aie.end
    }
  }
}

// -----

// CHECK: : memref<8xi72>
// CHECK: : memref<8xi72>

module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<8x!aiex.bfp<"v8bfp16ebs8">>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<8x!aiex.bfp<"v8bfp16ebs8">>, 0)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}
