//===- bfp-invalid.mlir ----------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-transform-bfp-types -split-input-file %s 2>&1 | FileCheck %s

// CHECK: Block type v16bfp16ebs16 is not supported in the specified model

module {
  aie.device(npu1) {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v16bfp16ebs16">>>
  }
}

// -----

// CHECK: Block type v8bfp16ebs8 is not supported in the specified model

// CHECK: Failed to convert memref element type
// CHECK: Failed to convert function input types
// CHECK: Failed to convert attribute type
// CHECK: Failed to convert result type
// There are currently no operands dependent on bfp types, even though they are checked for in the conversion

module {
  aie.device(npu1) {
    func.func private @eltwise_add_float(memref<16x!aiex.bfp<"v8bfp16ebs8">>, memref<16x!aiex.bfp<"v8bfp16ebs8">>, memref<16x!aiex.bfp<"v8bfp16ebs8">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
    %core_1_2 = aie.core(%tile_1_2) {
      %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"v8bfp16ebs8">>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"v8bfp16ebs8">>> -> memref<16x!aiex.bfp<"v8bfp16ebs8">>
      aie.end
    }
  }
}

// -----

// CHECK: Invalid block type: v32bfp16ebz8. Known types are: v8bfp16ebs8, v16bfp16ebs16.

module {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(2, 2)

    aiex.runtime_sequence(%arg0: memref<8x!aiex.bfp<"v32bfp16ebz8">>, %arg1: memref<10x!aiex.bfp<"v8bfp16ebs8">>) {
      %t1 = aiex.dma_configure_task(%tile_0_0, MM2S, 0) {
        aie.dma_bd(%arg0 : memref<8x!aiex.bfp<"v8bfp16ebs8">>, 0, 8) {bd_id = 7 : i32}
        aie.end
      } {issue_token = true}
    }
  }
}

// -----

// CHECK: 'aie.dma_bd' op transfer length must be multiple of 4 (i.e., represent 4 byte aligned address)
module {
  aie.device(npu2) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %lock_0_1 = aie.lock(%tile_0_1) {init = 1 : i32}
      %lock_0_1_0 = aie.lock(%tile_0_1) {init = 0 : i32}
      %buffer_0_1 = aie.buffer(%tile_0_1) {address = 0 : i32} : memref<7x!aiex.bfp<"v8bfp16ebs8">>
      %0 = aie.dma(S2MM, 0) [{
        aie.use_lock(%lock_0_1, AcquireGreaterEqual)
        aie.dma_bd(%buffer_0_1 : memref<7x!aiex.bfp<"v8bfp16ebs8">>, 0)
        aie.use_lock(%lock_0_1_0, Release)
      }]
      aie.end
    }
  }
}
