//===- bfp.mlir ----------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-transform-bfp-types %s -split-input-file 2>&1 | FileCheck %s

// CHECK: Block type bfp16ebs16 is not supported in the specified model

module {
  aie.device(npu1) {
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs16">>>
  }
}

// -----

// CHECK: Block type bfp16ebs8 is not supported in the specified model

// CHECK: Failed to convert memref element type
// CHECK: Failed to convert function input types
// CHECK: Failed to convert attribute type
// CHECK: Failed to convert result type
// There are currently no operands dependent on bfp types, even though the checks for conversion

module {
  aie.device(npu1) {
    func.func private @eltwise_add_float(memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs8">>>
    aie.objectfifo @in2(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs8">>>
    aie.objectfifo @out(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs8">>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>> -> memref<16x!aiex.bfp<"bfp16ebs8">>
        %2 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>> -> memref<16x!aiex.bfp<"bfp16ebs8">>
        %4 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>> -> memref<16x!aiex.bfp<"bfp16ebs8">>
        func.call @eltwise_add_float(%1, %3, %5) : (memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>) -> ()
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @in2(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<256x!aiex.bfp<"bfp16ebs8">>, %arg1: memref<256x!aiex.bfp<"bfp16ebs8">>, %arg2: memref<256x!aiex.bfp<"bfp16ebs8">>) {
      %0 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<256x!aiex.bfp<"bfp16ebs8">>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%arg1 : memref<256x!aiex.bfp<"bfp16ebs8">>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<256x!aiex.bfp<"bfp16ebs8">>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}
