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
    func.func private @bfp16_passthrough_vectorized(memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    // Proper conversion of objectfifo
    // CHECK: aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi72>>
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs8">>>
    aie.objectfifo @out(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs8">>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        // Proper conversion of objectfifosubview
        // CHECK: %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16xi72>>
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>> -> memref<16x!aiex.bfp<"bfp16ebs8">>
        %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs8">>> -> memref<16x!aiex.bfp<"bfp16ebs8">>
        func.call @bfp16_passthrough_vectorized(%1, %3) : (memref<16x!aiex.bfp<"bfp16ebs8">>, memref<16x!aiex.bfp<"bfp16ebs8">>) -> ()
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<32x!aiex.bfp<"bfp16ebs8">>, %arg1: memref<32x!aiex.bfp<"bfp16ebs8">>) {
      %0 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<32x!aiex.bfp<"bfp16ebs8">>, 0, 32, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<32x!aiex.bfp<"bfp16ebs8">>, 0, 32, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

// -----

module {
  aie.device(npu2) {
    // This is checking proper conversion of attributes and memrefs
    // CHECK: func.func private @bfp16_passthrough_vectorized(memref<16xi136>, memref<16xi136>)
    func.func private @bfp16_passthrough_vectorized(memref<16x!aiex.bfp<"bfp16ebs16">>, memref<16x!aiex.bfp<"bfp16ebs16">>)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    // Proper conversion of objectfifo
    // CHECK: aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16xi136>>
    aie.objectfifo @in1(%shim_noc_tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs16">>>
    aie.objectfifo @out(%tile_1_2, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<16x!aiex.bfp<"bfp16ebs16">>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c92233144036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c92233144036854775807 step %c1 {
        // Proper conversion of objectfifosubview
        // CHECK: %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16xi136>>
        %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs16">>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs16">>> -> memref<16x!aiex.bfp<"bfp16ebs16">>
        %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs16">>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16x!aiex.bfp<"bfp16ebs16">>> -> memref<16x!aiex.bfp<"bfp16ebs16">>
        func.call @bfp16_passthrough_vectorized(%1, %3) : (memref<16x!aiex.bfp<"bfp16ebs16">>, memref<16x!aiex.bfp<"bfp16ebs16">>) -> ()
        aie.objectfifo.release @in1(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence @sequence(%arg0: memref<32x!aiex.bfp<"bfp16ebs16">>, %arg1: memref<32x!aiex.bfp<"bfp16ebs16">>) {
      %0 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<32x!aiex.bfp<"bfp16ebs16">>, 0, 32, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg1 : memref<32x!aiex.bfp<"bfp16ebs16">>, 0, 32, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}
