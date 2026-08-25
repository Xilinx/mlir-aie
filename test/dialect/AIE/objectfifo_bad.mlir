//===- objectfifo_bad.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: 'aie.objectfifo' op does not have enough depths specified for producer and for each consumer.

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile13 = aie.tile(1, 3)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of_0 (%tile12, {%tile13, %tile23}, [2, 2]) : !aie.objectfifo<memref<16xi32>>
}

// -----

// CHECK: custom op 'aie.objectfifo' initial values should initialize all objects

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 3 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1, 2, 3]> : memref<4xi32>]
}

// -----

// CHECK: custom op 'aie.objectfifo' initial value should be an elements attribute

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 1 : i32) : !aie.objectfifo<memref<4xi32>> = [[0, 1, 2, 3]]
}

// -----

// CHECK: inferred shape of elements literal ({{\[}}2, 3]) does not match type ({{\[}}2, 2])

aie.device(xcve2302) {
   %tile12 = aie.tile(1, 2)
   %tile23 = aie.tile(2, 3)

   aie.objectfifo @of0 (%tile12, {%tile23}, 2 : i32) : !aie.objectfifo<memref<2x2xi32>> = [dense<[[4, 5], [6, 7]]> : memref<2x2xi32>,
                                                                                          dense<[[0, 1, 2], [3, 4, 5]]> : memref<2x2xi32>]
}

// -----

// CHECK: producer element size (35) must be an integer multiple of consumer element size (10)

aie.device(npu2) {
   %tile01 = aie.tile(0, 1)
   %tile02 = aie.tile(0, 2)

   aie.objectfifo @of0 (%tile01, {%tile02}, 1 : i32) : !aie.objectfifo<memref<35xi32>> -> !aie.objectfifo<memref<10xi32>>
}

// -----

// CHECK: producer and consumer must have the same scalar element type

aie.device(npu2) {
   %tile01 = aie.tile(0, 1)
   %tile02 = aie.tile(0, 2)

   aie.objectfifo @of0 (%tile01, {%tile02}, 1 : i32) : !aie.objectfifo<memref<40xi32>> -> !aie.objectfifo<memref<10xf32>>
}

// -----

// CHECK: consumer element count must be positive

aie.device(npu2) {
   %tile01 = aie.tile(0, 1)
   %tile02 = aie.tile(0, 2)

   aie.objectfifo @of0 (%tile01, {%tile02}, 1 : i32) : !aie.objectfifo<memref<40xi32>> -> !aie.objectfifo<memref<0xi32>>
}

// -----

// The direction of a dangling end is read off the flow that names it, so a
// second flow would leave it ambiguous.

// CHECK: 'aie.objectfifo.dangling_endpoint' op is named 2 times by flows; an endpoint drives one channel

aie.device(npu1) {
  %tile02 = aie.tile(0, 2)
  %shim0 = aie.tile(0, 0)
  %shim1 = aie.tile(1, 0)

  aie.objectfifo.pool @p(%tile02) {depth = 2 : i32,
      segments = [#aie.objectfifo_segment<offset = 0, size = 16>]} : memref<16xi32>
  aie.objectfifo.core_endpoint @fill(%tile02) fills @p
  aie.objectfifo.dma_endpoint @drain(%tile02) drains @p

  aie.objectfifo.dangling_endpoint @mid(%shim1) DMA
  aie.objectfifo.dangling_endpoint @sink(%shim0) DMA

  aie.objectfifo.flow from @drain to [@mid]
  aie.objectfifo.flow from @mid to [@sink]
}

// -----

// Both ends of one flow is the same ambiguity.

// CHECK: 'aie.objectfifo.dangling_endpoint' op is named 2 times by flows; an endpoint drives one channel

aie.device(npu1) {
  %shim0 = aie.tile(0, 0)

  aie.objectfifo.dangling_endpoint @loop(%shim0) DMA

  aie.objectfifo.flow from @loop to [@loop]
}
